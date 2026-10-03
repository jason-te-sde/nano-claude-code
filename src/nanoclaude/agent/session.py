"""One conversation, wired to everything it needs.

This is the only module that awaits a provider, and therefore the only one that
implements the retry policy from spec 7.4. It is also where compaction is
decided, because the budget verdict depends on the assembled context, which
depends on the model's window, which depends on the role -- all of which meet
here and nowhere else.

What it still does not know about is the CLI. Confirmation, progress and retry
notices go out through the UI protocol, so the same class backs the REPL,
headless mode and every test.

Two things are true of a session whatever happened to its last turn, so nothing
above this module has to repair one:

* a prompt can always follow. A turn cut short by Ctrl+C or a provider error can
  leave calls nobody answered, or a message the model never replied to, and
  either makes the next request invalid. They are closed when the next
  :meth:`Session.follow_up` or :meth:`Session.compact` arrives.
* the store holds the transcript the model is shown. Compaction and ``/clear``
  replace a transcript rather than extend it, so what was stored is compared
  with what is in memory each time and rewritten when they stop agreeing.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path

from nanoclaude.agent import loop
from nanoclaude.agent.executor import Executor
from nanoclaude.agent.loop import Done, LoopState
from nanoclaude.agent.router import Router
from nanoclaude.agent.ui import UI, AutoDecline
from nanoclaude.config.schema import ROLES, Config
from nanoclaude.context.assemble import assemble
from nanoclaude.context.mentions import expand_mentions
from nanoclaude.conversation.budget import Budget, HeuristicCounter
from nanoclaude.conversation.compaction import SUMMARY_TEMPLATE, full_compact, micro_compact
from nanoclaude.conversation.store import Store
from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ToolUseBlock,
    Transcript,
    assistant_text,
    user_text,
)
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.policy import Policy
from nanoclaude.permissions.redact import Redactor
from nanoclaude.providers.base import ModelError, ModelReply, ModelRequest, ToolSpec, Usage
from nanoclaude.providers.capabilities import Capabilities
from nanoclaude.providers.retry import RetryPolicy, with_retry
from nanoclaude.providers.texttools import MAX_PARSE_RETRIES, parse_reply, render_tools
from nanoclaude.tools.base import ToolOutcome
from nanoclaude.tools.registry import ToolRegistry
from nanoclaude.tools.todo import TodoState

#: What a call that never reported back is told. It may have run: a write can
#: land in the instant before the person stops the turn.
_INTERRUPTED_CALL = (
    "Interrupted: the turn was stopped before this call reported back, so it may or may "
    "not have run. Check the current state before relying on it or repeating it."
)

#: Stands where the reply a stopped turn never got would have been.
_INTERRUPTED_TURN = "[this turn was interrupted before it finished]"


def new_session_id() -> str:
    """A short random identifier for a session: twelve hexadecimal characters."""
    return uuid.uuid4().hex[:12]


def _correction(complaints: Sequence[str]) -> str:
    """What a model that wrote no usable tool call is told: what failed, then what to do."""
    listed = "\n".join(f"- {complaint}" for complaint in complaints)
    return (
        f"Your last reply contained no tool call that could be run:\n{listed}\n"
        'Write each call again as <tool name="Name"> followed by one JSON object and '
        "</tool>, or answer without a tool call if you do not need one."
    )


def _tools_text(specs: Sequence[ToolSpec]) -> str:
    """The tool definitions as a provider receives them, for counting."""
    return json.dumps(
        [
            {"name": spec.name, "description": spec.description, "input_schema": dict(spec.schema)}
            for spec in specs
        ]
    )


@dataclass
class Session:
    """One conversation with a model, from the first prompt until it is closed.

    Built from the pieces it works through: a router for the models, a registry of
    tools and a policy for what they may touch. :meth:`run` starts a conversation
    and :meth:`follow_up` continues it. ``ui`` is the only way it reaches a person.
    ``store`` and ``audit`` are optional, and a session without them keeps nothing
    on disk. The attributes are plain and mutable on purpose: a front end replaces
    ``state`` to clear a conversation.
    """

    root: str
    config: Config
    router: Router
    registry: ToolRegistry
    policy: Policy
    ui: UI = field(default_factory=AutoDecline)
    store: Store | None = None
    audit: AuditLog | None = None
    session_id: str = field(default_factory=new_session_id)
    todo_state: TodoState = field(default_factory=TodoState)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    redactor: Redactor = field(default_factory=Redactor)
    #: Where global instructions are looked for (``<home>/.nanoclaude/NANO.md``).
    #: A parameter rather than a read of the real home inside each turn, so that a
    #: session in a test cannot pick up the instructions of whoever runs it.
    home: str = field(default_factory=lambda: str(Path.home()))
    state: LoopState = field(init=False)
    executor: Executor = field(init=False)
    # The messages the store holds for this session, as of the last _persist().
    _stored: tuple[Message, ...] = field(default=(), init=False, repr=False)
    _call_ids: count[int] = field(default_factory=lambda: count(1), init=False, repr=False)

    def __post_init__(self) -> None:
        # Empty rather than a placeholder prompt: follow_up() is a valid first
        # call (the REPL makes it after /clear), and it must find the configured
        # turn limit and a transcript it can start.
        self.state = LoopState(Transcript(), turn=0, max_turns=self.config.limits.max_turns)
        if self.store is not None:
            self.store.create_session(
                self.session_id,
                cwd=self.root,
                roles={role: self.router.config.roles.alias_for(role) for role in ROLES},
            )
        self.executor = Executor(
            registry=self.registry,
            policy=self.policy,
            ui=self.ui,
            audit=self.audit,
            session_id=self.session_id,
            redactor=self.redactor,
        )

    @property
    def usage(self) -> Usage:
        """Everything every role has used so far."""
        return self.router.total_usage()

    @property
    def cost(self) -> float | None:
        """Dollars spent so far, or None when any model used has no price."""
        return self.router.total_cost()

    async def run(self, prompt: str) -> Done:
        """Start a new conversation with ``prompt`` and drive it until the model stops."""
        self.state = loop.start(self._expand(prompt), max_turns=self.config.limits.max_turns)
        return await self._drive()

    async def follow_up(self, prompt: str) -> Done:
        """Continue the conversation with ``prompt``."""
        self._mend()
        self.state = loop.resume(self.state, self._expand(prompt))
        return await self._drive()

    def _expand(self, prompt: str) -> str:
        text, _ = expand_mentions(
            prompt,
            root=self.root,
            redactor=self.redactor,
            allow_secrets=self.policy.allow_secrets,
        )
        return text

    async def _drive(self) -> Done:
        failures = 0  # replies in a row that held tool calls we could not parse, and no usable one
        while True:
            self._persist()
            capabilities = await self.router.capabilities_for("main")
            system, specs = self._prompt(capabilities)
            await self._maybe_compact(capabilities, system, specs)

            request = ModelRequest(
                system=system,
                transcript=self.state.transcript,
                tools=specs,
                max_output_tokens=capabilities.max_output,
            )
            raw = await self._complete("main", request)
            self.router.record("main", raw.usage, *self._adapter_and_model("main"))
            reply: ModelReply = raw
            complaints: tuple[str, ...] = ()
            if not capabilities.native_tools:
                reply, complaints = parse_reply(raw, next_id=self._next_call_id)
            self.ui.on_reply(reply)

            if complaints and not _has_calls(reply) and failures < MAX_PARSE_RETRIES:
                # Ask the same model again. What it said goes into the transcript as
                # it said it, and the complaints come back as the user's next message:
                # a tool_result cannot carry them, because there is no call to answer.
                failures += 1
                self.state = loop.resume(loop.step(self.state, raw).state, _correction(complaints))
                continue
            failures = 0

            outcome = loop.step(self.state, reply)
            self.state = outcome.state
            self._persist()
            if isinstance(outcome, Done):
                self._finish()
                return outcome

            results = await self.executor.run_batch(outcome.calls, self.state)
            self.state = loop.observe(self.state, results)

    def _prompt(self, capabilities: Capabilities) -> tuple[str, tuple[ToolSpec, ...]]:
        """The system prompt and the tool definitions the next request carries.

        One place, so the request and the budget are measured on the same text. A
        model without native tool calling gets the definitions inside the system
        prompt, as the protocol it is told to answer in, and none as tools.
        """
        specs = self.registry.specs()
        native = capabilities.native_tools
        context = assemble(
            self.root,
            cwd=self.root,
            home=self.home,
            tool_protocol=None if native else render_tools(specs),
        )
        return f"{context.system}\n\n{context.environment}", specs if native else ()

    async def _complete(self, role: str, request: ModelRequest) -> ModelReply:
        """One provider call under the retry policy; each retry is reported to the UI."""
        client = await self.router.client_for(role)

        async def once() -> ModelReply:
            return await client.complete(request)

        return await with_retry(once, policy=self.retry, on_retry=self.ui.on_retry)

    def _budget(self, capabilities: Capabilities) -> Budget:
        return Budget(
            capabilities,
            soft=self.config.limits.compact_soft,
            hard=self.config.limits.compact_hard,
        )

    async def _maybe_compact(
        self, capabilities: Capabilities, system: str, specs: tuple[ToolSpec, ...]
    ) -> None:
        budget = self._budget(capabilities)
        tools = _tools_text(specs)
        transcript = self.state.transcript
        budget.require_fits(transcript, system, tools)
        verdict = budget.verdict(transcript, system, tools)
        keep_recent = self.config.limits.keep_recent_turns
        if verdict == "micro":
            compacted = micro_compact(
                transcript, keep_recent=keep_recent, counter=HeuristicCounter()
            )
        elif verdict == "full":
            compacted = await full_compact(
                transcript, keep_recent=keep_recent, summarise=self._summarise
            )
        else:
            return
        self.state = replace(self.state, transcript=compacted)
        self._persist()

    async def compact(self, instructions: str | None = None) -> None:
        """Summarise the older part of the conversation now: the ``/compact`` command."""
        self._mend()
        extra = f"\n\nPay particular attention to: {instructions}" if instructions else ""

        async def summarise(text: str) -> str:
            return await self._summarise(text + extra)

        compacted = await full_compact(
            self.state.transcript,
            keep_recent=self.config.limits.keep_recent_turns,
            summarise=summarise,
        )
        self.state = replace(self.state, transcript=compacted)
        self._persist()

    async def _summarise(self, text: str) -> str:
        """Ask the compact role's model for a summary of ``text``.

        There is no fallback text. A summary that cannot be had stops the turn with
        the error, because the alternative is to replace the older conversation
        with a placeholder and carry on without it, which cannot be undone.
        """
        capabilities = await self.router.capabilities_for("compact")
        request = ModelRequest(
            system="You summarise coding sessions precisely and briefly.",
            transcript=Transcript((user_text(SUMMARY_TEMPLATE + text),)),
            tools=(),
            max_output_tokens=min(2048, capabilities.max_output),
        )
        try:
            reply = await self._complete("compact", request)
        except ModelError as error:
            raise ModelError(
                f"the compact role's model could not summarise the conversation: {error} "
                "— fix that model, or route the compact role elsewhere with --role compact=<model>",
                retryable=False,
                status=error.status,
            ) from error
        self.router.record("compact", reply.usage, *self._adapter_and_model("compact"))
        summary = "\n".join(b.text for b in reply.blocks if isinstance(b, TextBlock)).strip()
        if not summary:
            raise ModelError(
                "the compact role's model returned an empty summary, so the conversation was "
                "left as it was — try /compact again, or route the compact role to another "
                "model with --role compact=<model>"
            )
        return summary

    def _mend(self) -> None:
        """Close what an interrupted turn left open, so the next prompt can follow it."""
        pending = self.state.transcript.pending_tool_uses()
        if pending:
            self.state = loop.observe(
                self.state,
                [ToolOutcome(call.id, _INTERRUPTED_CALL, is_error=True) for call in pending],
            )
        last = self.state.transcript.last()
        if last is not None and last.role == "user":
            self.state = replace(
                self.state,
                transcript=self.state.transcript.append(assistant_text(_INTERRUPTED_TURN)),
            )

    def _adapter_and_model(self, role: str) -> tuple[str, str]:
        # The router decides which client answers a role, so it is also the
        # authority on which model to bill the call to.
        config = self.router.config
        entry = config.models[config.roles.alias_for(role)]
        return entry.adapter, entry.model

    def _next_call_id(self) -> str:
        return f"tt_{next(self._call_ids)}"

    def _persist(self) -> None:
        if self.store is None:
            return
        messages = self.state.transcript.messages
        stored = self._stored
        if messages[: len(stored)] == stored:
            for seq in range(len(stored), len(messages)):
                self.store.append_message(self.session_id, seq, messages[seq])
        else:
            # Not an extension of what was stored: compaction replaced the head, or
            # the transcript was swapped for another. Appending would leave the old
            # rows beside the new ones.
            self.store.replace_transcript(self.session_id, self.state.transcript)
        self._stored = messages

    def _finish(self) -> None:
        if self.store is not None:
            # An unknown cost is None here, and the store keeps it as NULL.
            self.store.finish_session(self.session_id, self.usage, self.cost)

    async def context_usage(self) -> tuple[int, int]:
        """(tokens used, tokens available) for /status. Estimated, not billed.

        Measured the way compaction measures it: against the capabilities the router
        resolves for the main role (the table, the cache, a probe, then the config's
        overrides), and counting the prompt a request carries as well as the
        conversation, so the figure is the one the thresholds act on.
        """
        capabilities = await self.router.capabilities_for("main")
        system, specs = self._prompt(capabilities)
        budget = self._budget(capabilities)
        used = budget.used(self.state.transcript, system, _tools_text(specs))
        return used, budget.available()

    async def aclose(self) -> None:
        """Close every client the router holds, then the store, even if a client would not close."""
        try:
            await self.router.aclose()
        finally:
            if self.store is not None:
                self.store.close()


def _has_calls(reply: ModelReply) -> bool:
    return any(isinstance(block, ToolUseBlock) for block in reply.blocks)
