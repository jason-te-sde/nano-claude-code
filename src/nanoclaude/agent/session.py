"""One conversation, wired to everything it needs.

This is the only module that awaits a provider, and therefore the only one that
implements the retry policy from spec 7.4. It is also where compaction is
decided, because the budget verdict depends on the assembled context, which
depends on the model's window, which depends on the role -- all of which meet
here and nowhere else.

What it still does not know about is the CLI. Confirmation, progress, retry
notices and the text of a reply as it arrives go out through the UI protocol, so
the same class backs the REPL, headless mode and every test.

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
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path

from nanoclaude.agent import loop
from nanoclaude.agent.executor import Executor
from nanoclaude.agent.loop import Done, LoopState, StopReason
from nanoclaude.agent.router import Router
from nanoclaude.agent.ui import UI, AutoDecline
from nanoclaude.config.schema import ROLES, Config
from nanoclaude.context.assemble import assemble
from nanoclaude.context.mentions import expand_mentions
from nanoclaude.conversation.budget import Budget, HeuristicCounter
from nanoclaude.conversation.compaction import SUMMARY_TEMPLATE, full_compact, micro_compact
from nanoclaude.conversation.store import MessageConflictError, Store
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
from nanoclaude.providers.base import (
    ModelError,
    ModelReply,
    ModelRequest,
    ToolSpec,
    Usage,
    new_call_id,
    partial_reply,
)
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

#: Ends the text of a reply whose stream broke midway, as the model sees it when the person
#: asks it to go on: it knows where it stopped, and that nobody saw the rest.
_CUT_OFF_REPLY = "[this reply was cut off before it finished]"

#: A main model whose window is smaller than this many tokens keeps at most
#: ``SMALL_WINDOW_KEEP_RECENT`` recent turns whatever ``keep_recent_turns`` says
#: (spec 7.3): three verbatim turns would leave it little room for a summary and
#: its own reply.
SMALL_WINDOW_TOKENS = 32_000
SMALL_WINDOW_KEEP_RECENT = 2


def new_session_id() -> str:
    """A short random identifier for a session: twelve hexadecimal characters."""
    return uuid.uuid4().hex[:12]


class UnknownSessionError(RuntimeError):
    """A session to resume is not in the store. The message is for the person, as it stands."""


class SessionChangedError(RuntimeError):
    """Another process wrote to the stored session this one is writing to.

    The message is for the person, as it stands. Nothing of the other process's was
    overwritten, and nothing of this one's was stored.
    """


def _correction(complaints: Sequence[str]) -> str:
    """What a model that wrote no usable tool call is told: what failed, then what to do."""
    listed = "\n".join(f"- {complaint}" for complaint in complaints)
    return (
        f"Your last reply contained no tool call that could be run:\n{listed}\n"
        'Write each call again as <tool name="Name"> followed by one JSON object and '
        "</tool>, or answer without a tool call if you do not need one."
    )


def _unsuitable(model: str, attempts: int) -> str:
    """What a person is told when a model has used every retry without writing a call.

    Spec 4.4 asks for exactly this, and spec 17.9 gives the form.
    """
    return (
        f"error: {model} did not produce a valid tool call in {attempts} attempts "
        "\u2014 choose a model with native tool calling (--model)"
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

    ``resume`` takes up a stored session instead of starting one: ``session_id`` is
    the one to continue, its conversation is loaded, and what the session stores and
    spends from then on is added to its rows. Continue it with :meth:`follow_up`;
    :meth:`run` starts a new conversation, in the same row.
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
    resume: bool = False
    state: LoopState = field(init=False)
    executor: Executor = field(init=False)
    # The messages the store holds for this session, as of the last _persist().
    _stored: tuple[Message, ...] = field(default=(), init=False, repr=False)
    # What a resumed session had used before this run, from its stored row. Nothing
    # for a new one, which has used nothing: a cost of zero, a known one.
    _earlier_usage: Usage = field(default_factory=Usage, init=False, repr=False)
    _earlier_cost: float | None = field(default=0.0, init=False, repr=False)
    # The audit numbers turns across the whole session, and the loop's own count starts
    # again with every prompt. A prompt's turn n is audited as _audit_turn_base + n, and
    # _next_audit_turn is the first number no batch has used: the base of the next prompt.
    _audit_turn_base: int = field(default=0, init=False, repr=False)
    _next_audit_turn: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Empty rather than a placeholder prompt: follow_up() is a valid first
        # call (the REPL makes it after /clear), and it must find the configured
        # turn limit and a transcript it can start.
        self.state = LoopState(Transcript(), turn=0, max_turns=self.config.limits.max_turns)
        if self.resume:
            self._take_up_stored_session()
        elif self.store is not None:
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

    def _take_up_stored_session(self) -> None:
        """Load the session ``session_id`` names, or say that the store does not hold it.

        Nothing is written: the row exists, and the first save of this session adds
        to what is stored. A tail the earlier process left open (a call nobody
        answered, a request nobody replied to) is closed when the next prompt
        arrives, as it is after an interruption within one process. What the
        conversation had read is not stored, so the model reads a file again before
        it edits it.
        """
        if self.store is None:
            raise ValueError("resume needs a store to read the session from")
        row = self.store.session_row(self.session_id)
        if row is None:
            raise UnknownSessionError(
                f'no stored session "{self.session_id}" \u2014 check the id, or leave out '
                "--resume to start a new session"
            )
        transcript = self.store.load_transcript(self.session_id)
        self.state = replace(self.state, transcript=transcript)
        self._stored = transcript.messages
        self._earlier_usage = Usage(row.total_input_tokens, row.total_output_tokens)
        self._earlier_cost = row.total_cost_usd
        self._next_audit_turn = self.store.next_tool_turn(self.session_id)

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
        self._audit_turn_base = self._next_audit_turn
        return await self._drive()

    async def follow_up(self, prompt: str) -> Done:
        """Continue the conversation with ``prompt``."""
        self._mend()
        # The limit in force is the config's now, as it is for run(): a front end
        # may have replaced the config since the session was built.
        state = replace(self.state, max_turns=self.config.limits.max_turns)
        self.state = loop.resume(state, self._expand(prompt))
        self._audit_turn_base = self._next_audit_turn
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

            try:
                raw = await self._ask_main(capabilities, system, specs)
            except ModelError as error:
                if error.partial is None:
                    raise
                raise self._keep_cut_off(error.partial, error) from error
            self.router.record("main", raw.usage, *self._adapter_and_model("main"))
            reply: ModelReply = raw
            complaints: tuple[str, ...] = ()
            if not capabilities.native_tools:
                reply, complaints = parse_reply(raw, next_id=self._next_call_id)
            self.ui.on_reply(reply)

            spent = 0  # the attempts made, once the model has used every retry for nothing
            if complaints and not _has_calls(reply):
                if failures < MAX_PARSE_RETRIES:
                    # Ask the same model again. What it said goes into the transcript
                    # as it said it, and the complaints come back as the user's next
                    # message: a tool_result cannot carry them, because there is no
                    # call to answer. Not loop.resume(): that starts a new prompt and
                    # its turn count, and this is the same prompt still running.
                    failures += 1
                    said = loop.step(self.state, raw).state
                    self.state = replace(
                        said, transcript=said.transcript.append(user_text(_correction(complaints)))
                    )
                    continue
                spent = failures + 1
            failures = 0

            outcome = loop.step(self.state, reply)
            self.state = outcome.state
            self._persist()
            if isinstance(outcome, Done):
                if spent and outcome.reason is StopReason.COMPLETED:
                    # A reply cut off by the output limit keeps its own reason: that
                    # is not a verdict on the model.
                    model = self._adapter_and_model("main")[1]
                    outcome = replace(
                        outcome,
                        text=_unsuitable(model, spent),
                        reason=StopReason.MODEL_UNSUITABLE,
                    )
                self._finish()
                return outcome

            audit_turn = self._audit_turn_base + self.state.turn
            # Claimed as the batch starts, not when it ends: one that is interrupted has
            # already audited its calls under this number.
            self._next_audit_turn = audit_turn + 1
            results = await self.executor.run_batch(outcome.calls, self.state, turn=audit_turn)
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

    def _request(
        self, capabilities: Capabilities, system: str, specs: tuple[ToolSpec, ...]
    ) -> ModelRequest:
        """The next request for the main model, from the conversation as it is now."""
        return ModelRequest(
            system=system,
            transcript=self.state.transcript,
            tools=specs,
            max_output_tokens=capabilities.max_output,
        )

    async def _ask_main(
        self, capabilities: Capabilities, system: str, specs: tuple[ToolSpec, ...]
    ) -> ModelReply:
        """Ask the main model, answering a context overflow the way spec 7.4 says to.

        When the provider reports that the conversation does not fit its window the
        session compacts it once, fully, and asks again once. A second overflow, or a
        conversation with nothing older than its recent turns to compact, ends the
        attempt: asking again could only fail the same way. This is the main request
        only. The summary is a request to a model with a window of its own, and an
        overflow there is that model's error and not something to retry.
        """
        # A native model's text is shown as it arrives. A text-protocol reply is not: it
        # carries tool markup, and is parsed once the whole of it is here.
        stream = capabilities.native_tools
        try:
            return await self._complete(
                "main", self._request(capabilities, system, specs), stream=stream
            )
        except ModelError as error:
            if not error.context_overflow:
                raise
            overflow = error
        if not await self._compact_fully(capabilities):
            raise self._does_not_fit(overflow, self._nothing_to_compact(capabilities)) from overflow
        try:
            return await self._complete(
                "main", self._request(capabilities, system, specs), stream=stream
            )
        except ModelError as error:
            if error.context_overflow:
                raise self._does_not_fit(error, "even after compacting it") from error
            raise

    def _nothing_to_compact(self, capabilities: Capabilities) -> str:
        """What a conversation that compaction could not shorten is said to lack."""
        kept = self._keep_recent(capabilities)
        last = "last turn" if kept == 1 else f"last {kept} turns"
        return f"and has nothing older than its {last} to compact"

    def _does_not_fit(self, cause: ModelError, why_not: str) -> ModelError:
        """The error for a conversation the main model's window cannot hold.

        ``why_not`` finishes the first half: compacting did not help, or there was
        nothing to compact. The second half says what to do, including the cause that
        is not the conversation's size at all: a ``context_window`` set above what the
        server really has, which compaction never catches because the budget believes it.
        """
        model = self._adapter_and_model("main")[1]
        return ModelError(
            f"the conversation does not fit the context window of {model} {why_not} "
            "\u2014 use a model with a larger window (--model), start over with /clear, or, "
            f"if context_window for {model} is set above the window the server really has "
            "(common with llama.cpp and vLLM), lower it",
            retryable=False,
            status=cause.status,
        )

    async def _complete(
        self, role: str, request: ModelRequest, *, stream: bool = False
    ) -> ModelReply:
        """One provider call under the retry policy; each retry is reported to the UI.

        The UI is told when the request starts and when it ends, however it ends: answered,
        failed or cancelled. The retries in between are part of the one request, so that
        whatever shows that something is running stays up through them. With ``stream`` the
        text of the reply goes to the UI as it arrives.
        """
        client = await self.router.client_for(role)
        streamed: list[str] = []

        def hand_on(piece: str) -> None:
            streamed.append(piece)
            self.ui.on_text(piece)

        async def once() -> ModelReply:
            if not stream:
                # Not handed to a client that was not asked to stream: one written to the
                # signature that took only the request would fail on every request.
                return await client.complete(request)
            try:
                return await client.complete(request, on_text=hand_on)
            except ModelError as error:
                if not (error.retryable and streamed):
                    raise
                # Whatever the client thinks of it, the person has read these words: asked
                # again, the reply would be written from its first word, below them (spec
                # 7.4). The adapters make this error themselves, and this is for the ones
                # that let a retryable one through after text.
                raise ModelError(
                    str(error),
                    retryable=False,
                    status=error.status,
                    partial=partial_reply(["".join(streamed)], Usage(), client.model_id),
                ) from error

        self.ui.on_request_start(role)
        try:
            return await with_retry(once, policy=self.retry, on_retry=self.ui.on_retry)
        finally:
            self.ui.on_request_end()

    def _keep_cut_off(self, partial: ModelReply, cause: ModelError) -> ModelError:
        """Keep what arrived of a reply whose stream broke midway, and say so (spec 7.4).

        The text is kept as an assistant message, marked as cut off, and stored with the rest
        of the conversation: the person can ask the model to continue, and it knows where it
        stopped. A call that had been half received is dropped, since it cannot be run, and
        so is thinking. What the provider had reported is recorded. Nothing is asked again,
        because the request may already have had effects: the turn stops, with the error that
        says so, and what to do next is the person's to decide.

        Returns the error to raise. The UI is given what is kept, as for any reply, so that
        one that did not show it as it arrived (a text-protocol reply is not streamed) can.
        """
        self.router.record("main", partial.usage, *self._adapter_and_model("main"))
        self.state = replace(self.state, usage=self.state.usage + partial.usage)
        text = "".join(b.text for b in partial.blocks if isinstance(b, TextBlock)).strip()
        what = f"the reply was cut off ({cause})"
        if not text:
            return ModelError(f"{what} \u2014 ask the model to continue", retryable=False)
        marked = assistant_text(f"{text}\n\n{_CUT_OFF_REPLY}")
        self.state = replace(self.state, transcript=self.state.transcript.append(marked))
        self._persist()
        self.ui.on_reply(partial)
        return ModelError(
            f"{what} \u2014 what arrived is kept; ask the model to continue", retryable=False
        )

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
        if verdict == "micro":
            compacted = micro_compact(
                transcript,
                keep_recent=self._keep_recent(capabilities),
                counter=HeuristicCounter(),
            )
            self.state = replace(self.state, transcript=compacted)
            self._persist()
        elif verdict == "full":
            await self._compact_fully(capabilities)

    def _keep_recent(self, capabilities: Capabilities) -> int:
        """How many recent turns a compaction leaves as they were.

        The configured number, except that a main model with a small window keeps at
        most two (spec 7.3). ``capabilities`` are the main model's: it is the one
        that has to fit the result.
        """
        configured = self.config.limits.keep_recent_turns
        if capabilities.context_window < SMALL_WINDOW_TOKENS:
            return min(configured, SMALL_WINDOW_KEEP_RECENT)
        return configured

    async def compact(self, instructions: str | None = None) -> bool:
        """Summarise the older part of the conversation now: the ``/compact`` command.

        True when a summary replaced older messages, False when there was nothing older than
        the recent turns to summarise. Not "whether the conversation changed": closing what an
        interrupted turn left open comes first and adds a note, so a conversation can change
        and have nothing compacted.
        """
        self._mend()
        # The summary may never arrive, and the store holds what the model is shown.
        self._persist()
        capabilities = await self.router.capabilities_for("main")
        return await self._compact_fully(capabilities, instructions)

    async def _compact_fully(
        self, capabilities: Capabilities, instructions: str | None = None
    ) -> bool:
        """Replace the older part of the conversation with a summary, and store it.

        The one place a full compaction happens: the budget calls it at the hard
        threshold, ``/compact`` calls it, and so does the answer to a provider that
        says the conversation does not fit. They differ in why, not in how.
        ``capabilities`` are the main model's. Returns whether anything was replaced.
        """
        extra = f"\n\nPay particular attention to: {instructions}" if instructions else ""

        async def summarise(text: str) -> str:
            return await self._summarise(text + extra)

        before = self.state.transcript
        compacted = await full_compact(
            before,
            keep_recent=self._keep_recent(capabilities),
            summarise=summarise,
        )
        self.state = replace(self.state, transcript=compacted)
        self._persist()
        return compacted != before

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
            if error.context_overflow:
                # The older history is what is being summarised, all of it in one
                # request, and it does not fit that model's window either. With the
                # compact role left at the main alias it is the same model that has
                # just refused it.
                model = self._adapter_and_model("compact")[1]
                raise ModelError(
                    f"the older conversation is too long for the context window of {model}, "
                    "the model that writes the summary \u2014 route the compact role to a model "
                    "with a larger window (--role compact=<model>), or, if context_window for "
                    f"{model} is set above the window the server really has, lower it so "
                    "conversations are compacted sooner",
                    retryable=False,
                    status=error.status,
                ) from error
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
        return new_call_id("tt")

    def _persist(self) -> None:
        if self.store is None:
            return
        messages = self.state.transcript.messages
        stored = self._stored
        if messages[: len(stored)] == stored:
            for seq in range(len(stored), len(messages)):
                try:
                    self.store.append_message(self.session_id, seq, messages[seq])
                except MessageConflictError as conflict:
                    # The store holds another message where this one was to go, so
                    # something else has written to the session since it was loaded.
                    raise SessionChangedError(
                        f"session {self.session_id} was changed by another process \u2014 "
                        "start a new session, or resume it again"
                    ) from conflict
        else:
            # Not an extension of what was stored: compaction replaced the head, or
            # the transcript was swapped for another. Appending would leave the old
            # rows beside the new ones.
            self.store.replace_transcript(self.session_id, self.state.transcript)
        self._stored = messages

    def _finish(self) -> None:
        if self.store is None:
            return
        # What the session had used before it was resumed, and this run. A cost that
        # is unknown on either side makes the total unknown, which is None here and
        # NULL in the store: it is not zero, and must not read as free.
        usage = self._earlier_usage + self.usage
        cost = self.cost
        if cost is not None and self._earlier_cost is not None:
            cost += self._earlier_cost
        else:
            cost = None
        self.store.finish_session(self.session_id, usage, cost)

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

    def clear(self) -> None:
        """Forget the conversation, as ``/clear`` does, and store that at once.

        The next prompt starts from nothing: no messages, no record of what was read
        (the model must read a file again before it edits it), no usage, and the turn
        limit the config has now. The store follows now, not at the next prompt, so a
        session that is closed or resumed next does not bring the conversation back.
        What was there stays in the archive.
        """
        self.state = LoopState(Transcript(), turn=0, max_turns=self.config.limits.max_turns)
        self._persist()

    async def aclose(self) -> None:
        """Store the conversation, record what the session spent, then close everything.

        The conversation is stored as it is, because a front end can change it between
        prompts (``/clear``) and the store otherwise learns of that only at the next
        one. A prompt that stops normally records what was spent as it goes; one that
        failed or was cancelled never reached that, so closing does it: the money for
        the rounds that were paid for is spent either way. Then every client is closed,
        then the store, and a step that fails does not stop the ones after it. Closing
        twice is harmless: a front end may close from its normal exit and from its
        cleanup.
        """
        if self._closed:
            return
        self._closed = True
        async with AsyncExitStack() as steps:
            # Registered in reverse, because they run last to first: the conversation is
            # stored, the spend recorded, the clients closed, and the store closed.
            steps.callback(self._close_store)
            steps.push_async_callback(self.router.aclose)
            steps.callback(self._finish)
            steps.callback(self._persist)

    def _close_store(self) -> None:
        if self.store is not None:
            self.store.close()


def _has_calls(reply: ModelReply) -> bool:
    return any(isinstance(block, ToolUseBlock) for block in reply.blocks)
