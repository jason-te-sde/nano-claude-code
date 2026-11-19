"""Running a sub-agent: an ordinary session, nested, and held to its limits.

A ``Task`` call starts a :class:`~nanoclaude.agent.session.Session` of its own. It is the
same class the person talks to, with four things different, and the loop has no branch for
it:

* its model is the one the sub-agent's role names, so that the cost lands under that role;
* its tools are the ones its persona lists and nothing else: no ``Task``, no write tool, no
  tool a server brought;
* its sandbox is made from the parent's by :meth:`Sandbox.narrowed_to`, so it can be
  narrower and never wider, and its policy is the parent's, so the mode and every rule hold;
* nobody can be asked anything. Its front end declines every question and tells the parent's
  nothing but that a request is under way, so a call that would have needed an answer is
  refused to the sub-agent and the person is never interrupted by it.

Two things stay with the parent. The session grants the person gave are inherited as they
stand and the sub-agent cannot add one, since a grant is made by answering "always" to a
question and none is asked. And what comes back is one string: its final text, with a line
naming the limit when it was stopped, cut to ``output_cap_bytes``. Its own conversation is
not stored, and the calls it made are written to the audit under the turn of the ``Task``
call, marked with that call's id.

The limits are three: model calls, estimated input tokens, and wall-clock time. The first two
are checked by a gate the session calls before each request; the third by a timeout around
the whole run, so that a model that never answers is stopped too.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from nanoclaude.agent.loop import StopReason
from nanoclaude.agent.session import Session
from nanoclaude.agent.ui import UI, SilentUI
from nanoclaude.conversation.budget import ContextTooSmallError
from nanoclaude.conversation.transcript import TextBlock, Transcript
from nanoclaude.providers.base import ModelError
from nanoclaude.subagents import SUBAGENTS, SubAgent, SubAgentCall, SubAgentResult
from nanoclaude.tools.glob import GlobTool
from nanoclaude.tools.grep import GrepTool
from nanoclaude.tools.read import ReadTool
from nanoclaude.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from nanoclaude.tools.base import Tool


@dataclass(frozen=True, slots=True)
class SubAgentLimits:
    """How far one sub-agent may go. Hitting any of them stops it."""

    max_model_calls: int = 30
    #: Summed over the requests it makes, each counted as the budget counts a request.
    max_input_tokens: int = 200_000
    max_seconds: float = 600.0


#: Looked up when a sub-agent starts, not when this module is imported.
DEFAULT_LIMITS = SubAgentLimits()

#: The tools a persona can name, and how each is made. All three read; a persona that names
#: another is an error, and no built-in one does.
TOOLS: dict[str, Callable[[], Tool]] = {
    "Read": ReadTool,
    "Glob": GlobTool,
    "Grep": GrepTool,
}


class LimitReached(Exception):
    """A sub-agent was about to go past one of its limits. ``limit`` says which, for the note."""

    def __init__(self, limit: str) -> None:
        super().__init__(limit)
        self.limit = limit


class RunGate:
    """The model-call and input-token limits, as the gate a session calls before each request.

    It counts the requests it has let through and the tokens they were estimated to carry, and
    refuses the one that would go past a limit: that request is never sent.
    """

    def __init__(self, limits: SubAgentLimits) -> None:
        self._limits = limits
        self.calls = 0
        self.input_tokens = 0

    def __call__(self, estimated_tokens: int) -> None:
        limits = self._limits
        if self.calls >= limits.max_model_calls:
            raise LimitReached(f"{limits.max_model_calls} model calls")
        if self.input_tokens + estimated_tokens > limits.max_input_tokens:
            raise LimitReached(f"{limits.max_input_tokens:,} estimated input tokens")
        self.calls += 1
        self.input_tokens += estimated_tokens


class _DelegatedUI(SilentUI):
    """The front end of a sub-agent: it asks nobody, and says only that it is working.

    A request in progress and its retries are passed to the parent's front end, so that what
    shows the person that something is running keeps showing it for as long as the sub-agent
    takes. Nothing else is: not its text, not its calls and above all not a question.
    """

    def __init__(self, parent: UI) -> None:
        self._parent = parent

    def on_request_start(self, role: str) -> None:
        self._parent.on_request_start(role)

    def on_request_end(self) -> None:
        self._parent.on_request_end()

    def on_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        self._parent.on_retry(attempt, delay_s, reason)


def build_registry(agent: SubAgent) -> ToolRegistry:
    """The tools ``agent`` is given: those its persona names, made fresh for this run."""
    unknown = [name for name in agent.tools if name not in TOOLS]
    if unknown:
        raise ValueError(
            f"sub-agent {agent.name!r} names {', '.join(unknown)}, "
            f"which a sub-agent cannot be given; the tools are {', '.join(sorted(TOOLS))}"
        )
    return ToolRegistry([TOOLS[name]() for name in agent.tools])


def cap_bytes(text: str, cap: int) -> str:
    """``text`` cut to at most ``cap`` bytes of UTF-8, with a note of how many were cut.

    The cut falls between characters, and what is dropped is counted in bytes, a character
    that did not fit included.
    """
    data = text.encode("utf-8")
    if len(data) <= cap:
        return text
    kept = data[:cap].decode("utf-8", errors="ignore")
    cut = len(data) - len(kept.encode("utf-8"))
    return f"{kept}\n[output cut: {cut} bytes beyond the {cap}-byte limit]"


def last_text(transcript: Transcript) -> str:
    """What the model last said in words, whether or not a tool call came with it; or ''."""
    for message in reversed(transcript.messages):
        if message.role != "assistant":
            continue
        said = "\n".join(b.text for b in message.blocks if isinstance(b, TextBlock)).strip()
        if said:
            return said
    return ""


def _duration(seconds: float) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} of wall-clock time"
    return f"{seconds:g} seconds of wall-clock time"


def _stopped(agent: SubAgent, limit: str) -> str:
    return f"[{agent.name} stopped: it reached its limit of {limit}]"


#: What a reply that ended for another reason than finishing is told to the model as.
_ENDED = {
    StopReason.MAX_TOKENS: "its last reply was cut off at the model's output limit",
    StopReason.REFUSAL: "the model declined to continue",
    StopReason.TURN_LIMIT: "it ran out of turns",
}


async def run_subagent(
    parent: Session,
    call: SubAgentCall,
    *,
    audit_turn: int,
    agent: SubAgent | None = None,
    root: str | None = None,
    limits: SubAgentLimits | None = None,
) -> SubAgentResult:
    """Run the sub-agent ``call`` names in a session nested in ``parent``; what it answered.

    ``audit_turn`` is the turn of the ``Task`` call, which the sub-agent's own calls are
    recorded under. ``root`` makes its sandbox narrower than the parent's, and is refused when
    it is not inside it (:class:`~nanoclaude.permissions.sandbox.WiderSandboxError`); the
    ``Task`` tool never gives one in this version. ``agent`` and ``limits`` are for a
    persona that is not built in and for a limit other than the shipped one.

    A sub-agent that cannot finish, because its model failed, comes back as an error result
    that says so and what it had written; it is not an exception, since it is the tool's
    answer. A stop by a limit is not an error: the result is the last text it wrote, with a
    line that names the limit.
    """
    agent = SUBAGENTS[call.agent] if agent is None else agent
    limits = DEFAULT_LIMITS if limits is None else limits
    policy = parent.executor.policy
    sandbox = policy.sandbox if root is None else policy.sandbox.narrowed_to(root)
    limit = limits.max_model_calls
    child = Session(
        root=sandbox.roots[0],
        # One more turn than calls, so that the gate is what stops it and the loop's own
        # turn limit, which is the person's and is for their conversation, never is.
        config=replace(parent.config, limits=replace(parent.config.limits, max_turns=limit + 1)),
        router=parent.router,
        registry=build_registry(agent),
        policy=replace(policy, sandbox=sandbox),
        ui=_DelegatedUI(parent.ui),
        store=None,
        audit=parent.audit,
        session_id=parent.session_id,
        retry=parent.retry,
        redactor=parent.redactor,
        home=parent.home,
        role=agent.role,
        system_prompt=agent.system_prompt,
        mentions=False,
        gate=RunGate(limits),
        parent_call_id=call.parent_call_id,
        audit_turn=audit_turn,
    )
    # What the person has already granted holds here as it does there. Nothing here can add to
    # it: a grant is the answer "always" to a question, and this front end answers no.
    child.executor.grants = parent.executor.grants

    stopped: str | None = None
    ended: str | None = None
    answer = ""
    timeout = asyncio.timeout(limits.max_seconds)
    try:
        async with timeout:
            done = await child.run(call.prompt)
    except LimitReached as reached:
        stopped = reached.limit
    except TimeoutError:
        if not timeout.expired():
            raise
        stopped = _duration(limits.max_seconds)
    except (ModelError, ContextTooSmallError) as error:
        return SubAgentResult(_failure(agent, str(error), last_text(child.state.transcript)), True)
    else:
        if done.reason is StopReason.MODEL_UNSUITABLE:
            return SubAgentResult(
                f"The {agent.name} sub-agent could not finish: its model did not write a tool "
                f"call that could be run. Do the work yourself, or tell the user to route the "
                f"{agent.role} role to a model with native tool calling "
                f"(--role {agent.role}=<model>).",
                True,
            )
        answer = done.text
        if done.reason is not StopReason.COMPLETED:
            ended = _ENDED[done.reason]
    if stopped is not None:
        answer = last_text(child.state.transcript)

    cap = parent.config.limits.output_cap_bytes
    notes = []
    if ended is not None:
        notes.append(f"[{agent.name} ended early: {ended}]")
    if stopped is not None:
        notes.append(_stopped(agent, stopped))
    body = cap_bytes(answer, cap) if answer else f"({agent.name} returned no text)"
    return SubAgentResult("\n\n".join([body, *notes]))


def _failure(agent: SubAgent, reason: str, written: str) -> str:
    """The error result for a sub-agent whose model failed: what failed, then what to do."""
    text = (
        f"The {agent.name} sub-agent could not finish: {reason}. "
        "Do the work yourself with your own tools, or tell the user what failed."
    )
    if written:
        text += f"\n\nWhat it had written before it stopped:\n{written}"
    return text
