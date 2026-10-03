"""The agent loop, with the awaiting taken out.

One turn is: ask the model, take what it said, run the tools it asked for, hand
back the results. None of that happens here. This module computes *what should
happen next* given the state and a reply; :mod:`nanoclaude.agent.session` does
the awaiting. The split is what makes a whole session testable from a scripted
model with no network, no disk and no clock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    assistant_text,
    user_text,
    validate,
)
from nanoclaude.providers.base import ModelReply, StopKind, Usage

DEFAULT_MAX_TURNS = 40


class LoopError(RuntimeError):
    """The loop was driven into a state it does not allow."""


class StopReason(StrEnum):
    COMPLETED = "completed"
    TURN_LIMIT = "turn_limit"
    REFUSAL = "refusal"
    MAX_TOKENS = "max_tokens"
    #: A model without native tool calling was asked again as often as the text
    #: protocol allows and still wrote no call that could be run. The session sets
    #: it, since the session owns the retries; the loop never produces it.
    MODEL_UNSUITABLE = "model_unsuitable"


@runtime_checkable
class Observation(Protocol):
    """What :func:`observe` needs from a tool result. ``ToolOutcome`` satisfies it.

    Read-only, because observe() only reads them. Plain attributes would declare
    them settable, which a frozen dataclass such as ``ToolOutcome`` is not.
    """

    @property
    def tool_use_id(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def is_error(self) -> bool: ...

    @property
    def observed(self) -> tuple[tuple[str, Any], ...]: ...


@dataclass(frozen=True, slots=True)
class LoopState:
    transcript: Transcript
    turn: int
    max_turns: int
    read_state: Mapping[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    # Declared unhashable rather than left to the default: read_state is always
    # a MappingProxyType (never hashable, see __post_init__ below) and
    # transcript can hold a ToolUseBlock, which is unhashable for the same
    # reason (see transcript.py). Left to frozen=True's default eq=True,
    # dataclass would generate a real __hash__ -- isinstance(x, Hashable) would
    # say True -- that only raises TypeError when actually called, and names
    # "mappingproxy" or "ToolUseBlock" rather than this class. Nothing needs to
    # put a LoopState in a set or use one as a dict key; compare by value with
    # == instead.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_state", MappingProxyType(dict(self.read_state)))

    @property
    def turns_left(self) -> int:
        return max(0, self.max_turns - self.turn)

    def stamp_for(self, resolved_path: str) -> Any | None:
        """What this session last saw at ``resolved_path``, if anything."""
        return self.read_state.get(resolved_path)


@dataclass(frozen=True, slots=True)
class Done:
    state: LoopState
    text: str
    reason: StopReason

    # Embeds a LoopState, which is itself declared unhashable above -- the
    # same defect would otherwise resurface one level up.
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class RunTools:
    state: LoopState
    calls: tuple[ToolUseBlock, ...]

    # Embeds a LoopState (unhashable) and a tuple of ToolUseBlock, each of
    # which is unhashable too -- see LoopState above and transcript.py.
    __hash__ = None  # type: ignore[assignment]


StepOutcome: TypeAlias = Done | RunTools


def start(prompt: str, *, max_turns: int = DEFAULT_MAX_TURNS) -> LoopState:
    if max_turns < 1:
        raise LoopError(f"max_turns must be at least 1, got {max_turns}")
    return LoopState(Transcript((user_text(prompt),)), turn=0, max_turns=max_turns)


def resume(state: LoopState, prompt: str) -> LoopState:
    """Continue the conversation with a new prompt.

    The prompt starts the turn count again: ``max_turns`` bounds what one prompt may
    take, not what a whole conversation does, so a long session is not stopped by
    the sum of everything it has done. What the conversation has read and used
    carries over.
    """
    if state.transcript.pending_tool_uses():
        raise LoopError("cannot resume while tool calls are unanswered")
    return replace(state, transcript=state.transcript.append(user_text(prompt)), turn=0)


def step(state: LoopState, reply: ModelReply) -> StepOutcome:
    if not reply.blocks:
        raise LoopError("model returned an empty reply")

    transcript = state.transcript.append(Message("assistant", tuple(reply.blocks)))
    advanced = replace(state, transcript=transcript, usage=state.usage + reply.usage)
    validate(transcript)

    calls = transcript.pending_tool_uses()
    if not calls:
        reason = {
            StopKind.REFUSAL: StopReason.REFUSAL,
            StopKind.MAX_TOKENS: StopReason.MAX_TOKENS,
        }.get(reply.stop, StopReason.COMPLETED)
        return Done(advanced, _text_of(reply.blocks), reason)

    if advanced.turn + 1 >= advanced.max_turns:
        # Refusing to run them is fine; leaving them unanswered is not, because
        # the next request would carry an invalid transcript. And the transcript
        # ends with text (spec 5.4), not with those refusals: a conversation that
        # ended on a user message looks like a turn nobody answered, so the next
        # prompt would be told this one was interrupted, and could not be added
        # to the transcript at all, two user messages in a row.
        refusals = tuple(
            ToolResultBlock(call.id, f"not executed: turn limit of {state.max_turns} reached", True)
            for call in calls
        )
        said = f"Stopped after {advanced.turn + 1} turns without finishing the task."
        closed = transcript.append(Message("user", refusals)).append(assistant_text(said))
        validate(closed)
        final = replace(advanced, transcript=closed, turn=advanced.turn + 1)
        return Done(final, said, StopReason.TURN_LIMIT)

    return RunTools(advanced, calls)


def observe(state: LoopState, outcomes: Sequence[Observation]) -> LoopState:
    """Record the results of the outstanding calls.

    Every pending call must be answered exactly once, in the order requested. A
    driver that drops or reorders results is caught here rather than three turns
    later when a provider rejects the transcript.
    """
    pending = state.transcript.pending_tool_uses()
    if not pending:
        raise LoopError("observe() called with no tool calls outstanding")

    requested = tuple(call.id for call in pending)
    answered = tuple(outcome.tool_use_id for outcome in outcomes)
    if requested != answered:
        raise LoopError(f"expected results for {list(requested)}, got {list(answered)}")

    transcript = state.transcript.append(
        Message(
            "user",
            tuple(ToolResultBlock(o.tool_use_id, o.content, o.is_error) for o in outcomes),
        )
    )
    validate(transcript)

    read_state = dict(state.read_state)
    for outcome in outcomes:
        for path, stamp in outcome.observed:
            read_state[path] = stamp

    return replace(state, transcript=transcript, turn=state.turn + 1, read_state=read_state)


def _text_of(blocks: Sequence[object]) -> str:
    return "\n".join(b.text for b in blocks if isinstance(b, TextBlock)).strip()
