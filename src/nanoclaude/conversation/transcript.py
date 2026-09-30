"""The conversation transcript and the rules that keep it valid.

A malformed transcript is rejected by every provider, but only after a round
trip and with an error that does not name the offending block. Validity is
therefore defined here, as a property of the data, and checked wherever a
transcript crosses a boundary.

Everything is immutable: the loop keeps earlier states and compares them.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

Role: TypeAlias = Literal["user", "assistant"]


class TranscriptError(ValueError):
    """A transcript violates one of the structural rules in :func:`validate`."""


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """Provider-side reasoning. Stored so it can be sent back verbatim."""

    text: str
    signature: str = ""


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    arguments: Mapping[str, Any]

    # Declared unhashable rather than left to the default: arguments is decoded
    # JSON and can hold lists or nested objects, so no version of this class
    # has reliably hashable arguments. Left to frozen=True's default eq=True,
    # dataclass would generate a real __hash__ -- isinstance(x, Hashable) would
    # say True -- that only raises TypeError when actually called, and names
    # "mappingproxy" rather than this class. Callers that need an identity for
    # a call should use .id, which is what identifies one anyway.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Deep-copied, not just wrapped: arguments is decoded JSON and can
        # hold lists or nested dicts (Edit's edit list, Grep's pattern
        # lists), and a shallow copy would leave those nested objects shared
        # with whatever the caller holds -- mutating them after construction
        # would still change what this block reports, which is exactly the
        # audit-row-disagrees-with-the-transcript failure this exists to
        # prevent.
        #
        # What this guarantees: nobody holding the mapping passed to the
        # constructor can change this block's record, and
        # ``block.arguments[k] = v`` fails.
        # What it does not: there is no recursive freeze, so reaching into
        # an already-nested value -- ``block.arguments["edits"]``, say --
        # and mutating that list or dict in place still mutates this
        # block's own copy. Nested dicts stay dicts and nested lists stay
        # lists on purpose: a recursive freeze would break
        # ``json.dumps(dict(block.arguments))`` on a nested mappingproxy,
        # and any ``isinstance(x, list)`` check a tool makes on its own
        # arguments.
        object.__setattr__(self, "arguments", MappingProxyType(deepcopy(dict(self.arguments))))


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


Block: TypeAlias = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    blocks: tuple[Block, ...]

    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return tuple(b for b in self.blocks if isinstance(b, ToolUseBlock))

    def tool_results(self) -> tuple[ToolResultBlock, ...]:
        return tuple(b for b in self.blocks if isinstance(b, ToolResultBlock))

    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, TextBlock))


@dataclass(frozen=True, slots=True)
class Transcript:
    messages: tuple[Message, ...] = ()

    def append(self, message: Message) -> Transcript:
        return Transcript((*self.messages, message))

    def last(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def pending_tool_uses(self) -> tuple[ToolUseBlock, ...]:
        last = self.last()
        if last is None or last.role != "assistant":
            return ()
        return last.tool_uses()

    def __len__(self) -> int:
        return len(self.messages)


def user_text(text: str) -> Message:
    return Message("user", (TextBlock(text),))


def assistant_text(text: str) -> Message:
    return Message("assistant", (TextBlock(text),))


def validate(transcript: Transcript) -> None:
    """Raise :class:`TranscriptError` unless the transcript is well formed.

    1. Starts with the user and alternates strictly.
    2. No message is empty.
    3. ``tool_use`` and ``thinking`` only in assistant messages, ``tool_result``
       only in user ones.
    4. Every ``tool_use`` is answered by exactly one ``tool_result`` in the very
       next message, in the same order -- unless it is in the last message, in
       which case it is pending and the loop still owes a result.
    5. Every ``tool_result`` answers a ``tool_use`` in the message before it.
    6. Tool-call ids are unique across the transcript.
    """
    messages = transcript.messages
    if not messages:
        return
    if messages[0].role != "user":
        raise TranscriptError("transcript must start with a user message")

    seen: set[str] = set()
    for index, message in enumerate(messages):
        if not message.blocks:
            raise TranscriptError(f"message {index} has no blocks")
        expected: Role = "user" if index % 2 == 0 else "assistant"
        if message.role != expected:
            raise TranscriptError(
                f"message {index} has role {message.role!r}, expected {expected!r}: "
                "roles must alternate starting with the user"
            )
        for block in message.blocks:
            if isinstance(block, ToolUseBlock):
                if message.role != "assistant":
                    raise TranscriptError(f"tool_use in {message.role} message {index}")
                if block.id in seen:
                    raise TranscriptError(f"duplicate tool_use id {block.id!r}")
                seen.add(block.id)
            elif isinstance(block, ToolResultBlock) and message.role != "user":
                raise TranscriptError(f"tool_result in {message.role} message {index}")
            elif isinstance(block, ThinkingBlock) and message.role != "assistant":
                raise TranscriptError(f"thinking in {message.role} message {index}")

    for index, message in enumerate(messages):
        uses = message.tool_uses()
        if uses and index != len(messages) - 1:
            answers = messages[index + 1].tool_results()
            if tuple(u.id for u in uses) != tuple(a.tool_use_id for a in answers):
                raise TranscriptError(
                    f"message {index} requested {[u.id for u in uses]} but message "
                    f"{index + 1} answered {[a.tool_use_id for a in answers]}: every "
                    "tool_use needs exactly one tool_result, in order"
                )
        results = message.tool_results()
        if results:
            if index == 0:
                raise TranscriptError("message 0 cannot contain tool_result blocks")
            offered = {u.id for u in messages[index - 1].tool_uses()}
            for result in results:
                if result.tool_use_id not in offered:
                    raise TranscriptError(
                        f"tool_result {result.tool_use_id!r} in message {index} answers "
                        "a tool_use that was never requested"
                    )
