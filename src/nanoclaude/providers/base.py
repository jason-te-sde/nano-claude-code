"""The provider boundary: one request type in, one reply type out.

Adapters translate between this and their vendor's wire format. Nothing above
this layer may branch on a provider name -- behaviour differences are expressed
as :class:`~nanoclaude.providers.capabilities.Capabilities` and nothing else.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from nanoclaude.conversation.transcript import Block, TextBlock, Transcript


def new_call_id(prefix: str) -> str:
    """An id for a tool call that arrived without one: ``<prefix>_`` and twelve hex digits.

    Random, because a transcript refuses an id it already holds, and a counter
    repeats itself: in the next reply (every reply would start again at the same
    number) and in the next process (a resumed conversation already holds the ids
    an earlier one made).
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class StopKind(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    #: Not a reason the model gave: the stream broke before the model had finished. Only
    #: the ``partial`` reply of a :class:`ModelError` carries it, and such a reply is
    #: never a whole one, so it is never handed to the loop as the model's answer.
    CUT_OFF = "cut_off"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    schema: Mapping[str, Any]

    # Declared unhashable rather than left to the default: schema is a JSON
    # Schema object decoded the same way ToolUseBlock.arguments is -- see
    # transcript.py -- so no version of this class has reliably hashable
    # schemas either (a MappingProxyType would not help: it is not hashable
    # itself). Left to frozen=True's default eq=True, dataclass would generate
    # a real __hash__ -- isinstance(x, Hashable) would say True -- that only
    # raises TypeError when actually called, and names "dict" rather than
    # this class.
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    system: str
    transcript: Transcript
    tools: Sequence[ToolSpec]
    max_output_tokens: int
    #: None sends no temperature at all, so the provider's default applies. Current
    #: Claude models and OpenAI's reasoning models refuse any other value.
    temperature: float | None = None

    # Embeds a Transcript and a sequence of ToolSpec, both declared unhashable
    # (see transcript.py and above) -- the same defect resurfacing one level
    # up, the way Done and RunTools embed LoopState in loop.py. Every
    # ModelRequest carries a transcript, so this is unconditional regardless
    # of what either field holds.
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ModelReply:
    blocks: tuple[Block, ...]
    stop: StopKind
    usage: Usage
    model: str

    # Declared unhashable rather than left to the default: blocks can hold a
    # ToolUseBlock, declared unhashable in transcript.py for holding
    # decoded-JSON arguments. Left to frozen=True's default eq=True, dataclass
    # would generate a real __hash__ -- isinstance(x, Hashable) would say True
    # -- that hashes a text-only reply just fine and only raises TypeError,
    # naming "ToolUseBlock" rather than this class, once a tool-call reply
    # reaches it. That split is exactly the trap: a test written with simple
    # values would pass and a real tool-call turn would fail.
    __hash__ = None  # type: ignore[assignment]


def partial_reply(texts: Iterable[str], usage: Usage, model: str) -> ModelReply:
    """What had arrived of a reply whose stream broke: its text, and the usage so far.

    Text only, one block for each non-empty piece of ``texts``. Thinking and tool calls are
    left out, whole or not: nothing may be run or sent back to the model from a reply that
    did not finish. A reply with no text is still a reply, so that what the provider had
    reported by then (the input it billed) is not lost with it.
    """
    return ModelReply(tuple(TextBlock(t) for t in texts if t), StopKind.CUT_OFF, usage, model)


class ModelError(RuntimeError):
    """The provider failed in a way the loop cannot paper over.

    ``retryable`` drives the policy in spec section 7.4: a 429 is retryable, a
    401 is not, and the difference must survive the trip up to the session.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        context_overflow: bool = False,
        partial: ModelReply | None = None,
    ) -> None:
        if partial is not None and retryable:
            # Asking again would show the same words twice and run again a call that
            # the first attempt may have made. Refused here, where the mistake is made,
            # rather than left to be found by whoever retries.
            raise ValueError(
                "a reply that was partly received is never retried (spec 7.4): "
                "an error that holds one cannot be retryable"
            )
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        #: True when the provider said the request does not fit the model's context
        #: window. Spec 7.4: the session answers it with one compaction and one retry.
        self.context_overflow = context_overflow
        #: What had arrived when a stream broke midway (spec 7.4): the text received, as
        #: :func:`partial_reply` builds it, and the usage the provider had reported by
        #: then. None when nothing a person could have seen or a tool could have run on
        #: had arrived, so that the request may be asked again. Never set on an error
        #: that is retryable.
        self.partial = partial


@runtime_checkable
class ModelClient(Protocol):
    """Something that answers a :class:`ModelRequest`.

    ``on_text``, when given, is called with every piece of visible reply text as it
    arrives, in order, so that a person can read a reply that takes tens of seconds
    while it is written. It is never given thinking text or the arguments of a tool
    call, and the pieces joined are the reply's text blocks joined. A stream that
    breaks midway raises a :class:`ModelError` whose ``partial`` holds what had arrived.
    """

    @property
    def model_id(self) -> str: ...

    async def complete(
        self, request: ModelRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelReply: ...
