"""The provider boundary: one request type in, one reply type out.

Adapters translate between this and their vendor's wire format. Nothing above
this layer may branch on a provider name -- behaviour differences are expressed
as :class:`~nanoclaude.providers.capabilities.Capabilities` and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from nanoclaude.conversation.transcript import Block, Transcript


class StopKind(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"


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
    temperature: float = 0.0

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


class ModelError(RuntimeError):
    """The provider failed in a way the loop cannot paper over.

    ``retryable`` drives the policy in spec section 7.4: a 429 is retryable, a
    401 is not, and the difference must survive the trip up to the session.
    """

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@runtime_checkable
class ModelClient(Protocol):
    @property
    def model_id(self) -> str: ...

    async def complete(self, request: ModelRequest) -> ModelReply: ...
