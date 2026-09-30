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


@dataclass(frozen=True, slots=True)
class ModelRequest:
    system: str
    transcript: Transcript
    tools: Sequence[ToolSpec]
    max_output_tokens: int
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelReply:
    blocks: tuple[Block, ...]
    stop: StopKind
    usage: Usage
    model: str


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
