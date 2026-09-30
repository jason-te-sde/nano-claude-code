"""A model that says exactly what it was told to say.

The most-used class in the test suite. A whole session -- many turns, tool
calls, errors, a turn-limit overrun -- runs in microseconds with no key and no
network. It is also how the suite plays a *badly behaved* model: editing before
reading, walking out of the sandbox, sending arguments that do not match the
schema. A recording of a well-behaved model never contains those, and they are
exactly what the safety rules exist for.

Exported from the package rather than kept in tests/ so that people writing
their own tools can test them the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nanoclaude.conversation.transcript import Block, TextBlock, ToolUseBlock
from nanoclaude.providers.base import ModelError, ModelReply, ModelRequest, StopKind, Usage
from nanoclaude.providers.capabilities import Capabilities, capabilities_for


class ScriptedModel:
    def __init__(
        self,
        replies: Sequence[ModelReply],
        *,
        model_id: str = "scripted",
        capabilities: Capabilities | None = None,
    ) -> None:
        self._replies = list(replies)
        self._model_id = model_id
        self.capabilities = capabilities or capabilities_for("anthropic", "claude-sonnet-5")
        self.requests: list[ModelRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def exhausted(self) -> bool:
        return not self._replies

    async def complete(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request)
        if not self._replies:
            raise ModelError(
                f"scripted model ran out of replies after {len(self.requests)} requests"
            )
        return self._replies.pop(0)


def says(text: str, *, input_tokens: int = 0, output_tokens: int = 0) -> ModelReply:
    return ModelReply(
        (TextBlock(text),), StopKind.END_TURN, Usage(input_tokens, output_tokens), "scripted"
    )


def calls(
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str = "call_1",
    preamble: str | None = None,
) -> ModelReply:
    blocks: tuple[Block, ...] = (ToolUseBlock(call_id, name, arguments),)
    if preamble is not None:
        blocks = (TextBlock(preamble), *blocks)
    return ModelReply(blocks, StopKind.TOOL_USE, Usage(), "scripted")


def calls_many(*requested: tuple[str, Mapping[str, Any], str]) -> ModelReply:
    blocks = tuple(ToolUseBlock(call_id, name, args) for name, args, call_id in requested)
    return ModelReply(blocks, StopKind.TOOL_USE, Usage(), "scripted")
