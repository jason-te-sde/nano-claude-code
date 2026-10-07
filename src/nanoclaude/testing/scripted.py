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

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nanoclaude.conversation.transcript import Block, TextBlock, ToolUseBlock
from nanoclaude.providers.base import (
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    Usage,
    partial_reply,
)
from nanoclaude.providers.capabilities import Capabilities, capabilities_for

#: How many pieces a text block is cut into when a scripted reply is streamed.
_PIECES = 3


def _pieces(text: str) -> list[str]:
    """``text`` cut into up to three pieces of nearly equal length, which join to ``text``."""
    if not text:
        return []
    size = -(-len(text) // _PIECES)
    return [text[start : start + size] for start in range(0, len(text), size)]


def stream_text(reply: ModelReply, on_text: Callable[[str], None]) -> None:
    """Hand the text of ``reply`` to ``on_text`` the way an adapter does.

    In order, a text block at a time, each in a few pieces and none of them empty. Whole,
    the text would hide what streaming exists to expose: a front end that copes with the
    first piece and with nothing after it. Tool calls are not text and are not given.
    """
    for block in reply.blocks:
        if isinstance(block, TextBlock):
            for piece in _pieces(block.text):
                on_text(piece)


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

    async def complete(
        self, request: ModelRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelReply:
        self.requests.append(request)
        if not self._replies:
            raise ModelError(
                f"scripted model ran out of replies after {len(self.requests)} requests"
            )
        reply = self._replies.pop(0)
        if on_text is not None:
            stream_text(reply, on_text)
        return reply


def says(text: str, *, input_tokens: int = 0, output_tokens: int = 0) -> ModelReply:
    return ModelReply(
        (TextBlock(text),), StopKind.END_TURN, Usage(input_tokens, output_tokens), "scripted"
    )


def cut_off(
    text: str, *, input_tokens: int = 0, output_tokens: int = 0, message: str | None = None
) -> ModelError:
    """The error of a stream that breaks after ``text`` had arrived (spec 7.4).

    For a script that plays a provider whose connection drops midway: a client raises it
    after streaming ``text``. With no text, it is a stream that broke inside its first
    tool call, which leaves nothing to keep.
    """
    return ModelError(
        message or "the connection to the provider was lost: connection reset by peer",
        retryable=False,
        partial=partial_reply([text], Usage(input_tokens, output_tokens), "scripted"),
    )


def calls(
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str = "call_1",
    preamble: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> ModelReply:
    blocks: tuple[Block, ...] = (ToolUseBlock(call_id, name, arguments),)
    if preamble is not None:
        blocks = (TextBlock(preamble), *blocks)
    return ModelReply(blocks, StopKind.TOOL_USE, Usage(input_tokens, output_tokens), "scripted")


def calls_many(
    *requested: tuple[str, Mapping[str, Any], str],
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> ModelReply:
    blocks = tuple(ToolUseBlock(call_id, name, args) for name, args, call_id in requested)
    return ModelReply(blocks, StopKind.TOOL_USE, Usage(input_tokens, output_tokens), "scripted")
