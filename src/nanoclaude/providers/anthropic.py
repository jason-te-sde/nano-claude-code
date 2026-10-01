"""The Anthropic adapter.

Streaming is used even though the loop only needs the finished reply: the CLI
shows text as it arrives, and a long tool-heavy turn is not a single silent
request that may or may not be progressing.

The consequence is that the interesting part is :class:`StreamAccumulator`, a
state machine over decoded events. It is testable from a recorded event list
with no HTTP involved, which is what the cassettes in tests/cassettes are.

Cache breakpoints go on the system prompt and the tool list, which are
byte-identical across turns. Anything that varies per request -- a timestamp,
tools in a different order -- costs the entire cache, so both are frozen.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from nanoclaude.conversation.transcript import (
    Block,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Transcript,
)
from nanoclaude.providers.base import (
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    ToolSpec,
    Usage,
)
from nanoclaude.providers.retry import classify_status

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"

_STOP = {
    "end_turn": StopKind.END_TURN,
    "stop_sequence": StopKind.END_TURN,
    "tool_use": StopKind.TOOL_USE,
    "max_tokens": StopKind.MAX_TOKENS,
    "refusal": StopKind.REFUSAL,
}


def encode_messages(transcript: Transcript) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in transcript.messages:
        content: list[dict[str, Any]] = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ThinkingBlock):
                content.append(
                    {"type": "thinking", "thinking": block.text, "signature": block.signature}
                )
            elif isinstance(block, ToolUseBlock):
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.arguments),
                    }
                )
            else:
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
        messages.append({"role": message.role, "content": content})
    return messages


def encode_tools(specs: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": s.name, "description": s.description, "input_schema": dict(s.schema)}
        for s in specs
    ]


class StreamAccumulator:
    """Folds server-sent events into one reply.

    Tool arguments arrive as JSON fragments that only parse once concatenated.
    A fragment stream that stops early is reported loudly: quietly passing ``{}``
    to a tool would look like the model asked for something it did not.
    """

    def __init__(self, *, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._kinds: dict[int, tuple[str, Any]] = {}
        self._json: dict[int, list[str]] = {}
        self._text: dict[int, list[str]] = {}
        # A thinking block's signature streams separately from its text and must
        # be sent back verbatim on the next turn, so it is kept per block.
        self._signature: dict[int, list[str]] = {}
        self._closed: set[int] = set()
        self._stop = StopKind.END_TURN
        self._usage = Usage()
        self._saw_message_stop = False

    def handle(self, event: str, data: Mapping[str, Any]) -> None:
        if event == "message_start":
            message = data.get("message", {})
            self._model = message.get("model", self._model)
            usage = message.get("usage", {})
            self._usage = Usage(
                usage.get("input_tokens", 0),
                0,
                usage.get("cache_read_input_tokens", 0),
                usage.get("cache_creation_input_tokens", 0),
            )
        elif event == "content_block_start":
            index, block = int(data["index"]), data["content_block"]
            kind = block["type"]
            if kind == "text":
                self._kinds[index] = ("text", None)
                self._text[index] = []
            elif kind == "thinking":
                self._kinds[index] = ("thinking", None)
                self._text[index] = []
                self._signature[index] = [str(block.get("signature", ""))]
            elif kind == "tool_use":
                self._kinds[index] = ("tool_use", (block["id"], block["name"]))
                self._json[index] = []
        elif event == "content_block_delta":
            index, delta = int(data["index"]), data["delta"]
            kind = delta["type"]
            if kind in ("text_delta", "thinking_delta"):
                self._text.setdefault(index, []).append(
                    delta.get("text") or delta.get("thinking", "")
                )
            elif kind == "input_json_delta":
                self._json.setdefault(index, []).append(delta["partial_json"])
            elif kind == "signature_delta":
                self._signature.setdefault(index, []).append(delta["signature"])
        elif event == "content_block_stop":
            self._closed.add(int(data["index"]))
        elif event == "message_delta":
            reason = data.get("delta", {}).get("stop_reason")
            if reason is not None:
                self._stop = _STOP.get(reason, StopKind.END_TURN)
            out = data.get("usage", {}).get("output_tokens")
            if out is not None:
                self._usage = Usage(
                    self._usage.input_tokens,
                    out,
                    self._usage.cache_read_tokens,
                    self._usage.cache_write_tokens,
                )
        elif event == "message_stop":
            self._saw_message_stop = True
        elif event == "error":
            error = data.get("error", {})
            raise ModelError(
                f"{error.get('type', 'error')}: {error.get('message', data)}",
                retryable=error.get("type") in {"overloaded_error", "rate_limit_error"},
            )

    def result(self) -> ModelReply:
        blocks: list[Block] = []
        for index in sorted(self._kinds):
            kind, meta = self._kinds[index]
            if index not in self._closed and not self._saw_message_stop:
                raise ModelError(
                    f"the response stream ended with content block {index} incomplete; "
                    "the turn may have partially executed, so it is not retried "
                    "automatically",
                    retryable=False,
                )
            if kind == "text":
                blocks.append(TextBlock("".join(self._text.get(index, ()))))
            elif kind == "thinking":
                blocks.append(
                    ThinkingBlock(
                        "".join(self._text.get(index, ())),
                        "".join(self._signature.get(index, ())),
                    )
                )
            else:
                call_id, name = meta
                blocks.append(ToolUseBlock(call_id, name, self._decode(index, name)))
        if not blocks:
            raise ModelError("the provider returned no content blocks")
        return ModelReply(tuple(blocks), self._stop, self._usage, self._model)

    def _decode(self, index: int, name: str) -> Mapping[str, Any]:
        raw = "".join(self._json.get(index, ()))
        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelError(
                f"arguments for {name} were incomplete or not valid JSON: {exc}",
                retryable=False,
            ) from exc
        if not isinstance(decoded, dict):
            raise ModelError(f"arguments for {name} decoded to {type(decoded).__name__}")
        return decoded


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event = ""
    payload: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if event and payload:
                yield event, json.loads("".join(payload))
            event, payload = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            payload.append(value)


class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        client: httpx.AsyncClient | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 600.0,
        cache: bool = True,
    ) -> None:
        if not api_key:
            raise ModelError(
                'no API key for adapter "anthropic" — set ANTHROPIC_API_KEY or run: ncc init',
                retryable=False,
            )
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._base_url = base_url.rstrip("/")
        self._cache = cache
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }

    @property
    def model_id(self) -> str:
        return self._model

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        system: list[dict[str, Any]] = [{"type": "text", "text": request.system}]
        tools = encode_tools(request.tools)
        if self._cache:
            system[-1]["cache_control"] = {"type": "ephemeral"}
            if tools:
                tools[-1]["cache_control"] = {"type": "ephemeral"}
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "system": system,
            "messages": encode_messages(request.transcript),
            "stream": True,
        }
        if tools:
            body["tools"] = tools
        return body

    async def complete(self, request: ModelRequest) -> ModelReply:
        accumulator = StreamAccumulator(model=self._model)
        url = f"{self._base_url}/v1/messages"
        try:
            async with self._client.stream(
                "POST", url, headers=self._headers, json=self.payload(request)
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise classify_status(response.status_code, body)
                async for event, data in iter_sse(response.aiter_lines()):
                    accumulator.handle(event, data)
        except httpx.TimeoutException as exc:
            raise ModelError(f"the provider timed out: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ModelError(f"could not reach the provider: {exc}", retryable=True) from exc
        return accumulator.result()
