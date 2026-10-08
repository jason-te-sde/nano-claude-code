"""One adapter for every endpoint that speaks the OpenAI chat-completions shape.

OpenAI, OpenRouter, Groq, DeepSeek, Together, Azure, vLLM, LM Studio and
llama.cpp's server all do. Writing this against the *format* rather than
against OpenAI is the highest reach-per-line decision in the project, so vendor
names stay out of the logic but for one choice: which parameter carries the
output cap (see ``_COMPLETION_TOKENS_DOMAINS``).

Two places where reality and the documentation differ, both handled:
``function.arguments`` is documented as a JSON string and is sometimes an
object already, and reasoning content arrives under three different key names
depending on who is serving.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from nanoclaude.conversation.transcript import (
    Block,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
)
from nanoclaude.providers.base import (
    CredentialsError,
    EmptyReplyError,
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    ToolSpec,
    Usage,
    new_call_id,
    partial_reply,
)
from nanoclaude.providers.retry import classify_status, classify_stream_error

_FINISH = {
    "stop": StopKind.END_TURN,
    "tool_calls": StopKind.TOOL_USE,
    "function_call": StopKind.TOOL_USE,
    "length": StopKind.MAX_TOKENS,
    "content_filter": StopKind.REFUSAL,
}

#: Providers disagree on where reasoning lives.
_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")

#: Domains whose hosts take max_completion_tokens: OpenAI's own API with its
#: regional hosts (eu.api.openai.com and the like), and Azure OpenAI. There
#: max_tokens is deprecated and reasoning models refuse it. The other servers
#: this adapter talks to mostly know only max_tokens, so they keep it.
_COMPLETION_TOKENS_DOMAINS = ("api.openai.com", "openai.azure.com")


def _output_cap_parameter(base_url: str) -> str:
    try:
        host = (urlsplit(base_url).hostname or "").rstrip(".")
    except ValueError:  # a malformed address, which the request itself reports
        return "max_tokens"
    first_party = any(
        host == domain or host.endswith(f".{domain}") for domain in _COMPLETION_TOKENS_DOMAINS
    )
    return "max_completion_tokens" if first_party else "max_tokens"


def parse_arguments(raw: Any) -> Mapping[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ModelError(f"tool arguments arrived as {type(raw).__name__}")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelError(f"tool arguments were not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ModelError(f"tool arguments decoded to {type(decoded).__name__}")
    return decoded


def encode_messages(transcript: Transcript) -> list[dict[str, Any]]:
    """Blocks in, chat-completions messages out.

    A tool result is its own message with ``role: tool``, not a content block,
    so one of our user messages can become several of theirs.
    """
    messages: list[dict[str, Any]] = []
    for message in transcript.messages:
        if message.role == "assistant":
            text = "".join(b.text for b in message.blocks if isinstance(b, TextBlock))
            calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(dict(b.arguments))},
                }
                for b in message.blocks
                if isinstance(b, ToolUseBlock)
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
            continue

        results = [b for b in message.blocks if isinstance(b, ToolResultBlock)]
        text = "".join(b.text for b in message.blocks if isinstance(b, TextBlock))
        if text:
            messages.append({"role": "user", "content": text})
        for result in results:
            messages.append(
                {"role": "tool", "tool_call_id": result.tool_use_id, "content": result.content}
            )
    return messages


def encode_tools(specs: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": dict(s.schema),
            },
        }
        for s in specs
    ]


class ChunkAccumulator:
    """Folds chat-completion chunks into one reply.

    ``on_text`` hears each piece of visible text as it arrives. Reasoning and tool
    arguments are not text, and never reach it.
    """

    def __init__(self, *, model: str, on_text: Callable[[str], None] | None = None) -> None:
        self._model = model
        self._on_text = on_text
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, dict[str, Any]] = {}
        self._stop = StopKind.END_TURN
        self._usage = Usage()
        # Whether any choice reported a finish_reason. Without it a tool call's
        # arguments may simply have stopped arriving, and an empty or partial
        # argument string must not be passed to a tool as if it were complete.
        self._finished = False

    def end(self) -> None:
        """The server said the stream is over (``[DONE]``), which is an end marker too."""
        self._finished = True

    def handle(self, chunk: Mapping[str, Any]) -> None:
        if "error" in chunk:
            error = chunk["error"]
            message = error.get("message", error) if isinstance(error, Mapping) else error
            # An error in the middle of the reply is the stream breaking, like a dropped
            # connection: what arrived is kept (and is not asked for again).
            raise ModelError(str(message), partial=self.partial())
        usage = chunk.get("usage")
        if usage:
            self._usage = Usage(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                0,
            )
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason")
            if finish:
                self._finished = True
                self._stop = _FINISH.get(finish, StopKind.END_TURN)
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self._text.append(delta["content"])
                if self._on_text is not None:
                    self._on_text(delta["content"])
            for key in _REASONING_KEYS:
                if delta.get(key):
                    self._reasoning.append(delta[key])
                    break
            for fragment in delta.get("tool_calls") or []:
                index = int(fragment.get("index", 0))
                slot = self._calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if fragment.get("id"):
                    slot["id"] = fragment["id"]
                function = fragment.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                arguments = function.get("arguments")
                if arguments:
                    # Usually a string fragment; some servers send the whole object.
                    slot["arguments"] += (
                        arguments if isinstance(arguments, str) else json.dumps(arguments)
                    )

    def partial(self) -> ModelReply | None:
        """What had arrived, as a reply; None while nothing a person could have read has.

        Visible text, or a tool call that had begun. Reasoning does not count: it is not
        shown and cannot have run anything, so a stream that broke after nothing but
        reasoning may be asked again. A call that had begun is not in the reply, whole or
        not (see :func:`~nanoclaude.providers.base.partial_reply`), but it is why a stream
        with no text yet is not asked again: an earlier call in the turn may have run.
        """
        text = "".join(self._text)
        if not text and not self._calls:
            return None
        return partial_reply([text], self._usage, self._model)

    def result(self) -> ModelReply:
        blocks: list[Block] = []
        if self._reasoning:
            blocks.append(ThinkingBlock("".join(self._reasoning)))
        if self._text:
            blocks.append(TextBlock("".join(self._text)))
        if (self._text or self._calls or self._reasoning) and not self._finished:
            # No finish_reason and no [DONE]: the stream closed without saying the reply was
            # whole (spec 7.4: interrupted midway, whatever closed it). Prose is held to it
            # as well as calls: a reply that stops in the middle of a sentence has been cut
            # off. A call must not be handed over either, whatever its arguments look like,
            # for an earlier call in the turn may already have run.
            what = "its tool calls were" if self._calls else "its reply was"
            raise ModelError(
                f"the response stream ended before {what} complete; the turn "
                "may have partially executed, so it is not retried automatically",
                retryable=False,
                partial=self.partial(),
            )
        for index in sorted(self._calls):
            slot = self._calls[index]
            if not slot["name"]:
                raise ModelError(f"tool call {index} arrived with no function name")
            blocks.append(
                ToolUseBlock(
                    slot["id"] or new_call_id("call"),
                    slot["name"],
                    parse_arguments(slot["arguments"]),
                )
            )
        if not blocks:
            if not self._finished:
                # Not one chunk said the reply was over, and none carried anything: a page, an
                # empty body, a whole reply that was not streamed. It never was a reply stream.
                raise ModelError("the provider's answer was not a reply stream")
            raise EmptyReplyError("the provider returned no content")
        return ModelReply(tuple(blocks), self._stop, self._usage, self._model)


class OpenAICompatClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 600.0,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        # api_key_env only names the variable in the error. One adapter serves
        # OpenRouter, Groq, DeepSeek and the rest, and each model alias in the
        # config says which variable holds its key (spec 17.6, api_key_env), so
        # telling an OpenRouter user to set OPENAI_API_KEY would be wrong.
        if not api_key:
            raise CredentialsError(
                f'no API key for adapter "openai_compat" — set {api_key_env} or run: ncc init'
            )
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._base_url = base_url.rstrip("/")
        self._output_cap = _output_cap_parameter(base_url)
        self._headers = {"content-type": "application/json", "authorization": f"Bearer {api_key}"}

    @property
    def model_id(self) -> str:
        return self._model

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            self._output_cap: request.max_output_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                *encode_messages(request.transcript),
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.tools:
            body["tools"] = encode_tools(request.tools)
        return body

    async def complete(
        self, request: ModelRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelReply:
        accumulator = ChunkAccumulator(model=self._model, on_text=on_text)
        url = f"{self._base_url}/chat/completions"
        try:
            async with self._client.stream(
                "POST", url, headers=self._headers, json=self.payload(request)
            ) as response:
                if not response.is_success:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise classify_status(response.status_code, body)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line.removeprefix("data:").strip()
                    if payload == "[DONE]":
                        accumulator.end()
                    elif payload:
                        accumulator.handle(json.loads(payload))
        except (httpx.RequestError, ValueError) as exc:
            failure = classify_stream_error(exc, accumulator.partial())
            if failure is None:
                raise
            raise failure from exc
        return accumulator.result()
