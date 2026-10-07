"""Ollama, using its native API rather than its OpenAI-compatible endpoint.

The compatibility endpoint exists and the adapter in Task 18 would talk to it.
The native API is used anyway for one reason: ``/api/show`` returns the model's
prompt template, and whether that template mentions tools is the only reliable
way to know before the first request whether this model can call them. Guessing
wrong means a confusing failure several turns into someone's first session.

Two divergences from a naive reading of the wire format, both load-bearing:
``function.arguments`` is documented as an object and some templates emit a
JSON string instead, so both are accepted; and the final ``"done": true`` line is
the only thing that says a reply is whole. A stream that ends without it, or that
carries an ``{"error": ...}`` line (the server reports a runner that died that way,
and then closes the stream cleanly), was interrupted midway (spec 7.4). What arrived
of its text is kept, its calls are dropped, since the tool may already have run
partially, and nothing is asked again.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from nanoclaude.conversation.transcript import (
    Block,
    TextBlock,
    ToolResultBlock,
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
    new_call_id,
    partial_reply,
)
from nanoclaude.providers.capabilities import CONSERVATIVE_DEFAULT, Capabilities
from nanoclaude.providers.retry import classify_status, connection_lost

DEFAULT_BASE_URL = "http://localhost:11434"


def parse_arguments(raw: Any) -> Mapping[str, Any]:
    """Ollama's native shape is an object; some templates emit a JSON string instead."""
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


def encode_messages(system: str, transcript: Transcript) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in transcript.messages:
        text = "".join(b.text for b in message.blocks if isinstance(b, TextBlock))
        calls = [b for b in message.blocks if isinstance(b, ToolUseBlock)]
        results = [b for b in message.blocks if isinstance(b, ToolResultBlock)]
        if message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": dict(c.arguments)}} for c in calls
                ]
            messages.append(entry)
            continue
        if text:
            messages.append({"role": "user", "content": text})
        for result in results:
            messages.append({"role": "tool", "content": result.content})
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


class OllamaClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._base_url = base_url.rstrip("/")

    @property
    def model_id(self) -> str:
        return self._model

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def payload(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": encode_messages(request.system, request.transcript),
            "stream": True,
            "options": {"num_predict": request.max_output_tokens},
        }
        if request.temperature is not None:
            body["options"]["temperature"] = request.temperature
        if request.tools:
            body["tools"] = encode_tools(request.tools)
        return body

    async def complete(
        self, request: ModelRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelReply:
        text: list[str] = []
        calls: list[ToolUseBlock] = []
        usage = Usage()
        done = False
        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/api/chat", json=self.payload(request)
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise classify_status(response.status_code, body)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict):
                        # Valid JSON need not be an object -- an unexpected
                        # line must become a ModelError, not an AttributeError
                        # from calling .get() on a list or a string below.
                        raise ModelError(f"ollama sent an unexpected stream chunk: {chunk!r}")
                    if chunk.get("error"):
                        # The server's own report that the reply will not be finished, on a
                        # line of its own and followed by a clean close. Spec 7.4: whatever
                        # arrived is kept, and the reason is the server's, not ours.
                        raise ModelError(
                            f"ollama reported an error: {chunk['error']}",
                            retryable=False,
                            partial=(
                                partial_reply(["".join(text)], usage, self._model)
                                if text or calls
                                else None
                            ),
                        )
                    message = chunk.get("message") or {}
                    if message.get("content"):
                        text.append(message["content"])
                        if on_text is not None:
                            on_text(message["content"])
                    for raw in message.get("tool_calls") or []:
                        function = raw.get("function") or {}
                        calls.append(
                            ToolUseBlock(
                                raw.get("id") or new_call_id("call"),
                                function.get("name", ""),
                                parse_arguments(function.get("arguments")),
                            )
                        )
                    if chunk.get("done"):
                        done = True
                        usage = Usage(
                            chunk.get("prompt_eval_count") or 0, chunk.get("eval_count") or 0
                        )
        except httpx.TransportError as exc:
            if text or calls:
                # Spec 7.4: some of the reply had arrived, so it is kept and not asked for
                # again. The server was up and answering, so this is not "start ollama".
                partial = partial_reply(["".join(text)], usage, self._model)
                raise connection_lost(exc, partial, peer="ollama") from exc
            if isinstance(exc, httpx.TimeoutException):
                raise ModelError(f"ollama timed out: {exc}", retryable=True) from exc
            raise ModelError(
                f"could not reach ollama at {self._base_url}: {exc} — start it with: ollama serve",
                retryable=False,
            ) from exc

        if (text or calls) and not done:
            # The stream closed without the line that says the reply is whole (spec 7.4:
            # interrupted midway, whatever the cause). Handing a call over anyway risks
            # running a tool on arguments the model never finished sending, and asking
            # again is wrong too, since an earlier call in the same turn may already have
            # run. Plain prose is held to the same rule: a reply that stops in the middle
            # of a sentence is not a shorter reply, it is a reply that was cut off, and
            # the person has to be told so.
            what = "its tool calls were" if calls else "its reply was"
            raise ModelError(
                f"the response stream ended before {what} complete; the turn "
                "may have partially executed, so it is not retried automatically",
                retryable=False,
                partial=partial_reply(["".join(text)], usage, self._model),
            )

        blocks: list[Block] = []
        if text:
            blocks.append(TextBlock("".join(text)))
        blocks.extend(calls)
        stop = StopKind.TOOL_USE if calls else StopKind.END_TURN
        if not blocks:
            raise ModelError("ollama returned no content")
        return ModelReply(tuple(blocks), stop, usage, self._model)


async def probe_ollama(
    model: str, *, base_url: str = DEFAULT_BASE_URL, client: httpx.AsyncClient | None = None
) -> Capabilities | None:
    """Ask the server what this model's template supports, or None if it could not say.

    Anything unexpected -- server down or not up yet, a timeout, model not pulled,
    template unreadable, a response that is valid JSON but not the object this
    endpoint documents -- is None. It is deliberately not the conservative default:
    a caller that remembered that would remember "no native tools" for a model that
    was only unreachable for a moment. The caller falls back to the default for now
    and asks again next time. Being wrong in that direction costs speed; the other
    direction costs someone's first session.
    """
    owns = client is None
    http = client or httpx.AsyncClient(timeout=10)
    try:
        response = await http.post(f"{base_url.rstrip('/')}/api/show", json={"model": model})
        if response.status_code >= 400:
            return None
        data = response.json()
        if not isinstance(data, dict):
            # Valid JSON need not be an object: an unexpected response (a
            # proxy's error page, a bare string or list) must degrade to "we
            # do not know" rather than raise out of .get() below.
            return None
        template = str(data.get("template", ""))
        info = data.get("model_info")
        if not isinstance(info, dict):
            info = {}
        window = next(
            (int(v) for k, v in info.items() if k.endswith("context_length")),
            CONSERVATIVE_DEFAULT.context_window,
        )
    except (httpx.HTTPError, ValueError, TypeError):
        # The failures that mean "we do not know": the server is unreachable or
        # timed out, the body is not JSON, or a context length is present but
        # not a number. Named rather than caught as Exception, so that a
        # programming error below surfaces instead of quietly making every
        # model look tool-less.
        return None
    finally:
        if owns:
            await http.aclose()
    return Capabilities(
        native_tools=".Tools" in template or "tools" in template.lower(),
        parallel_tools=False,  # no local model has been reliable at this
        cache="none",
        context_window=window,
        max_output=min(4096, window // 4),
    )
