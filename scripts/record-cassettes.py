#!/usr/bin/env python3
"""Record provider SSE streams into tests/cassettes/.

Run once, with real keys, when a provider changes its wire format. The output
is committed, so the test suite never needs a key. Anything that looks like a
credential is stripped on the way out.

Usage: ANTHROPIC_API_KEY=... python scripts/record-cassettes.py anthropic
       OPENAI_API_KEY=...    python scripts/record-cassettes.py openai

Not run as part of this change: the four Anthropic cassettes and the two
openai_compat cassettes already in tests/cassettes/ are hand-written synthetic
fixtures (see that directory's README.md), because writing this script needs
no key but running it does, and none was available here. Running it later
replaces those files with real recordings.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from nanoclaude.conversation.transcript import Transcript, user_text
from nanoclaude.permissions.redact import Redactor
from nanoclaude.providers.anthropic import DEFAULT_MODEL, AnthropicClient, iter_sse
from nanoclaude.providers.base import ModelRequest, ToolSpec
from nanoclaude.providers.openai_compat import OpenAICompatClient
from nanoclaude.tools.registry import default_registry

CASSETTES = Path(__file__).resolve().parents[1] / "tests" / "cassettes"

SCENARIOS: dict[str, tuple[str, bool]] = {
    "anthropic_text": ("Reply with exactly the word: hello", False),
    "anthropic_tool_use": ("Read the file README.md. Use the Read tool.", True),
    "anthropic_parallel_tools": (
        "Read both README.md and pyproject.toml. Call Read twice in one turn.",
        True,
    ),
}

#: openai_compat talks to any endpoint that speaks the chat-completions shape;
#: this records against OpenAI itself, the one every other endpoint mirrors.
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-5"

OPENAI_SCENARIOS: dict[str, tuple[str, bool]] = {
    "openai_text": ("Reply with exactly the word: hello", False),
    "openai_tool_use": ("Read the file README.md. Use the Read tool.", True),
}


async def record(name: str, prompt: str, with_tools: bool) -> None:
    key = os.environ["ANTHROPIC_API_KEY"]
    tools: tuple[ToolSpec, ...] = default_registry().specs() if with_tools else ()
    client = AnthropicClient(key, model=DEFAULT_MODEL)
    request = ModelRequest("You are a test fixture.", Transcript((user_text(prompt),)), tools, 1024)
    redactor = Redactor()
    lines: list[str] = []
    async with (
        httpx.AsyncClient(timeout=120) as http,
        http.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
            json=client.payload(request),
        ) as response,
    ):
        response.raise_for_status()
        async for event, data in iter_sse(response.aiter_lines()):
            cleaned, _ = redactor.scrub(json.dumps({"event": event, "data": data}))
            lines.append(cleaned)
    (CASSETTES / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} events")


async def record_openai(name: str, prompt: str, with_tools: bool) -> None:
    """Record one openai_compat scenario, raw ``data:`` lines, against OpenAI.

    Unlike Anthropic's streams, a chat-completions chunk carries no SSE
    ``event:`` field at all, so ``iter_sse`` (which only yields once it has
    seen one) cannot parse this format -- every line here is read straight off
    ``aiter_lines()`` and its ``data:`` prefix stripped by hand, the same way
    ``OpenAICompatClient.complete`` itself does. Each payload is re-parsed and
    re-dumped rather than stored as raw text, so every committed line is its
    own valid JSON value -- a chunk object, or the literal sentinel string
    ``"[DONE]"`` standing in for the bare ``data: [DONE]`` the wire sends.
    """
    key = os.environ["OPENAI_API_KEY"]
    tools: tuple[ToolSpec, ...] = default_registry().specs() if with_tools else ()
    client = OpenAICompatClient(key, model=OPENAI_MODEL, base_url=OPENAI_BASE_URL)
    request = ModelRequest("You are a test fixture.", Transcript((user_text(prompt),)), tools, 1024)
    redactor = Redactor()
    lines: list[str] = []
    async with (
        httpx.AsyncClient(timeout=120) as http,
        http.stream(
            "POST",
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            json=client.payload(request),
        ) as response,
    ):
        response.raise_for_status()
        async for raw in response.aiter_lines():
            if not raw.startswith("data:"):
                continue
            text = raw.removeprefix("data:").strip()
            if not text:
                continue
            value: Any = "[DONE]" if text == "[DONE]" else json.loads(text)
            cleaned, _ = redactor.scrub(json.dumps(value))
            lines.append(cleaned)
    (CASSETTES / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} events")


async def main() -> None:
    CASSETTES.mkdir(parents=True, exist_ok=True)
    for name, (prompt, with_tools) in SCENARIOS.items():
        await record(name, prompt, with_tools)
    # The truncated fixture is made by hand from the tool-use one: drop the
    # trailing content_block_stop and message_stop. A provider will not produce
    # this on demand, and it is the case most likely to be wrong.
    source = (CASSETTES / "anthropic_tool_use.jsonl").read_text().splitlines()
    kept = [
        line
        for line in source
        if '"content_block_stop"' not in line and '"message_stop"' not in line
    ]
    (CASSETTES / "anthropic_truncated.jsonl").write_text("\n".join(kept) + "\n")
    print("anthropic_truncated: derived")

    for name, (prompt, with_tools) in OPENAI_SCENARIOS.items():
        await record_openai(name, prompt, with_tools)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
