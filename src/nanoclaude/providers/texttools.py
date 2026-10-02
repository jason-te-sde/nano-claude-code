"""Tool calling for models that cannot call tools.

The definitions go into the system prompt with a grammar to answer in, and the
answer is parsed back into the same blocks a native call would have produced.
Above this module nothing knows the difference.

This is the least reliable path in the project and it is documented as such:
small models emit malformed JSON, forget the closing tag, and wrap everything
in code fences. Two of those three are tolerated here; the third is reported to
the model so it can try again, at most :data:`MAX_PARSE_RETRIES` times before
the session gives up and says this model is not suitable.

The grammar is XML-ish rather than JSON because a model that cannot emit a
well-formed function call reliably also cannot emit a well-formed JSON envelope
reliably, and a broken tag is easier to detect than a broken brace.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence

from nanoclaude.conversation.transcript import Block, TextBlock, ToolUseBlock
from nanoclaude.providers.base import ModelReply, StopKind, ToolSpec

MAX_PARSE_RETRIES = 2

_OPEN = re.compile(r"<tool\s+name=\"([A-Za-z_][A-Za-z0-9_]*)\"\s*>", re.IGNORECASE)
_CLOSE = "</tool>"

TOOL_PROTOCOL_PROMPT = """\
You do not have a function-calling interface. To use a tool, write a block in \
exactly this form and nothing else on those lines:

<tool name="ToolName">
{"argument": "value"}
</tool>

Rules:
- The content between the tags must be a single valid JSON object.
- One block per tool call. You may write several blocks in one reply.
- Write the block exactly as shown. Do not rename the tags or add attributes.
- You may write ordinary prose before or after a block; it is shown to the user.
- After a block, stop and wait. The result comes back in the next message.

Available tools:
"""


def render_tools(specs: Sequence[ToolSpec]) -> str:
    parts: list[str] = []
    for spec in specs:
        schema = json.dumps(dict(spec.schema), indent=2, sort_keys=True)
        parts.append(f"### {spec.name}\n{spec.description}\n\nArguments (JSON schema):\n{schema}")
    return TOOL_PROTOCOL_PROMPT + "\n\n".join(parts)


def parse_text_tools(text: str, *, next_id: Callable[[], str]) -> tuple[list[Block], list[str]]:
    """Split a reply into prose and tool calls. Returns (blocks, problems)."""
    blocks: list[Block] = []
    problems: list[str] = []
    position = 0

    while True:
        match = _OPEN.search(text, position)
        if match is None:
            break
        prose = _clean(text[position : match.start()])
        if prose:
            blocks.append(TextBlock(prose))
        name = match.group(1)
        end = text.find(_CLOSE, match.end())
        if end == -1:
            problems.append(f'the <tool name="{name}"> block has no closing </tool> tag')
            position = match.end()
            continue
        payload = text[match.end() : end].strip().strip("`").strip()
        try:
            arguments = json.loads(payload)
        except json.JSONDecodeError as exc:
            problems.append(
                f"the arguments for {name} were not valid JSON ({exc.msg}). "
                "Write a single JSON object between the tags."
            )
        else:
            if isinstance(arguments, dict):
                blocks.append(ToolUseBlock(next_id(), name, arguments))
            else:
                problems.append(
                    f"the arguments for {name} were a {type(arguments).__name__}, not a JSON object"
                )
        position = end + len(_CLOSE)

    tail = _clean(text[position:])
    if tail:
        blocks.append(TextBlock(tail))
    if not blocks:
        blocks.append(TextBlock(text.strip()))
    return blocks, problems


def _clean(fragment: str) -> str:
    """Drop the code fences small models wrap everything in."""
    stripped = fragment.strip()
    if stripped in ("```", "```json", "```xml"):
        return ""
    return stripped.strip("`").strip() if stripped.startswith("```") else stripped


def wrap_reply(reply: ModelReply, *, next_id: Callable[[], str]) -> ModelReply:
    """Re-parse a plain-text reply as though it had been a native tool call."""
    text = "".join(b.text for b in reply.blocks if isinstance(b, TextBlock))
    if "<tool" not in text:
        return reply
    blocks, problems = parse_text_tools(text, next_id=next_id)
    calls = [b for b in blocks if isinstance(b, ToolUseBlock)]
    if problems and not calls:
        note = "\n".join(f"- {p}" for p in problems)
        return ModelReply(
            (TextBlock(f"{text}\n\n[tool protocol] {note}"),),
            reply.stop,
            reply.usage,
            reply.model,
        )
    stop = StopKind.TOOL_USE if calls else reply.stop
    return ModelReply(tuple(blocks), stop, reply.usage, reply.model)
