# src/nanoclaude/mcp/result.py
"""What an MCP tool's answer becomes before the model reads it.

Everything in a result is somebody else's text. This module makes it text, and then makes it
safe to hand over, in a fixed order that each step depends on:

1. **Content into text.** Text blocks as they are, joined by newlines. A block of anything
   else is replaced by a line that says what was left out and nothing a server could use
   to talk to the model: an image or audio block by its type and the size of what its data
   decodes to, a resource by its URI, with no text of the resource. With no text block, the
   result's ``structuredContent`` is given as JSON. Nothing is ever fetched or opened.
2. **Cleaned and scrubbed**, before anything is added. Terminal controls are removed first, so
   that no later step sees a different text from the one the model will, and then credentials
   are scrubbed, so that the cut below cannot leave half of one.
3. **Wrapped** in ``<mcp-result server="..." tool="...">`` and ``</mcp-result>``, which says
   where the server's words begin and end. Every ``</mcp-result`` in the content, in any case,
   is written ``<\\/mcp-result`` first, so that nothing a server sends can end the wrapper
   early. This is done on the cleaned text: a tag with a control character in the middle is
   the tag once the control is gone.
4. **Held to the output cap.** The cut is made on the content, to what is left of the cap
   after the wrapper, so that the closing tag is always there; the note that says how many
   bytes were cut is written before it. A cap smaller than the wrapper leaves the wrapper
   whole and the content empty.
5. **Cleaned and scrubbed once more**, as every tool's result is, over the whole.

The attributes of the wrapper are the server's configured name and the tool's registry name,
which are made of ASCII letters, digits and underscores and cannot carry a quotation mark.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from nanoclaude.mcp.client import CallResult
from nanoclaude.permissions.redact import Redactor
from nanoclaude.tools.base import cap_bytes, sanitize, without_lone_surrogates

#: What a result with no content at all is given as, so that the wrapper is never empty.
NO_CONTENT = "(the tool returned no content)"

#: A MIME type, as far as a placeholder repeats one: ``type/subtype`` of the characters RFC 6838
#: allows in a name, and short. Anything else is called unknown, so that the line says what was
#: left out and is not a place for a server to write in.
_MIME = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,60}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,60}")

#: The longest a URI or a block's kind is shown.
_SHOWN_CHARS = 200

#: What would end the wrapper, in any case: the case a server used is kept when it is written
#: so that it does not.
_CLOSING_TAG = re.compile(r"</(mcp-result)", re.IGNORECASE)


def _shown(value: object) -> str | None:
    """``value`` as one short line of text, or None when it is not text."""
    if not isinstance(value, str):
        return None
    line = " ".join(without_lone_surrogates(sanitize(value)).split())
    return line if len(line) <= _SHOWN_CHARS else line[: _SHOWN_CHARS - 1] + "…"


def _decoded_size(data: object) -> int:
    """How many bytes the base64 text ``data`` stands for; 0 when it is not text."""
    if not isinstance(data, str):
        return 0
    padding = len(data) - len(data.rstrip("="))
    return max(len(data) * 3 // 4 - padding, 0)


def _placeholder(block: object) -> tuple[bool, str]:
    """``(is_text, line)`` for one content block: its text, or the line that stands for it."""
    if not isinstance(block, Mapping) or not isinstance(block.get("type"), str):
        return False, "[unrecognised content omitted]"
    kind: str = block["type"]
    if kind == "text":
        value = block.get("text")
        if isinstance(value, str):
            return True, value
        return False, "[text omitted: it holds no text]"
    if kind in ("image", "audio"):
        mime = block.get("mimeType")
        shown = mime if isinstance(mime, str) and _MIME.fullmatch(mime) else "unknown type"
        return False, f"[{kind} omitted: {shown}, {_decoded_size(block.get('data'))} bytes]"
    if kind in ("resource", "resource_link"):
        resource = block.get("resource") if kind == "resource" else block
        uri = _shown(resource.get("uri")) if isinstance(resource, Mapping) else None
        return False, f"[resource omitted: {uri or 'unknown uri'}]"
    return False, f"[{_shown(kind) or 'unrecognised'} content omitted]"


def result_text(result: CallResult) -> str:
    """The content of ``result`` as text: see the module docstring, step 1."""
    parts: list[str] = []
    has_text = False
    for block in result.content:
        is_text, line = _placeholder(block)
        has_text = has_text or is_text
        parts.append(line)
    if not has_text and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, ensure_ascii=False))
    text = "\n".join(parts) if parts else NO_CONTENT
    return without_lone_surrogates(text)


def _clean(text: str, redactor: Redactor) -> str:
    """Controls removed first, then credentials scrubbed: a secret that an escape sequence
    splits would otherwise get past the scrubber."""
    scrubbed, _ = redactor.scrub(sanitize(text))
    return scrubbed


def _written_safe(text: str) -> str:
    """``text`` with each ``</mcp-result`` written ``<\\/mcp-result``."""
    return _CLOSING_TAG.sub(r"<\\/\1", text)


def wrap_result(server: str, tool: str, content: str, *, cap: int, redactor: Redactor) -> str:
    """``content`` as the model is given it: see the module docstring, steps 2 to 5.

    ``server`` and ``tool`` are the attributes of the wrapper; ``cap`` is
    the session's output-cap limit, in bytes.
    """
    opening = f'<mcp-result server="{server}" tool="{tool}">\n'
    closing = "\n</mcp-result>"
    room = max(cap - len((opening + closing).encode("utf-8")), 0)
    body = cap_bytes(_written_safe(_clean(without_lone_surrogates(content), redactor)), room)
    return _clean(opening + body + closing, redactor)
