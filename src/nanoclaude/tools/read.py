# src/nanoclaude/tools/read.py
"""Read: show the model a file, with line numbers it must not rely on.

The numbers let the model talk about the file and choose a next window. They are
deliberately not an addressing scheme for edits -- Edit matches on content, for
the reasons in docs/design/0007-edit-by-string-not-lineno.md. The prefix is one
number, one tab, then the line exactly as it is on disk, so stripping it is
unambiguous.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import (
    ToolContext,
    ToolOutcome,
    failed,
    ok,
    optional_int,
    require_str,
)
from nanoclaude.tools.fs import FileSystemError, read_text

DEFAULT_LIMIT = 2000
MAX_LINE_CHARS = 2000

# Copied verbatim from spec 17.10. This is a prompt, not documentation.
DESCRIPTION = """Read a UTF-8 text file from the working directory.

Output is one line per source line, prefixed with the line number and a single tab.
Everything after that tab is the file's own content; when you later pass a fragment to
Edit, do not include the number or the tab.

- `path` may be absolute or relative to the working directory.
- `offset` is the 1-based first line to show; `limit` is how many lines (default 2000).
- Files over 1 MB, non-UTF-8 files and binaries are refused rather than mangled. Use
  Grep to search a file that is too large to read.
- Reading a file records what it contained. Edit and Write require that record and will
  refuse if the file has changed since."""

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File to read."},
        "offset": {"type": "integer", "minimum": 1, "description": "First line, 1-based."},
        "limit": {"type": "integer", "minimum": 1, "description": "How many lines."},
    },
    "required": ["path"],
    "additionalProperties": False,
}


class ReadTool:
    name = "Read"
    read_only = True

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, DESCRIPTION, SCHEMA)

    def permission_request(
        self, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        resolved = ctx.resolve(require_str(arguments, "path"))
        return PermissionRequest(self.name, resolved, (resolved,), is_write=False)

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        resolved = ctx.resolve(require_str(arguments, "path"))
        offset = optional_int(arguments, "offset", 1, minimum=1)
        limit = optional_int(arguments, "limit", DEFAULT_LIMIT, minimum=1)
        shown = ctx.display(resolved)

        try:
            snapshot = read_text(resolved)
        except FileNotFoundError:
            return failed(call_id, f"{shown} does not exist")
        except IsADirectoryError:
            return failed(call_id, f"{shown} is a directory. Use Glob to list its contents.")
        except FileSystemError as exc:
            return failed(call_id, str(exc))

        lines = snapshot.content.splitlines()
        if lines and offset > len(lines):
            return failed(
                call_id,
                f"{shown} has {len(lines)} lines; offset {offset} is past the end",
            )

        window = lines[offset - 1 : offset - 1 + limit]
        header = f"{shown} ({len(lines)} lines)"
        if len(window) < len(lines):
            header += f", showing {offset}-{offset + len(window) - 1}"
        body = (
            "\n".join(f"{offset + i}\t{_clip(line)}" for i, line in enumerate(window))
            if lines
            else "(empty file)"
        )
        redacted, _ = ctx.redactor.scrub(f"{header}\n{body}")
        # The stamp covers the whole file, not the window: its job is to tell
        # "you have not looked at this" from "it changed since you did".
        return ok(call_id, redacted, ((resolved, snapshot.stamp),))


def _clip(line: str) -> str:
    if len(line) <= MAX_LINE_CHARS:
        return line
    return f"{line[:MAX_LINE_CHARS]}... [{len(line) - MAX_LINE_CHARS} more characters]"
