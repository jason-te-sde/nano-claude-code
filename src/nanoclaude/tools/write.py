# src/nanoclaude/tools/write.py
"""Write: create a file, or replace one the session has already looked at."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolOutcome,
    failed,
    ok,
    require_str,
)
from nanoclaude.tools.edit import unified_diff
from nanoclaude.tools.fs import FileSystemError, read_text, stamp_of, write_atomic

DESCRIPTION = """Write a UTF-8 text file, creating it or replacing it entirely.

- Prefer Edit for changing part of a file. Use Write for new files or a full rewrite.
- Replacing a file that already exists requires having Read it in this session, and
  fails if it changed since. Read it again in that case.
- The write is atomic: if it fails, the previous content is still intact."""

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File to write."},
        "content": {"type": "string", "description": "Complete new contents."},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def line_count(content: str) -> int:
    """How many lines ``content`` has: its newlines, and one more for a last line without one."""
    return content.count("\n") + (0 if content.endswith("\n") or not content else 1)


@dataclass(frozen=True, slots=True)
class WritePreview:
    """What a Write call would do to the file as it is now. See :func:`preview_write`."""

    #: ``create``: the file is not there. ``replace``: it is, and ``diff`` is what changes.
    #: ``unchanged``: it already holds this content. ``overwrite``: it is there and cannot
    #: be compared, and ``why`` says why (it is binary, too large, a directory, unreadable).
    kind: Literal["create", "replace", "unchanged", "overwrite"]
    #: What would be written, and how many lines and bytes (as UTF-8) that is.
    content: str
    lines: int
    size: int
    diff: str = ""
    why: str = ""


def preview_write(path: str, shown: str, arguments: Mapping[str, Any]) -> WritePreview:
    """What :meth:`WriteTool.run` would do for ``arguments``, without doing it.

    What a confirmation shows before a person says yes. ``path`` is the resolved file and
    ``shown`` is the name a diff carries. It raises ``ToolArgumentError`` for a call that
    could not be written at all: no content, content that is not text, or text that cannot
    be encoded. It does not know what the session has read, so it cannot say that ``run``
    will refuse to replace a file for not having been read.
    """
    content = require_str(arguments, "content")
    try:
        size = len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ToolArgumentError(f"content cannot be written as UTF-8: {exc}") from exc
    lines = line_count(content)
    try:
        current = read_text(path)
    except FileNotFoundError:
        return WritePreview("create", content, lines, size)
    except OSError as exc:
        return WritePreview("overwrite", content, lines, size, why=str(exc))
    if current.content == content:
        return WritePreview("unchanged", content, lines, size)
    return WritePreview(
        "replace", content, lines, size, diff=unified_diff(current.content, content, shown)
    )


class WriteTool:
    name = "Write"
    read_only = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, DESCRIPTION, SCHEMA)

    def permission_request(
        self, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        resolved = ctx.resolve(require_str(arguments, "path"))
        return PermissionRequest(self.name, resolved, (resolved,), is_write=True)

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        resolved = ctx.resolve(require_str(arguments, "path"))
        content = require_str(arguments, "content")
        shown = ctx.display(resolved)

        try:
            current = stamp_of(resolved)
        except FileSystemError as exc:
            return failed(call_id, str(exc))
        if current is not None:
            seen = ctx.read_state.get(resolved)
            if seen is None:
                return failed(
                    call_id,
                    f"{shown} already exists and has not been read in this session. "
                    "Read it first, then write.",
                )
            if seen.sha256 != current.sha256:
                return failed(
                    call_id,
                    f"{shown} changed since it was read ({seen.size} bytes then, "
                    f"{current.size} now). Read it again before writing.",
                )

        stamp = write_atomic(resolved, content)
        verb = "Updated" if current is not None else "Created"
        return ok(
            call_id,
            f"{verb} {shown} ({stamp.size} bytes, {line_count(content)} lines)",
            ((resolved, stamp),),
        )
