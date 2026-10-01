# src/nanoclaude/tools/write.py
"""Write: create a file, or replace one the session has already looked at."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import ToolContext, ToolOutcome, failed, ok, require_str
from nanoclaude.tools.fs import stamp_of, write_atomic

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

        current = stamp_of(resolved)
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
        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return ok(
            call_id, f"{verb} {shown} ({stamp.size} bytes, {lines} lines)", ((resolved, stamp),)
        )
