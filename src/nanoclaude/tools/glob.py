# src/nanoclaude/tools/glob.py
"""Glob: find files by pattern, most recently modified first.

The ordering is the useful part. In a repository someone is working in, the
files they touched last are almost always the relevant ones, so a truncated
list is still a good list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import ToolContext, ToolOutcome, ok, require_str
from nanoclaude.tools.search import DEFAULT_LIMIT, walk_files

DESCRIPTION = """Find files by glob pattern, most recently modified first.

- `pattern` is a glob such as `src/**/*.py` or `**/test_*.py`.
- `path` limits the search to a subdirectory (default: the working directory).
- Paths ignored by .gitignore are skipped. At most 200 results are returned."""

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string"},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


class GlobTool:
    name = "Glob"
    read_only = True

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, DESCRIPTION, SCHEMA)

    def permission_request(
        self, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        base = arguments.get("path")
        paths = (ctx.resolve(base),) if isinstance(base, str) and base else ()
        return PermissionRequest(self.name, require_str(arguments, "pattern"), paths)

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        pattern = require_str(arguments, "pattern")
        base = arguments.get("path")
        root = ctx.resolve(base) if isinstance(base, str) and base else ctx.root
        found = walk_files(root, pattern=pattern, limit=DEFAULT_LIMIT)
        if not found:
            return ok(call_id, f"No files matched {pattern} under {ctx.display(root)}")
        listing = "\n".join(ctx.display(path) for path in found)
        header = f"{len(found)} file(s) matching {pattern}"
        if len(found) == DEFAULT_LIMIT:
            header += " (truncated at 200; narrow the pattern)"
        return ok(call_id, f"{header}\n{listing}")
