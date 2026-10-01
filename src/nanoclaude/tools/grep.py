# src/nanoclaude/tools/grep.py
"""Grep: search file contents.

Three output modes because the right amount of detail differs by an order of
magnitude between "where is this defined" and "how widely is this used", and a
model that has to read 400 matching lines to answer the second question has
spent its context on the wrong thing.

Both search backends walk the tree themselves rather than consulting a list of
paths the caller already cleared, so a broad search can surface a file the
caller never named -- an un-ignored ``.env`` sitting next to code, say. That
file is covered by the secret-path rule tools/base.py's callers enforce before
a call that *names* it, but a path discovered mid-walk never goes through that
check. Matches are filtered against the same redactor-owned rule here, after
the backend returns, so it applies regardless of which backend ran.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import ToolContext, ToolOutcome, failed, ok, optional_int, require_str
from nanoclaude.tools.search import Match, search

DESCRIPTION = """Search file contents with a regular expression.

- `pattern` is a regular expression (ripgrep syntax), not a shell glob.
- `glob` limits which files are searched, for example `*.py`.
- `output_mode` is `content` (matching lines, the default), `files` (paths only) or
  `count` (matches per file).
- `-A`, `-B` and `-C` add lines of context; `-i` is case-insensitive.
- Paths ignored by .gitignore are skipped. Use `head_limit` to cap the output."""

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string"},
        "glob": {"type": "string"},
        "output_mode": {"type": "string", "enum": ["content", "files", "count"]},
        "-i": {"type": "boolean"},
        "head_limit": {"type": "integer", "minimum": 1},
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


class GrepTool:
    name = "Grep"
    read_only = True

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, DESCRIPTION, SCHEMA)

    def permission_request(
        self, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        base = arguments.get("path")
        paths = (ctx.resolve(base),) if isinstance(base, str) and base else ()
        # See glob.py's permission_request for why this is not require_str.
        pattern = arguments.get("pattern")
        subject = pattern if isinstance(pattern, str) else ""
        return PermissionRequest(self.name, subject, paths)

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        pattern = require_str(arguments, "pattern")
        base = arguments.get("path")
        root = ctx.resolve(base) if isinstance(base, str) and base else ctx.root
        mode = arguments.get("output_mode", "content")
        limit = optional_int(arguments, "head_limit", 200, minimum=1)

        try:
            re.compile(pattern)
        except re.error as exc:
            return failed(call_id, f"{pattern!r} is not a valid regular expression: {exc}")
        try:
            matches = search(
                pattern,
                root=root,
                glob=arguments.get("glob"),
                ignore_case=bool(arguments.get("-i", False)),
                limit=limit,
            )
        except RuntimeError as exc:
            return failed(call_id, f"search failed: {exc}")

        matches = self._drop_secret_files(ctx, matches)

        if not matches:
            return ok(call_id, f"No matches for {pattern}")

        if mode == "files":
            body = "\n".join(sorted({ctx.display(m.path) for m in matches}))
        elif mode == "count":
            counts = Counter(ctx.display(m.path) for m in matches)
            body = "\n".join(f"{path}: {n}" for path, n in counts.most_common())
        else:
            body = "\n".join(
                f"{ctx.display(m.path)}:{m.line_no}: {m.line.strip()}" for m in matches
            )
        redacted, _ = ctx.redactor.scrub(body)
        return ok(call_id, f"{len(matches)} match(es)\n{redacted}")

    @staticmethod
    def _drop_secret_files(ctx: ToolContext, matches: list[Match]) -> list[Match]:
        """Requirement (added 2026-10-01, task-14-brief.md): a credentials-shaped
        file discovered while walking the tree must not have its contents
        surfaced, on either search backend, unless the policy explicitly opts in.
        """
        if ctx.policy.allow_secrets:
            return matches
        return [m for m in matches if not ctx.redactor.is_secret_path(m.path, ctx.display(m.path))]
