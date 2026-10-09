# src/nanoclaude/tools/grep.py
"""Grep: search file contents.

Three output modes because the right amount of detail differs by an order of
magnitude between "where is this defined" and "how widely is this used", and a
model that has to read 400 matching lines to answer the second question has
spent its context on the wrong thing.

Both search backends walk the tree themselves rather than consulting a list of
paths the caller already cleared, so a broad search can surface a file the
caller never named -- an un-ignored ``.env`` sitting next to code, say. That
file is covered by the policy's refusal to read it, which the executor enforces
before a call that *names* it, but a path discovered mid-walk never goes through
that check. Matches are filtered against the same verdict here, after the
backend returns, so it applies regardless of which backend ran: a credentials
path, a path outside the sandbox, and any file a ``Read(...)`` deny rule covers.
A path is judged by its own spelling and by what it resolves to, so a link
cannot carry a file past the verdict that its target would not get.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from nanoclaude.permissions.policy import PermissionRequest, read_refusal
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import ToolContext, ToolOutcome, failed, ok, optional_int, require_str
from nanoclaude.tools.keyblocks import PRIVATE_KEY_MARKER, masked_line_numbers
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
        "-A": {"type": "integer", "minimum": 0},
        "-B": {"type": "integer", "minimum": 0},
        "-C": {"type": "integer", "minimum": 0},
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
        return PermissionRequest(self.name, require_str(arguments, "pattern"), paths)

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        pattern = require_str(arguments, "pattern")
        base = arguments.get("path")
        root = ctx.resolve(base) if isinstance(base, str) and base else ctx.root
        mode = arguments.get("output_mode", "content")
        # head_limit counts matches, not lines: it caps how many matches are
        # found (search.py's _apply_match_limit), and a kept match's own
        # context is never separately capped or cut off partway through.
        limit = optional_int(arguments, "head_limit", 200, minimum=1)
        before, after = self._resolve_context(arguments, mode)

        try:
            re.compile(pattern)
        except re.error as exc:
            return failed(call_id, f"{pattern!r} is not a valid regular expression: {exc}")
        readable = self._readability(ctx)
        try:
            # The backends are told which paths to leave out, so that they leave them out
            # before the limit is applied: a refused file with a great many matches must not
            # use up a limit that the files that are shown would have filled.
            matches = search(
                pattern,
                root=root,
                glob=arguments.get("glob"),
                ignore_case=bool(arguments.get("-i", False)),
                limit=limit,
                before=before,
                after=after,
                keep=readable,
            )
        except RuntimeError as exc:
            return failed(call_id, f"search failed: {exc}")

        # And again here, on whatever a backend reported anyway: the verdict is not left to
        # the backend having honoured the question.
        matches = [m for m in matches if readable(m.path)]

        if not matches:
            return ok(call_id, f"No matches for {pattern}")

        if mode == "content":
            matches = self._mask_private_keys(ctx, matches)

        match_count = sum(1 for m in matches if not m.is_context)
        if mode == "files":
            body = "\n".join(sorted({ctx.display(m.path) for m in matches}))
        elif mode == "count":
            # before/after are forced to 0 above for this mode, so every
            # entry here is a real match; nothing to exclude.
            counts = Counter(ctx.display(m.path) for m in matches)
            body = "\n".join(f"{path}: {n}" for path, n in counts.most_common())
        else:
            body = self._format_content(ctx, matches, merging=bool(before or after))
        redacted, _ = ctx.redactor.scrub(body)
        return ok(call_id, f"{match_count} match(es)\n{redacted}")

    @staticmethod
    def _resolve_context(arguments: Mapping[str, Any], mode: str) -> tuple[int, int]:
        """-C sets both sides; an explicit -A or -B overrides that side (its
        presence in `arguments`, not its truthiness, is what "explicit" means,
        so an explicit `-B: 0` does override a nonzero -C). files and count
        modes ignore context entirely -- -A/-B/-C are still validated (a
        malformed value is still an error in those modes), just not acted on.
        """
        around = optional_int(arguments, "-C", 0, minimum=0)
        before = optional_int(arguments, "-B", around, minimum=0)
        after = optional_int(arguments, "-A", around, minimum=0)
        if mode != "content":
            return 0, 0
        return before, after

    @staticmethod
    def _format_content(ctx: ToolContext, matches: list[Match], *, merging: bool) -> str:
        """`path:N:line` for a match, `path-N-line` for context -- ripgrep's own
        convention for ``--no-heading --line-number``. A `--` separator marks a
        break between groups, where a group is a run of contiguous line numbers
        in one file; crossing into a different file always counts as a break.
        Only applied when context was actually requested (`merging`): with no
        context, every match is its own result and no separator belongs
        between two of them, however far apart they are in the file.
        """
        lines: list[str] = []
        previous: tuple[str, int] | None = None
        for m in matches:
            contiguous = (
                previous is not None and previous[0] == m.path and m.line_no == previous[1] + 1
            )
            if merging and previous is not None and not contiguous:
                lines.append("--")
            display = ctx.display(m.path)
            sep = "-" if m.is_context else ":"
            lines.append(f"{display}{sep}{m.line_no}{sep}{m.line.strip()}")
            previous = (m.path, m.line_no)
        return "\n".join(lines)

    @staticmethod
    def _mask_private_keys(ctx: ToolContext, matches: list[Match]) -> list[Match]:
        """Replace the text of every shown line that is inside a private key block.

        Matches and context lines alike. Whether a line is inside a block is a fact about the
        whole file, which no scrub of the lines shown can know: a search for a piece of a key's
        body shows a line of base64 that is not recognisable as anything. It is worked out only
        for the files that have lines to show, after the limit has been applied, reading each
        as a stream as far as the last line shown from it.
        """
        last: dict[str, int] = {}
        for match in matches:
            last[match.path] = max(last.get(match.path, 0), match.line_no)
        masked = {
            path: masked_line_numbers(ctx.redactor, path, upto=upto) for path, upto in last.items()
        }
        return [
            replace(match, line=PRIVATE_KEY_MARKER)
            if match.line_no in masked[match.path]
            else match
            for match in matches
        ]

    @staticmethod
    def _readability(ctx: ToolContext) -> Callable[[str], bool]:
        """Whether the policy would let Read show the model this file; a path is asked once.

        Requirement (added 2026-10-01, task-14-brief.md): a credentials-shaped file
        discovered while walking the tree must not have its contents surfaced, on
        either search backend, unless the policy explicitly opts in. It has since been
        widened to the whole of Read's verdict (see ``read_refusal``), since a deny
        rule for Read is no less a statement about what the model may see. Applies to a
        context line exactly as it does to a match: both carry the same path, and the
        verdict is keyed on the path alone.
        """
        verdicts: dict[str, bool] = {}

        def readable(path: str) -> bool:
            if path not in verdicts:
                # Both spellings, so that neither a link named like a harmless file to a
                # credentials file, nor one named like a credentials file to a harmless
                # one, gets past.
                verdicts[path] = read_refusal(ctx.policy, path, os.path.realpath(path)) is None
            return verdicts[path]

        return readable
