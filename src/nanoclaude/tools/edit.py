# src/nanoclaude/tools/edit.py
"""Edit: exact string replacement, in a batch that is all or nothing.

Three decisions worth knowing about.

**Content, not line numbers.** Models miscount lines; they do not miscount
text. A wrong line number silently edits the wrong place, a wrong string is an
error the model can see and correct.

**Uniqueness is required.** If ``old_string`` matches twice we do not pick one.
The error says how many times it matched, which tells the model exactly what to
do: add context, or pass ``replace_all``.

**The batch is atomic.** A rename that touches four places must not be able to
apply three of them; the result compiles as neither the old code nor the new.

Quotes and line endings are the two places where what the model types and what
is in the file reliably disagree, and both are handled rather than reported.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import ToolArgumentError, ToolContext, ToolOutcome, failed, ok
from nanoclaude.tools.fs import FileSystemError, read_text, write_atomic

_CURLY = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
)  # left/right single and double curly quotes -> straight


class EditError(ValueError):
    """An edit could not be applied. The message is written for the model."""


@dataclass(frozen=True, slots=True)
class EditSpec:
    old: str
    new: str
    replace_all: bool = False


def detect_eol(content: str) -> str:
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _to_eol(text: str, eol: str) -> str:
    normalised = text.replace("\r\n", "\n")
    return normalised.replace("\n", eol) if eol != "\n" else normalised


def _fold(text: str) -> str:
    """Quote-insensitive view. The mapping is 1:1 so offsets still line up."""
    return text.translate(_CURLY)


def _locate(haystack: str, needle: str) -> tuple[str, int]:
    """Return the literal text present in ``haystack`` and how often it occurs."""
    if needle in haystack:
        return needle, haystack.count(needle)
    folded_hay, folded_needle = _fold(haystack), _fold(needle)
    index = folded_hay.find(folded_needle)
    if index == -1:
        return needle, 0
    return haystack[index : index + len(needle)], folded_hay.count(folded_needle)


def apply_edits(content: str, edits: Sequence[EditSpec]) -> str:
    """Apply every edit in order, or raise and change nothing."""
    if not edits:
        raise EditError("no edits were given")
    eol = detect_eol(content)
    result = content
    for position, edit in enumerate(edits, start=1):
        old = _to_eol(edit.old, eol)
        new = _to_eol(edit.new, eol)
        if not old:
            raise EditError(f"edit {position}: old_string must not be empty")
        literal, count = _locate(result, old)
        if count == 0:
            raise EditError(
                f"edit {position}: old_string was not found. Read the file again and "
                "copy the text exactly, without the line-number prefix."
            )
        if count > 1 and not edit.replace_all:
            raise EditError(
                f"edit {position}: old_string appears {count} times. Add surrounding "
                "context to make it unique, or set replace_all."
            )
        result = result.replace(literal, new, -1 if edit.replace_all else 1)
    return result


def unified_diff(before: str, after: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )


DESCRIPTION = """\
Make one or more exact string replacements in a single file. All edits apply, or none do.

- You must Read the file in this session first. If it changed since you read it, this
  tool fails and you must read it again.
- Each edit's `old_string` must appear exactly once in the file, unless `replace_all` is
  true. A non-unique `old_string` is an error, not a guess: add surrounding context.
- Edits apply in order, each to the result of the previous one.
- Never include the line-number prefix from Read output in `old_string` or `new_string`.
- Preserve the file's existing indentation and line endings.
- Prefer editing an existing file over writing a new one."""

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File to edit."},
        "edits": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["path", "edits"],
    "additionalProperties": False,
}


def _parse_edits(arguments: Mapping[str, Any]) -> list[EditSpec]:
    raw = arguments.get("edits")
    if not isinstance(raw, list) or not raw:
        raise ToolArgumentError("edits must be a list with at least one entry")
    specs: list[EditSpec] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ToolArgumentError(f"edit {index} must be an object")
        old, new = item.get("old_string"), item.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolArgumentError(f"edit {index}: old_string and new_string must be strings")
        specs.append(EditSpec(old, new, bool(item.get("replace_all", False))))
    return specs


def preview_edit(path: str, shown: str, arguments: Mapping[str, Any]) -> str:
    """The diff :meth:`EditTool.run` would write for ``arguments``, without writing it.

    What a confirmation shows before a person says yes: the file as it is now, with the
    call's edits applied by the same algorithm, as the unified diff the tool reports
    afterwards. ``path`` is the resolved file and ``shown`` is the name the diff carries.

    It raises what the edit itself would fail with, so that the front end can say the
    edit cannot be applied instead of showing nothing: ``ToolArgumentError`` for arguments
    that are not edits, ``EditError`` for an edit that does not apply or changes nothing,
    and ``OSError`` for a file that cannot be read. It does not know what the session has
    read, so it cannot say that ``run`` will refuse the file for not having been read.
    """
    specs = _parse_edits(arguments)
    snapshot = read_text(path)
    updated = apply_edits(snapshot.content, specs)
    if updated == snapshot.content:
        raise EditError("the edit produced no change")
    return unified_diff(snapshot.content, updated, shown)


class EditTool:
    name = "Edit"
    read_only = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, DESCRIPTION, SCHEMA)

    def permission_request(
        self, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        resolved = ctx.resolve(_require_path(arguments))
        return PermissionRequest(self.name, resolved, (resolved,), is_write=True)

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        resolved = ctx.resolve(_require_path(arguments))
        shown = ctx.display(resolved)
        try:
            specs = _parse_edits(arguments)
        except ToolArgumentError as exc:
            return failed(call_id, str(exc))

        seen = ctx.read_state.get(resolved)
        if seen is None:
            return failed(
                call_id,
                f"{shown} has not been read in this session. Read it, then edit it.",
            )
        try:
            snapshot = read_text(resolved)
        except FileNotFoundError:
            return failed(call_id, f"{shown} no longer exists")
        except FileSystemError as exc:
            return failed(call_id, str(exc))
        if snapshot.stamp.sha256 != seen.sha256:
            return failed(
                call_id,
                f"{shown} changed since it was read. Read it again, then re-apply "
                "your edit to the new content.",
            )

        try:
            updated = apply_edits(snapshot.content, specs)
        except EditError as exc:
            return failed(call_id, f"{shown}: {exc}")

        if updated == snapshot.content:
            return failed(call_id, f"{shown}: the edit produced no change")

        stamp = write_atomic(resolved, updated)
        diff = unified_diff(snapshot.content, updated, shown)
        return ok(call_id, f"Edited {shown}\n{diff}", ((resolved, stamp),))


def _require_path(arguments: Mapping[str, Any]) -> str:
    value = arguments.get("path")
    if not isinstance(value, str):
        raise ToolArgumentError(f"path must be a string, got {type(value).__name__}")
    return value
