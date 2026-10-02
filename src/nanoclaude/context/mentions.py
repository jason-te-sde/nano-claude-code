"""`@path` in a user message inlines that file at the point it was mentioned.

The explicit half of the context strategy: no automatic retrieval, but a direct
way to say "this file". Anything outside the working directory, or that does
not exist, is left as literal text -- an unexpanded mention is confusing, but a
silently expanded `/etc/passwd` is worse.

A mention of a credentials-shaped path (``redactor.is_secret_path``) is left
unexpanded the same way, unless ``allow_secrets`` is set. Without this check,
an ``@.env`` mention would place the file's content in the request sent to the
model provider with only the generic content redactor in the way -- and that
redactor matches known shapes, not every line a credentials file can contain
(a database URL with an inline password, for one). The secret-path rule is
what already refuses this file to Read and skips it in Grep; a mention goes
through neither of those, so it needed this check of its own.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from nanoclaude.permissions.redact import Redactor
from nanoclaude.permissions.sandbox import is_within
from nanoclaude.tools.fs import FileSystemError, read_text

#: Not preceded by a word character, so an email address is not a mention.
_MENTION = re.compile(r"(?<![\w@])@([\w./~-]+)")
MAX_MENTION_BYTES = 100_000


def expand_mentions(
    text: str, *, root: str, redactor: Redactor, allow_secrets: bool = False
) -> tuple[str, tuple[str, ...]]:
    expanded: list[str] = []
    paths: list[str] = []
    position = 0
    for match in _MENTION.finditer(text):
        raw = match.group(1)
        candidate = os.path.realpath(raw if Path(raw).is_absolute() else Path(root) / raw)
        if not is_within(root, candidate) or not Path(candidate).is_file():
            continue
        relative = os.path.relpath(candidate, root)
        if redactor.is_secret_path(candidate, relative) and not allow_secrets:
            continue
        try:
            snapshot = read_text(candidate)
        except (OSError, FileSystemError):
            continue
        body, _ = redactor.scrub(snapshot.content[:MAX_MENTION_BYTES])
        expanded.append(text[position : match.end()])
        expanded.append(f'\n\n<file path="{relative}">\n{body}\n</file>\n\n')
        paths.append(candidate)
        position = match.end()
    expanded.append(text[position:])
    return "".join(expanded), tuple(paths)
