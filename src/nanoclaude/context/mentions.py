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

#: Not preceded by a word character, so an email address is not a mention, and
#: not ending in a dot, so "see @main.py." at the end of a sentence still works.
_MENTION = re.compile(r"(?<![\w@])@([\w./~-]*[\w/~-])")
MAX_MENTION_BYTES = 100_000


def expand_mentions(
    text: str, *, root: str, redactor: Redactor, allow_secrets: bool = False
) -> tuple[str, tuple[str, ...]]:
    expanded: list[str] = []
    paths: list[str] = []
    position = 0
    # Resolved like the candidate is: a root reached through a symlink (on macOS,
    # anything under /tmp or /var) would otherwise never contain any resolved
    # candidate, and every mention would be silently dropped.
    real_root = os.path.realpath(root)
    for match in _MENTION.finditer(text):
        raw = match.group(1)
        candidate = os.path.realpath(raw if Path(raw).is_absolute() else Path(real_root) / raw)
        if not is_within(real_root, candidate) or not Path(candidate).is_file():
            continue
        if candidate in paths:
            continue  # inline each file once, however often it is mentioned
        relative = os.path.relpath(candidate, real_root)
        if redactor.is_secret_path(candidate, relative) and not allow_secrets:
            continue
        try:
            snapshot = read_text(candidate)
        except (OSError, FileSystemError):
            continue
        # Scrub the whole file before cutting it, so a credential that straddles the
        # cut cannot survive as an unrecognised half. The cap is in bytes, as named.
        scrubbed, _ = redactor.scrub(snapshot.content)
        encoded = scrubbed.encode("utf-8")
        body = encoded[:MAX_MENTION_BYTES].decode("utf-8", errors="ignore")
        if len(encoded) > MAX_MENTION_BYTES:
            body += f"\n[truncated at {MAX_MENTION_BYTES} bytes]"
        expanded.append(text[position : match.end()])
        expanded.append(f'\n\n<file path="{relative}">\n{body}\n</file>\n\n')
        paths.append(candidate)
        position = match.end()
    expanded.append(text[position:])
    return "".join(expanded), tuple(paths)
