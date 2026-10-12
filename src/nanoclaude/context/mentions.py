"""`@path` in a user message inlines that file at the point it was mentioned.

The explicit half of the context strategy: no automatic retrieval, but a direct
way to say "this file". Anything outside the working directory, or that does
not exist, is left as literal text -- an unexpanded mention is confusing, but a
silently expanded `/etc/passwd` is worse.

A mention of a file the policy would not let Read show the model is not inlined
either, and says so in one line where the file would have gone: a credentials-shaped
path (unless ``allow_secrets`` is set), or any file a ``Read(...)`` deny rule covers.
Without this check, an ``@.env`` mention would place the file's content in the request
sent to the model provider with only the generic content redactor in the way -- and that
redactor matches known shapes, not every line a credentials file can contain (a database
URL with an inline password, for one). The refusal is what already stops Read and Grep
from showing the file; a mention goes through neither of those, so it asks the same
question of its own (``read_refusal``). The line is there because the person wrote
``@path`` believing the model would see the file, and the model should not be left to
guess at what it was not shown.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from nanoclaude.permissions.policy import Policy, read_refusal
from nanoclaude.permissions.redact import Redactor
from nanoclaude.permissions.sandbox import is_within
from nanoclaude.tools.base import sanitize
from nanoclaude.tools.fs import FileSystemError, read_text

#: Not preceded by a word character, so an email address is not a mention, and
#: not ending in a dot, so "see @main.py." at the end of a sentence still works.
_MENTION = re.compile(r"(?<![\w@])@([\w./~-]*[\w/~-])")
MAX_MENTION_BYTES = 100_000


def expand_mentions(
    text: str, *, root: str, redactor: Redactor, policy: Policy
) -> tuple[str, tuple[str, ...]]:
    """``text`` with each ``@path`` it names followed by that file, and the files inlined.

    ``policy`` is required, and there is no default that would let a caller leave it out: a
    mention that does not ask it is a way past every refusal the policy makes.
    """
    expanded: list[str] = []
    paths: list[str] = []
    refused: set[str] = set()
    position = 0
    # Resolved like the candidate is: a root reached through a symlink (on macOS,
    # anything under /tmp or /var) would otherwise never contain any resolved
    # candidate, and every mention would be silently dropped.
    if not Path(root).is_absolute():
        # Resolving a relative root would anchor it to whatever directory the
        # process happens to be in -- the ambient state Sandbox refuses too.
        raise ValueError(f"root must be an absolute path, got {root!r}")
    real_root = os.path.realpath(root)
    for match in _MENTION.finditer(text):
        raw = match.group(1)
        candidate = os.path.realpath(raw if Path(raw).is_absolute() else Path(real_root) / raw)
        if not is_within(real_root, candidate) or not Path(candidate).is_file():
            continue
        if candidate in paths or candidate in refused:
            continue  # inline each file once, however often it is mentioned
        relative = os.path.relpath(candidate, real_root)
        # The name the mention gave the file as well as what it leads to, when that
        # is a name inside the root (an absolute spelling through an alias of the root is
        # not, and is judged by what it resolves to).
        spelled = os.path.normpath(Path(real_root) / raw)
        spellings = (spelled, candidate) if is_within(real_root, spelled) else (candidate,)
        refusal = read_refusal(policy, *spellings)
        if refusal is not None:
            refused.add(candidate)
            reason = " ".join(sanitize(refusal.reason).split())
            expanded.append(text[position : match.end()])
            expanded.append(
                f"\n\n[@{raw} was not inlined \u2014 refused ({refusal.rule}): {reason}]\n\n"
            )
            position = match.end()
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
