"""A shallow, gitignore-aware picture of the repository.

Three limits, all of them load-bearing: depth, because nobody needs to see the
fifth level of a node_modules-shaped tree; an entry cap, because a monorepo
would otherwise eat the whole window; and a visited-inode set, because symlink
loops exist in real repositories and an unguarded walk never returns.

The map is in the system prompt of every request, so it is held to what the other ways
a file reaches the model are held to, as far as a name can: an entry the policy would not
let Read show (a credentials file, anything a ``Read(...)`` deny rule covers) is not listed,
and is not counted against the entry cap. The names of a project's credentials files are
themselves a map of where its secrets are. An entry is judged by its own spelling and, when
it is a link that leads somewhere in the project, by what it leads to; a link that leads out
of the project is listed and not entered, as it always was.
"""

from __future__ import annotations

import os
from pathlib import Path

import pathspec

from nanoclaude.permissions.policy import Policy, read_refusal
from nanoclaude.permissions.rules import GLOB_PATTERN_FACTORY
from nanoclaude.permissions.sandbox import is_within

DEFAULT_DEPTH = 3
DEFAULT_MAX_ENTRIES = 200


def project_map(
    root: str,
    *,
    policy: Policy,
    depth: int = DEFAULT_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> str:
    ignore_lines = ["/.git/"]
    gitignore = Path(root) / ".gitignore"
    if gitignore.exists():
        ignore_lines.extend(gitignore.read_text(errors="replace").splitlines())
    ignore = pathspec.PathSpec.from_lines(GLOB_PATTERN_FACTORY, ignore_lines)

    lines: list[str] = []
    seen: set[tuple[int, int]] = set()
    truncated = False

    def walk(directory: Path, level: int) -> None:
        nonlocal truncated
        if level > depth or truncated:
            return
        try:
            key = (directory.stat().st_dev, directory.stat().st_ino)
        except OSError:
            return
        if key in seen:
            return  # symlink loop
        seen.add(key)
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError:
            return
        for entry in entries:
            relative = os.path.relpath(entry, root)
            if ignore.match_file(relative + ("/" if entry.is_dir() else "")):
                continue
            if refused(entry):
                continue
            if len(lines) >= max_entries:
                truncated = True
                return
            indent = "  " * (level - 1)
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                # A symlink pointing outside the project is listed but not entered:
                # what lies beyond the project root is not the model's to see.
                if not entry.is_symlink() or is_within(real_root, os.path.realpath(entry)):
                    walk(entry, level + 1)
            else:
                lines.append(f"{indent}{entry.name}")

    real_root = os.path.realpath(root)

    def refused(entry: Path) -> bool:
        # Spelled inside the real root, which is how the policy's sandbox knows it: a root
        # reached through a link would otherwise have every entry outside the sandbox.
        spelled = os.path.join(real_root, os.path.relpath(entry, root))  # noqa: PTH118
        spellings = [spelled]
        real = os.path.realpath(entry)
        if real != spelled and is_within(real_root, real):
            spellings.append(real)
        return read_refusal(policy, *spellings) is not None

    walk(Path(root), 1)
    body = "\n".join(lines)
    if truncated:
        body += f"\n... truncated at {max_entries} entries; use Glob to look further"
    return body
