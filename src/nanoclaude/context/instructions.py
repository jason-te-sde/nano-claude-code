"""Project instructions: NANO.md, layered from the repository's top level downwards.

CLAUDE.md is read when NANO.md is absent, because most repositories that
already have project instructions for an agent have them under that name and
asking people to duplicate the file is a bad first impression.

The size cap matters more than it looks. Instructions are in every request, so
a 400 KB file is not a large prompt, it is a session that cannot hold a
conversation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from nanoclaude.context.git import run_git

INSTRUCTION_FILENAMES = ("NANO.md", "CLAUDE.md")
MAX_INSTRUCTION_BYTES = 32_000


def _read_one(directory: Path) -> str:
    for name in INSTRUCTION_FILENAMES:
        candidate = directory / name
        if candidate.is_file():
            data = candidate.read_bytes()[: MAX_INSTRUCTION_BYTES + 1]
            text = data.decode("utf-8", errors="replace")
            if len(data) > MAX_INSTRUCTION_BYTES:
                text = (
                    text[:MAX_INSTRUCTION_BYTES]
                    + f"\n\n[truncated at {MAX_INSTRUCTION_BYTES} bytes]"
                )
            return f"<!-- {candidate} -->\n{text.strip()}"
    return ""


def _repository_top(root: Path) -> Path | None:
    """The top-level directory of the repository ``root`` is in, or None when it is in none.

    Asked of git through :func:`run_git`, which holds what keeps a repository somebody else
    prepared from running anything. Not being able to ask (git is not installed, it did not
    answer in time) is the same as not being in a repository: the answer is only ever more
    instructions, and a root's own are always read.
    """
    try:
        asked = run_git(str(root), "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError):
        return None
    top = asked.stdout.rstrip("\n")
    return Path(top).resolve() if asked.returncode == 0 and top else None


def _above_the_root(root: Path) -> list[Path]:
    """The directories from the repository's top level down to, not including, ``root``."""
    top = _repository_top(root)
    if top is None:
        return []
    return [parent for parent in reversed(root.parents) if parent == top or top in parent.parents]


def load_instructions(root: str, *, cwd: str, home: str | None) -> str:
    """Global, then the repository's top level down to the root, then down to ``cwd``.

    Later wins. A root outside a repository, or one git cannot be asked about, reads its own
    directory and below only. The configuration is not layered like this: it is read from
    the root alone.
    """
    parts: list[str] = []
    if home:
        parts.append(_read_one(Path(home) / ".nanoclaude"))

    root_path, cwd_path = Path(root).resolve(), Path(cwd).resolve()
    chain = [*_above_the_root(root_path), root_path]
    if cwd_path != root_path and root_path in cwd_path.parents:
        relative = cwd_path.relative_to(root_path)
        current = root_path
        for part in relative.parts:
            current = current / part
            chain.append(current)
    parts.extend(_read_one(directory) for directory in chain)
    return "\n\n".join(p for p in parts if p)
