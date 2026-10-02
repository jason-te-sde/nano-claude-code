"""Project instructions: NANO.md, layered from the repository root downwards.

CLAUDE.md is read when NANO.md is absent, because most repositories that
already have project instructions for an agent have them under that name and
asking people to duplicate the file is a bad first impression.

The size cap matters more than it looks. Instructions are in every request, so
a 400 KB file is not a large prompt, it is a session that cannot hold a
conversation.
"""

from __future__ import annotations

from pathlib import Path

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


def load_instructions(root: str, *, cwd: str, home: str | None) -> str:
    """Global, then root, then each directory down to ``cwd``. Later wins."""
    parts: list[str] = []
    if home:
        parts.append(_read_one(Path(home) / ".nanoclaude"))

    root_path, cwd_path = Path(root).resolve(), Path(cwd).resolve()
    chain = [root_path]
    if cwd_path != root_path and root_path in cwd_path.parents:
        relative = cwd_path.relative_to(root_path)
        current = root_path
        for part in relative.parts:
            current = current / part
            chain.append(current)
    parts.extend(_read_one(directory) for directory in chain)
    return "\n\n".join(p for p in parts if p)
