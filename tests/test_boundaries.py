import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "nanoclaude"

# The awaiting, the clock and identity generation all live in modules the loop
# must never reach for itself -- that is Task 4's whole bet (see loop.py's
# module docstring). Listed exactly as the task names them, not "every stdlib
# module that smells like I/O", so this stays a precise regression lock rather
# than a guess at what else might count.
BANNED_IN_AGENT = ("os", "subprocess", "time", "random", "uuid", "asyncio", "sqlite3", "httpx")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_agent_never_imports_the_cli():
    """agent/ talks to the front end through the UI protocol, not the other way."""
    offenders = [
        f"{path.relative_to(SRC)} imports {name}"
        for path in (SRC / "agent").rglob("*.py")
        for name in imported_modules(path)
        if name.startswith("nanoclaude.cli")
    ]
    assert offenders == []


def test_agent_does_no_io_keeps_no_clock_and_makes_no_ids():
    """The loop computes what happens next; it never awaits, times or names anything.

    That split is what lets a whole multi-turn session run from a scripted model
    in microseconds -- no network, no disk, no clock -- which is what makes the
    adversarial cases (editing before reading, walking out of the sandbox,
    blowing the token budget) cheap enough to test exhaustively. A stray
    ``import time`` or ``import uuid`` would quietly end that bet, so the rule
    is checked here rather than left to only be documented in the module
    docstring.
    """
    offenders = [
        f"{path.relative_to(SRC)} imports {name}"
        for path in (SRC / "agent").rglob("*.py")
        for name in imported_modules(path)
        for banned in BANNED_IN_AGENT
        if name == banned or name.startswith(f"{banned}.")
    ]
    assert offenders == []
