import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "nanoclaude"

# The awaiting, the clock and identity generation all live in modules the loop
# must never reach for itself -- that is Task 4's whole bet (see loop.py's
# module docstring). Listed exactly as the task names them, not "every stdlib
# module that smells like I/O", so this stays a precise regression lock rather
# than a guess at what else might count.
BANNED_IN_AGENT = ("os", "subprocess", "time", "random", "uuid", "asyncio", "sqlite3", "httpx")

# The no-I/O rule above is loop.py's own design (Task 4's bet), not a
# constraint on the whole agent/ package. Spec 3 places Executor, Session,
# Router and Loop all in agent/, and Task 16's executor.py genuinely needs
# both asyncio (it runs tools concurrently) and time (it measures them) to do
# its job -- the package's only hard boundary is that agent/ never imports the
# CLI (test_agent_never_imports_the_cli, below). Enforced over this explicit
# list rather than "every file under agent/" so that adding a new, impure
# module to the package cannot silently widen what the no-I/O rule applies to;
# a later module that is itself meant to be pure joins this tuple in its own
# task. test_pure_modules_is_non_empty_and_every_listed_module_exists (below)
# pins that the list itself cannot quietly empty out from under the test that
# reads it.
PURE_MODULES = ("loop.py",)


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


def test_pure_modules_is_non_empty_and_every_listed_module_exists():
    """PURE_MODULES is the entire enforcement surface for the no-I/O rule
    below. If it were ever edited down to an empty tuple, the test below would
    iterate zero modules and pass having checked nothing -- a regression that
    looks exactly like success. Pinning both halves here (the tuple is
    non-empty, and every name on it is a real file under agent/) means that
    mistake fails loudly at this test instead of silently emptying the
    regression lock one test down.
    """
    assert PURE_MODULES, "PURE_MODULES must not be empty"
    missing = [name for name in PURE_MODULES if not (SRC / "agent" / name).is_file()]
    assert missing == []


def test_the_pure_modules_do_no_io_keep_no_clock_and_make_no_ids():
    """The modules named in PURE_MODULES compute what happens next; they never
    await, time or name anything.

    That split is what lets a whole multi-turn session run from a scripted model
    in microseconds -- no network, no disk, no clock -- which is what makes the
    adversarial cases (editing before reading, walking out of the sandbox,
    blowing the token budget) cheap enough to test exhaustively. A stray
    ``import time`` or ``import uuid`` in one of these specific modules would
    quietly end that bet, so the rule is checked here rather than left to only
    be documented in the module docstring.

    Only loop.py is held to this today (Task 4's bet, see its module
    docstring). agent/executor.py (Task 16) runs tools concurrently and
    measures them, so it needs both asyncio and time; see PURE_MODULES' own
    comment, above, for why that does not weaken this rule for loop.py itself.
    """
    offenders = [
        f"{module} imports {name}"
        for module in PURE_MODULES
        for name in imported_modules(SRC / "agent" / module)
        for banned in BANNED_IN_AGENT
        if name == banned or name.startswith(f"{banned}.")
    ]
    assert offenders == []


def test_gitwildmatch_has_a_single_home():
    """permissions/rules.py's GLOB_PATTERN_FACTORY keeps the deprecated pathspec
    factory name ``"gitwildmatch"`` in exactly one place on purpose -- see that
    constant's comment -- so that migrating to pathspec's new name (``"gitignore"``)
    is a one-line change instead of a search across every file that matches a
    glob. A second literal copy anywhere else would quietly reintroduce the
    multi-site problem the constant exists to avoid.
    """
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "gitwildmatch" in path.read_text(encoding="utf-8")
    )
    assert offenders == ["permissions/rules.py"]
