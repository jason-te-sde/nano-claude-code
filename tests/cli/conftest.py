"""Fixtures for the CLI tests."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.agent.router import Router
from nanoclaude.agent.session import Session
from nanoclaude.cli import main as ncc_main
from nanoclaude.conversation.store import Store
from nanoclaude.providers.base import ModelError, ModelReply
from nanoclaude.testing.session import ScriptedClient


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment block asks git for the branch on every turn: two forks.

    Which branch the repository is on has no bearing on anything tested here, and a
    temp directory that happened to sit inside another repository would change the
    system prompt from machine to machine. assemble() has its own tests.
    """
    monkeypatch.setattr("nanoclaude.context.assemble.git_state", lambda _root: "")


@pytest.fixture(autouse=True)
def _leave_the_real_terminal_input_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confirmation discards what was typed before it, and the default does that to stdin.

    Under ``pytest -s`` stdin is the terminal the person is typing into, and a test run
    must not eat their keystrokes. The tests of the function itself call it directly, on a
    terminal of their own.
    """
    monkeypatch.setattr("nanoclaude.cli.render.discard_pending_input", lambda: None)


@pytest.fixture(autouse=True)
def _leave_the_real_terminal_echo_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live display turns the echo of the terminal off for as long as it is drawn.

    Under ``pytest -s`` standard input is the terminal the person is typing into, and a test
    run must not switch off the echo of what they type. The tests of the function itself call
    it directly, on a terminal of their own.
    """
    monkeypatch.setattr("nanoclaude.cli.render.silence_echo", lambda: lambda: None)


@pytest.fixture(autouse=True)
def _a_terminal_that_can_move_the_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether Rich draws a live display on a terminal depends on ``TERM``.

    ``dumb`` (an Emacs shell, some CI images) makes the consoles these tests call terminals
    into terminals that cannot move the cursor, and nothing live is drawn on them. The test
    of a dumb terminal says so itself.
    """
    monkeypatch.setenv("TERM", "xterm-256color")


# ---------------------------------------------------------------------------
# The ``ncc`` command: a home with a configuration, a project, and scripted models
# ---------------------------------------------------------------------------

#: Three models that need no network. ``m`` is the main role's, ``other`` is a second one to
#: route a role to, and ``textual`` is one that answers in the text protocol.
NCC_CONFIG = """\
[models.m]
adapter = "anthropic"
model = "claude-sonnet-5"

[models.other]
adapter = "anthropic"
model = "claude-haiku-4-5"

[models.textual]
adapter = "anthropic"
model = "claude-sonnet-5"
native_tools = false
"""


@pytest.fixture
def ncc_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home directory with a configuration in it, which is the one ``ncc`` reads.

    ``HOME`` and ``NANOCLAUDE_HOME`` both point at it, so nothing here reads the home of
    whoever runs the tests, and ``~`` in a flag means this directory.
    """
    home = tmp_path / "home"
    (home / ".nanoclaude").mkdir(parents=True)
    (home / ".nanoclaude" / "config.toml").write_text(NCC_CONFIG)
    home = home.resolve()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(home))
    monkeypatch.delenv("NO_COLOR", raising=False)
    return home


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """The working directory ``--root`` names: a directory that exists, resolved."""
    root = tmp_path / "project"
    root.mkdir()
    (root / ".git").mkdir()
    return root.resolve()


Script = Sequence[ModelReply | ModelError]


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict[str, ScriptedClient]]:
    """Answer for the models of the configuration from scripts, one per alias.

    ``serve(m=[says("hi")])`` makes the model defined as ``m`` reply "hi". The router is
    otherwise the real one, built from the configuration that was loaded, so the roles, the
    capability table and the prices are the shipped ones. An alias with no script has no
    client, and the router would build a real one for it: a test names every alias it uses.
    """

    def install(**scripts: Script) -> dict[str, ScriptedClient]:
        clients = {alias: ScriptedClient(script) for alias, script in scripts.items()}
        monkeypatch.setattr(
            ncc_main, "Router", lambda config, cache: Router(config, cache, dict(clients))
        )
        return clients

    return install


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> list[Session]:
    """Every session ``ncc`` builds in a test, for looking at what the flags made of it."""
    built: list[Session] = []

    def make(**fields: Any) -> Session:
        session = Session(**fields)
        built.append(session)
        return session

    monkeypatch.setattr(ncc_main, "Session", make)
    return built


@pytest.fixture
def stores(monkeypatch: pytest.MonkeyPatch) -> list[Store]:
    """Every store ``ncc`` opens in a test, to check that it was closed on the way out."""
    opened: list[Store] = []

    class Recording(Store):
        def __init__(self, path: Path | str) -> None:
            super().__init__(path)
            opened.append(self)

    monkeypatch.setattr(ncc_main, "Store", Recording)
    return opened


@pytest.fixture(autouse=True)
def _somebody_at_the_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """ncc starts the REPL only where stdin is a terminal, and under pytest it is not.

    The tests that start the REPL are meant to; the one that is about there being nobody to
    type says so itself.
    """
    monkeypatch.setattr(ncc_main, "_stdin_is_terminal", lambda: True)
