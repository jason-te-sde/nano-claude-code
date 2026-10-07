"""Fixtures for the CLI tests."""

import pytest


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
