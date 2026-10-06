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
