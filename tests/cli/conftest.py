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
