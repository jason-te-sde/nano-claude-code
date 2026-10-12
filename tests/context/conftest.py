"""Fixtures for the context tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _git_looks_no_higher_than_the_temporary_directories(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git is not let out of the directory every test's temporary directory is made in.

    The helpers ask git which repository a directory is in, and git answers for the whole
    machine unless it is told where to stop: ``tmp_repo``'s empty ``.git`` is not taken for a
    repository, so git goes on up, and a temporary directory inside a work tree gets that work
    tree's top, and its ``NANO.md`` or ``CLAUDE.md`` read into a test that built everything it
    reads. A ceiling at the base temporary directory is above everything a test makes and below
    everything around it; a repository a test makes in its own directory is still found.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path_factory.getbasetemp()))
