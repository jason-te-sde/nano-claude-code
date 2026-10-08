"""Tests for the files ``ncc`` keeps under its home: what it makes there is private."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.testing.scripted import says
from nanoclaude.testing.session import ScriptedClient
from tests.cli.conftest import NCC_CONFIG
from tests.cli.helpers import run_ncc
from tests.conftest import mode_of


def test_a_first_run_makes_ncc_s_home_and_its_session_store_private(
    tmp_path: Path,
    project: Path,
    serve: Callable[..., dict[str, ScriptedClient]],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no ``ncc init`` before it: the configuration here is the project's, and the
    home has no ``.nanoclaude`` until the run makes one."""
    home = tmp_path / "fresh-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(home))
    (project / ".nanoclaude").mkdir()
    (project / ".nanoclaude" / "config.toml").write_text(NCC_CONFIG)
    serve(m=[says("hi")])

    code, _, _ = run_ncc(capsys, "--root", str(project), "-p", "hello")

    assert code == EXIT_CODES["completed"]
    state: Path = home / ".nanoclaude"
    assert (mode_of(state), mode_of(state / "sessions.db")) == (0o700, 0o600)
