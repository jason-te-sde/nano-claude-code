"""Tests for the files ``ncc`` keeps under its home: what it makes there is private."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.conversation.store import Store
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


def test_a_home_an_earlier_version_made_open_to_others_is_narrowed_and_the_config_is_left(
    tmp_path: Path,
    project: Path,
    serve: Callable[..., dict[str, ScriptedClient]],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What ncc keeps there (its store and its capability cache) becomes private when it is
    opened; the person's own ``config.toml`` keeps the mode the person gave it."""
    home = tmp_path / "old-home"
    state = home / ".nanoclaude"
    state.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(home))
    (state / "config.toml").write_text(NCC_CONFIG)
    (state / "capabilities.json").write_text("{}")
    store = Store(state / "sessions.db")
    store.open()
    store.close()
    state.chmod(0o755)
    for name in ("config.toml", "capabilities.json", "sessions.db"):
        (state / name).chmod(0o644)
    serve(m=[says("hi")])

    code, _, _ = run_ncc(capsys, "--root", str(project), "-p", "hello")

    assert code == EXIT_CODES["completed"]
    assert mode_of(state) == 0o700
    assert mode_of(state / "sessions.db") == 0o600
    assert mode_of(state / "capabilities.json") == 0o600
    assert mode_of(state / "config.toml") == 0o644
    for sidecar in state.glob("sessions.db-*"):
        assert mode_of(sidecar) == 0o600, sidecar.name
