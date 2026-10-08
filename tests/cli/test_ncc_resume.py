"""Tests for ``ncc --continue`` and ``ncc --resume``, and for closing what ``ncc`` opened.

A resumed session is the stored one, taken up where it stopped: its conversation is what the
model is shown, and what it spends is added to its row. These tests run a first prompt and
a second one as two commands, the way a person would, and look at what the second one's model
was shown and at what the store holds afterwards.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.agent.router import Router
from nanoclaude.agent.session import Session
from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.config.schema import Config
from nanoclaude.conversation.store import Store
from nanoclaude.conversation.transcript import assistant_text, user_text
from nanoclaude.providers.base import ModelError, ModelRequest
from nanoclaude.providers.capabilities import CapabilityCache
from nanoclaude.testing.scripted import says
from nanoclaude.testing.session import ScriptedClient
from tests.cli.helpers import INTERRUPTED, run_ncc


def start(capsys: pytest.CaptureFixture[str], root: Path, prompt: str = "first question") -> str:
    """Run a first prompt in ``root`` and return the id of the session it started."""
    code, out, _ = run_ncc(capsys, "--root", str(root), "-p", prompt, "--output-format", "json")
    assert code == EXIT_CODES["completed"]
    session_id: str = json.loads(out)["session_id"]
    return session_id


def stored(home: Path) -> Store:
    store = Store(home / ".nanoclaude" / "sessions.db")
    store.open()
    return store


def shown(request: ModelRequest) -> list[tuple[str, str]]:
    return [(m.role, m.text()) for m in request.transcript.messages]


def assert_closed(store: Store) -> None:
    with pytest.raises(RuntimeError, match="not open"):
        store.db  # noqa: B018


# --------------------------------------------------------------------------
# Taking a session up again
# --------------------------------------------------------------------------


def test_resume_continues_the_stored_conversation_instead_of_starting_another(
    ncc_home, project, serve, capsys
):
    clients = serve(
        m=[
            says("one", input_tokens=10, output_tokens=5),
            says("two", input_tokens=20, output_tokens=7),
        ]
    )
    first = start(capsys, project)
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--resume", first, "-p", "second question"
    )
    assert (code, out, err) == (EXIT_CODES["completed"], "two\n", "")
    # The model was shown the whole conversation so far, and the new prompt at the end.
    assert shown(clients["m"].requests[1]) == [
        ("user", "first question"),
        ("assistant", "one"),
        ("user", "second question"),
    ]
    store = stored(ncc_home)
    # The same row, holding both prompts. run() would have started a new conversation in it
    # and sent the old messages to the archive.
    assert [m.text() for m in store.load_transcript(first).messages] == [
        "first question",
        "one",
        "second question",
        "two",
    ]
    archived = store.db.execute(
        "SELECT COUNT(*) FROM messages_archive WHERE session_id = ?", (first,)
    ).fetchone()[0]
    assert archived == 0
    row = store.session_row(first)
    assert row is not None
    assert (row.total_input_tokens, row.total_output_tokens) == (30, 12)  # both runs, one row
    assert len(store.recent_sessions(10)) == 1


def test_a_resumed_runs_json_names_the_session_and_counts_only_this_run(
    ncc_home, project, serve, capsys
):
    serve(
        m=[
            says("one", input_tokens=10, output_tokens=5),
            says("two", input_tokens=20, output_tokens=7),
        ]
    )
    first = start(capsys, project)
    _, out, _ = run_ncc(
        capsys, "--root", str(project), "-r", first, "-p", "again", "--output-format", "json"
    )
    payload = json.loads(out)
    assert payload["session_id"] == first
    assert payload["usage"]["input_tokens"] == 20 and payload["usage"]["output_tokens"] == 7


def test_continue_takes_up_the_latest_session_started_here(ncc_home, project, serve, capsys):
    clients = serve(m=[says("one"), says("two")])
    first = start(capsys, project)
    _, out, _ = run_ncc(
        capsys, "--root", str(project), "--continue", "-p", "again", "--output-format", "json"
    )
    assert json.loads(out)["session_id"] == first
    assert shown(clients["m"].requests[1])[0] == ("user", "first question")


def test_continue_means_the_latest_session_in_this_directory_and_not_in_the_store(
    ncc_home, tmp_path, serve, capsys
):
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    clients = serve(m=[says("a"), says("b"), says("c"), says("d")])
    in_here = start(capsys, here, "a question about here")
    in_there = start(capsys, there, "a question about there")  # the latest in the whole store
    assert stored(ncc_home).latest_session_id() == in_there

    _, out, _ = run_ncc(capsys, "--root", str(here), "-c", "-p", "more", "--output-format", "json")
    assert json.loads(out)["session_id"] == in_here
    assert shown(clients["m"].requests[2])[0] == ("user", "a question about here")

    _, out, _ = run_ncc(capsys, "--root", str(there), "-c", "-p", "more", "--output-format", "json")
    assert json.loads(out)["session_id"] == in_there
    assert shown(clients["m"].requests[3])[0] == ("user", "a question about there")


def test_continue_with_nothing_started_here_is_a_usage_error_and_not_a_new_session(
    ncc_home, tmp_path, serve, capsys, stores
):
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    clients = serve(m=[says("a"), says("never")])
    start(capsys, there)
    stores.clear()
    code, out, err = run_ncc(capsys, "--root", str(here), "-c", "-p", "more")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        f"error: no earlier session was started in {here} — leave out --continue to start one\n"
    )
    assert len(clients["m"].requests) == 1  # only the first run asked anything
    assert len(stored(ncc_home).recent_sessions(10)) == 1
    for store in stores:
        assert_closed(store)


def test_a_session_that_is_not_in_the_store_is_a_usage_error_and_the_store_is_closed(
    ncc_home, project, serve, capsys, stores
):
    clients = serve(m=[says("never")])
    code, out, err = run_ncc(capsys, "--root", str(project), "--resume", "nope", "-p", "hi")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        'error: no stored session "nope" — check the id, or leave out --resume to start '
        "a new session\n"
    )
    assert clients["m"].requests == []
    assert len(stores) == 1
    assert_closed(stores[0])  # the Session never existed to close it


def test_a_session_from_another_directory_is_refused_naming_both(
    ncc_home, tmp_path, serve, capsys, stores
):
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    clients = serve(m=[says("a"), says("never")])
    in_there = start(capsys, there)
    stores.clear()
    code, out, err = run_ncc(capsys, "--root", str(here), "--resume", in_there, "-p", "hi")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        f"error: session {in_there} was started in {there}, not in {here} "
        f"— run ncc from {there} (or pass --root {there}), or leave out --resume\n"
    )
    assert len(clients["m"].requests) == 1
    # Nothing was added to the session that was refused.
    assert len(stored(ncc_home).load_transcript(in_there).messages) == 2
    for store in stores:
        assert_closed(store)


def test_the_directory_of_a_session_is_compared_as_the_place_it_is_not_as_a_spelling(
    ncc_home, tmp_path, serve, capsys
):
    # A row written by another front end can hold the spelling that went through a symlink.
    # It is still the directory the session was started in.
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    serve(m=[says("b")])
    store = stored(ncc_home)
    store.create_session("abc123abc123", cwd=str(link), roles={})
    store.close()
    code, out, _ = run_ncc(capsys, "--root", str(real), "-r", "abc123abc123", "-p", "again")
    assert (code, out) == (EXIT_CODES["completed"], "b\n")


def test_continue_and_resume_together_are_a_usage_error(ncc_home, project, capsys):
    code, out, err = run_ncc(capsys, "--root", str(project), "-c", "-r", "abc", "-p", "hi")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == "error: --resume and --continue cannot be combined \u2014 pass only one\n"
    assert "usage:" not in err


# --------------------------------------------------------------------------
# What ncc opened is closed again, whichever way the run ended
# --------------------------------------------------------------------------


def test_the_store_is_closed_after_a_run(ncc_home, project, serve, capsys, stores):
    serve(m=[says("ok")])
    run_ncc(capsys, "--root", str(project), "-p", "hi")
    assert len(stores) == 1
    assert_closed(stores[0])


def test_the_store_is_closed_after_a_run_that_failed(ncc_home, project, capsys, stores, serve):
    serve(m=[ModelError("down", retryable=False)])
    code, _, _ = run_ncc(capsys, "--root", str(project), "-p", "hi")
    assert code == EXIT_CODES["provider"]
    assert_closed(stores[0])


def test_the_store_is_closed_when_the_models_are_not_what_the_configuration_says(
    ncc_home, project, capsys, stores
):
    # The router refuses the configuration after the store has been opened.
    code, _, _ = run_ncc(capsys, "--root", str(project), "--model", "ghost", "-p", "hi")
    assert code == EXIT_CODES["config"]
    assert len(stores) == 1
    assert_closed(stores[0])


class Interfering(ScriptedClient):
    """A model that answers while another process writes to the session being run."""

    def __init__(self, database: Path, session_of: list[str]) -> None:
        super().__init__([])
        self._database = database
        self._session_of = session_of

    async def complete(self, request, *, on_text=None):  # noqa: ARG002
        (session_id,) = self._session_of
        other = sqlite3.connect(self._database)
        other.execute(
            "INSERT INTO messages (session_id, seq, role, blocks_json, created_at) "
            'VALUES (?, 1, \'assistant\', \'[{"type": "text", "text": "someone else"}]\', 0)',
            (session_id,),
        )
        other.commit()
        other.close()
        return says("my answer")


def test_a_session_changed_by_another_process_is_reported_and_the_store_is_closed(
    ncc_home, project, capsys, monkeypatch, stores
):
    holder: list[str] = []
    client = Interfering(ncc_home / ".nanoclaude" / "sessions.db", holder)

    def route(config: Config, cache: CapabilityCache) -> Router:
        return Router(config, cache, {"m": client})

    monkeypatch.setattr(ncc_main, "Router", route)

    def remember(**fields: Any) -> Session:
        holder.append(fields["session_id"])
        return Session(**fields)

    monkeypatch.setattr(ncc_main, "Session", remember)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hi")
    assert code == EXIT_CODES["stopped"]
    assert out == ""
    assert err == (
        f"error: session {holder[0]} was changed by another process "
        "— start a new session, or resume it again\n"
    )
    assert_closed(stores[0])


def test_ctrl_c_while_the_session_is_being_built_still_closes_the_store(
    ncc_home, project, capsys, monkeypatch, stores
):
    def interrupted(_config: Config, _cache: CapabilityCache) -> Router:
        raise KeyboardInterrupt

    monkeypatch.setattr(ncc_main, "Router", interrupted)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hi")
    assert code == EXIT_CODES["interrupted"]
    assert out == "" and err == INTERRUPTED
    assert_closed(stores[0])


def test_a_session_nothing_was_said_in_is_not_the_one_continue_takes_up(
    ncc_home, project, serve, capsys
):
    # A REPL that was left at once, after the conversation that matters.
    serve(m=[says("one"), says("two")])
    first = start(capsys, project)
    store = stored(ncc_home)
    store.create_session("abandoned00", cwd=str(project), roles={})
    store.close()
    _, out, _ = run_ncc(
        capsys, "--root", str(project), "-c", "-p", "again", "--output-format", "json"
    )
    assert json.loads(out)["session_id"] == first


def test_continue_finds_a_session_that_was_stored_under_a_symlink_spelling(
    ncc_home, tmp_path, serve, capsys
):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    clients = serve(m=[says("again")])
    store = stored(ncc_home)
    store.create_session("viaalink000", cwd=str(link), roles={})
    store.append_message("viaalink000", 0, user_text("an earlier question"))
    store.append_message("viaalink000", 1, assistant_text("an earlier answer"))
    store.close()
    _, out, _ = run_ncc(capsys, "--root", str(real), "-c", "-p", "more", "--output-format", "json")
    assert json.loads(out)["session_id"] == "viaalink000"
    assert shown(clients["m"].requests[0])[0] == ("user", "an earlier question")


def test_continue_with_a_root_given_through_a_symlink_finds_what_was_started_there(
    ncc_home, tmp_path, serve, capsys
):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    serve(m=[says("a"), says("b")])
    first = start(capsys, link)
    _, out, _ = run_ncc(capsys, "--root", str(real), "-c", "-p", "more", "--output-format", "json")
    assert json.loads(out)["session_id"] == first


def test_the_advice_to_run_ncc_from_another_directory_quotes_it_for_a_shell(
    ncc_home, tmp_path, serve, capsys
):
    here, there = tmp_path / "here", tmp_path / "the other one"
    here.mkdir()
    there.mkdir()
    serve(m=[says("a")])
    in_there = start(capsys, there)
    _, _, err = run_ncc(capsys, "--root", str(here), "--resume", in_there, "-p", "hi")
    assert err == (
        f"error: session {in_there} was started in {there}, not in {here} "
        f"\u2014 run ncc from '{there}' (or pass --root '{there}'), or leave out --resume\n"
    )


def test_a_session_is_resumed_from_a_directory_spelled_in_another_case_where_the_disk_agrees(
    ncc_home, tmp_path, serve, capsys
):
    (tmp_path / "Project").mkdir()
    if not (tmp_path / "project").exists():
        pytest.skip("this file system tells Project from project")
    serve(m=[says("a"), says("b")])
    first = start(capsys, tmp_path / "Project")
    code, out, _ = run_ncc(capsys, "--root", str(tmp_path / "project"), "-r", first, "-p", "again")
    assert (code, out) == (EXIT_CODES["completed"], "b\n")
