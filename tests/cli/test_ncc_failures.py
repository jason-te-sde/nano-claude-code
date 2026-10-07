"""Tests for what ``ncc`` does when something goes wrong that nobody planned for.

A script that runs ``ncc -p`` reads the exit code and the first line of stderr. A Python
traceback is neither: it is forty lines of the program's own insides, and the exit code is
whatever Python makes of it. So every exception that is not one of the mapped ones (usage,
configuration, credentials, provider, session) ends the run as a single line in the form of
spec 17.9, with nothing on stdout and exit 1. Two of them are common enough to be worded for
the person: a store another process is using, and a store that cannot be opened.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.agent.router import Router
from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.conversation.store import Store
from nanoclaude.providers.base import ModelReply, StopKind, Usage
from nanoclaude.testing.session import ScriptedClient
from tests.cli.helpers import run_ncc

FORMATS = [[], ["--output-format", "json"]]

UNEXPECTED = "error: unexpected error ({what}) — this is a bug; please report it\n"


class Exploding(ScriptedClient):
    """A model whose request fails with whatever it was given: not a ModelError."""

    def __init__(self, failure: BaseException) -> None:
        super().__init__([])
        self._failure = failure

    async def complete(self, _request, *, on_text=None):  # noqa: ARG002
        raise self._failure


def explode_with(monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> Exploding:
    client = Exploding(failure)
    monkeypatch.setattr(
        ncc_main, "Router", lambda config, cache: Router(config, cache, {"m": client})
    )
    return client


def closed(store: Store) -> bool:
    try:
        store.db  # noqa: B018
    except RuntimeError:
        return True
    return False


# --------------------------------------------------------------------------
# Anything else: one line, nothing on stdout, exit 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS, ids=["text", "json"])
def test_an_exception_nobody_mapped_is_one_line_and_exit_one(
    ncc_home, project, capsys, monkeypatch, stores, fmt
):
    client = explode_with(monkeypatch, RuntimeError("boom"))
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello", *fmt)
    assert code == EXIT_CODES["stopped"]
    assert out == ""
    assert err == UNEXPECTED.format(what="RuntimeError: boom")
    # The session was closed on the way out, as for any other end.
    assert client.closed and closed(stores[0])


def test_what_the_exception_says_is_one_line_and_never_markup_or_a_control_sequence(
    ncc_home, project, capsys, monkeypatch
):
    awkward = "bad [/] and [bold]x[/bold]\nsecond  line " + "y" * 200 + " \x1b]0;t\x07 end"
    explode_with(monkeypatch, ValueError(awkward))
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["stopped"] and out == ""
    assert err.count("\n") == 1  # not wrapped, and the line break in the message is not one
    assert err.startswith(
        "error: unexpected error (ValueError: bad [/] and [bold]x[/bold] second line "
    )
    assert "y" * 200 in err and "\x1b" not in err and "\x07" not in err


def test_an_exception_with_no_message_is_named_by_its_type(ncc_home, project, capsys, monkeypatch):
    explode_with(monkeypatch, KeyError())
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert err == UNEXPECTED.format(what="KeyError")


def test_a_conversation_the_loop_refuses_is_reported_and_not_raised(
    ncc_home, project, serve, capsys
):
    # A reply with nothing in it: the loop raises LoopError, which nobody maps.
    empty = ModelReply((), StopKind.END_TURN, Usage(), "scripted")
    serve(m=[empty])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out) == (EXIT_CODES["stopped"], "")
    assert err == UNEXPECTED.format(what="LoopError: model returned an empty reply")


def test_a_store_that_fails_part_way_is_reported_in_the_same_way(
    ncc_home, project, serve, capsys, monkeypatch
):
    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    serve(m=[])
    monkeypatch.setattr(Store, "append_message", broken)
    code, out, err = run_ncc(
        capsys, "--root", str(project), "-p", "hello", "--output-format", "json"
    )
    assert (code, out) == (EXIT_CODES["stopped"], "")
    assert err == UNEXPECTED.format(what="OperationalError: disk I/O error")


def test_the_exceptions_that_are_mapped_are_not_swallowed_by_the_general_one(
    ncc_home, project, capsys
):
    # The general clause is last. If it came first, each of these would be exit 1.
    assert run_ncc(capsys, "--root", str(project), "--model", "ghost", "-p", "hi")[0] == 3
    assert run_ncc(capsys, "--root", str(project), "--resume", "nope", "-p", "hi")[0] == 2
    assert run_ncc(capsys, "--root", str(project), "--mode", "bypass", "-p", "hi")[0] == 2


def test_an_interrupt_is_still_130_and_not_an_unexpected_error(
    ncc_home, project, capsys, monkeypatch
):
    async def interrupted(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(ncc_main, "_run_headless", interrupted)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out, err) == (EXIT_CODES["interrupted"], "", "interrupted\n")


def test_the_repl_is_covered_too(ncc_home, project, capsys, monkeypatch):
    async def explodes(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("the REPL broke")

    monkeypatch.setattr(ncc_main, "run_repl", explodes)
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert (code, out) == (EXIT_CODES["stopped"], "")
    assert err == UNEXPECTED.format(what="RuntimeError: the REPL broke")


# --------------------------------------------------------------------------
# A store another process is using
# --------------------------------------------------------------------------

BUSY = "error: the session store is busy — another ncc may be using it; try again\n"


@pytest.fixture
def impatient(monkeypatch: pytest.MonkeyPatch) -> None:
    """SQLite waits five seconds for a lock by default. Nobody is waiting here."""
    connect: Callable[..., sqlite3.Connection] = sqlite3.connect

    def at_once(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return connect(*args, **{**kwargs, "timeout": 0})

    monkeypatch.setattr(sqlite3, "connect", at_once)


@pytest.mark.parametrize("fmt", FORMATS, ids=["text", "json"])
def test_a_store_another_ncc_holds_a_lock_on_is_busy_and_not_a_bug(
    ncc_home, project, serve, capsys, impatient, stores, fmt
):
    database = ncc_home / ".nanoclaude" / "sessions.db"
    first = Store(database)
    first.open()
    first.close()
    serve(m=[])
    other = sqlite3.connect(database)
    other.execute("BEGIN IMMEDIATE")  # what another ncc does while it writes
    try:
        code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello", *fmt)
    finally:
        other.rollback()
        other.close()
    assert (code, out, err) == (EXIT_CODES["stopped"], "", BUSY)
    assert closed(stores[0])


def test_a_store_that_is_busy_as_it_is_opened_is_busy_and_not_unopenable(
    ncc_home, project, capsys, monkeypatch
):
    def busy(self: Store) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Store, "open", busy)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out, err) == (EXIT_CODES["stopped"], "", BUSY)


@pytest.mark.parametrize(
    "message",
    [
        "database is locked",
        "database table is locked",
        "DATABASE IS LOCKED",
        "the database is busy",
    ],
)
def test_a_store_that_is_locked_or_busy_is_worded_for_the_person(message):
    line = ncc_main._unexpected(sqlite3.OperationalError(message))
    assert line == BUSY.removeprefix("error: ").rstrip("\n")


@pytest.mark.parametrize(
    "failure",
    [
        sqlite3.OperationalError("disk I/O error"),
        sqlite3.DatabaseError("database is locked"),  # not an OperationalError
        RuntimeError("database is locked"),
    ],
    ids=["another operational error", "another sqlite3 error", "not sqlite3 at all"],
)
def test_only_an_operational_error_that_says_locked_or_busy_is_the_busy_line(failure):
    assert "busy" not in ncc_main._unexpected(failure)
    assert ncc_main._unexpected(failure).startswith("unexpected error (")


# --------------------------------------------------------------------------
# A store that cannot be opened
# --------------------------------------------------------------------------

CANNOT_OPEN = (
    "error: cannot open the session store at {path} ({detail}) "
    "— check that the directory exists and that you can write to it\n"
)


def test_an_os_error_opening_the_store_names_the_path(
    ncc_home, project, capsys, monkeypatch, stores
):
    database = ncc_home / ".nanoclaude" / "sessions.db"

    def refuse(self: Store) -> None:
        raise PermissionError(13, "Permission denied", str(database))

    monkeypatch.setattr(Store, "open", refuse)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out) == (EXIT_CODES["stopped"], "")
    assert err == CANNOT_OPEN.format(path=database, detail="Permission denied")


def test_a_database_that_cannot_be_opened_names_the_path_too(ncc_home, project, capsys, stores):
    # The file is a directory: SQLite says it cannot open it, which is not an OSError.
    database = ncc_home / ".nanoclaude" / "sessions.db"
    database.mkdir()
    code, out, err = run_ncc(
        capsys, "--root", str(project), "-p", "hello", "--output-format", "json"
    )
    assert (code, out) == (EXIT_CODES["stopped"], "")
    assert err == CANNOT_OPEN.format(path=database, detail="unable to open database file")
    assert closed(stores[0])  # the connection it did make is not left open


def test_a_store_that_cannot_be_opened_is_the_same_line_in_the_repl(
    ncc_home, project, capsys, monkeypatch
):
    database = ncc_home / ".nanoclaude" / "sessions.db"
    database.mkdir()
    started: list[object] = []

    async def never(*args: object, **_kwargs: object) -> int:
        started.append(args)
        return 0

    monkeypatch.setattr(ncc_main, "run_repl", never)
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert (code, out) == (EXIT_CODES["stopped"], "")
    assert err == CANNOT_OPEN.format(path=database, detail="unable to open database file")
    assert started == []


def test_the_open_error_says_the_path_and_what_the_system_said():
    path = Path("/a/b/sessions.db")
    error = ncc_main.StoreOpenError(path, OSError(2, "No such file or directory"))
    assert str(error) == (
        "cannot open the session store at /a/b/sessions.db (No such file or directory) "
        "— check that the directory exists and that you can write to it"
    )
