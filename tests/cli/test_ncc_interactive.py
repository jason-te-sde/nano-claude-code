"""Tests for ``ncc`` with no prompt: the REPL, built from the same flags and configuration.

The REPL itself is tested in ``test_repl.py``; what is tested here is that the command builds
the session it runs, hands it the front end it needs, and ends cleanly however it ends.
``run_repl`` is replaced by a stand-in that records what it was given, or wraps the real one
with a scripted keyboard.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from rich.console import Console

from nanoclaude.agent.session import Session, SessionChangedError
from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.cli.render import ConsoleUI
from nanoclaude.cli.repl import run_repl
from nanoclaude.conversation.store import Store
from nanoclaude.permissions.policy import PermissionMode
from nanoclaude.testing.scripted import says
from tests.cli.helpers import ScriptedPrompter, run_ncc


class Repl:
    """A stand-in for ``run_repl`` that records what it was given and does ``body``."""

    def __init__(self) -> None:
        self.session: Session | None = None
        self.console: Console | None = None
        self.history: str | None = None
        self.body: Callable[[Session], Any] | None = None
        self.result = 0

    async def __call__(
        self, session: Session, console: Console, history_path: str | None = None, **_: object
    ) -> int:
        self.session, self.console, self.history = session, console, history_path
        if self.body is not None:
            outcome = self.body(session)
            if hasattr(outcome, "__await__"):
                await outcome
        return self.result


def ran(repl: Repl) -> Session:
    """The session the REPL was given: it must have been started."""
    assert repl.session is not None
    return repl.session


@pytest.fixture
def repl(monkeypatch: pytest.MonkeyPatch) -> Repl:
    stand_in = Repl()
    monkeypatch.setattr(ncc_main, "run_repl", stand_in)
    return stand_in


@pytest.fixture
def interfaces(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What ``ncc`` passed each ``ConsoleUI`` it built (the real one is still built)."""
    built: list[dict[str, Any]] = []

    def make(console: Console, **options: Any) -> ConsoleUI:
        built.append({"console": console, **options})
        return ConsoleUI(console, **options)

    monkeypatch.setattr(ncc_main, "ConsoleUI", make)
    return built


# --------------------------------------------------------------------------
# Starting the REPL
# --------------------------------------------------------------------------


def test_with_no_prompt_the_repl_runs_in_the_root_with_a_history_under_the_home(
    ncc_home, project, serve, capsys, repl
):
    serve(m=[])
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert (code, out, err) == (EXIT_CODES["completed"], "", "")
    assert ran(repl).root == str(project)
    assert repl.history == str(ncc_home / ".nanoclaude" / "history")


def test_the_exit_code_is_the_repls(ncc_home, project, serve, capsys, repl):
    serve(m=[])
    repl.result = 7
    assert run_ncc(capsys, "--root", str(project))[0] == 7


def test_the_repl_writes_to_a_stdout_console_and_the_errors_do_not(
    ncc_home, project, serve, capsys, repl
):
    serve(m=[])
    run_ncc(capsys, "--root", str(project))
    assert repl.console is not None and repl.console.stderr is False


@pytest.mark.parametrize("how", ["flag", "environment"])
def test_no_color_reaches_the_console_the_repl_writes_to(
    ncc_home, project, serve, capsys, repl, monkeypatch, how
):
    serve(m=[])
    argv = ["--root", str(project)]
    if how == "flag":
        argv.append("--no-color")
    else:
        monkeypatch.setenv("NO_COLOR", "1")
    run_ncc(capsys, *argv)
    assert repl.console is not None and repl.console.no_color is True


def test_the_console_has_colour_unless_told_otherwise(ncc_home, project, serve, capsys, repl):
    serve(m=[])
    run_ncc(capsys, "--root", str(project))
    assert repl.console is not None and repl.console.no_color is False


# --------------------------------------------------------------------------
# The front end it is given
# --------------------------------------------------------------------------


def test_the_ui_shows_paths_relative_to_the_sessions_root(
    ncc_home, project, serve, capsys, repl, interfaces
):
    serve(m=[])
    run_ncc(capsys, "--root", str(project))
    assert len(interfaces) == 1
    assert interfaces[0]["root"] == ran(repl).root == str(project)
    assert interfaces[0]["console"] is repl.console


def test_the_ui_names_the_model_that_plays_a_role_as_the_router_has_it_now(
    ncc_home, project, serve, capsys, repl, interfaces
):
    serve(m=[], other=[])
    seen: list[str] = []

    async def switch(session: Session) -> None:
        model_of = interfaces[0]["model_of"]
        seen.append(model_of("main"))
        await session.router.route("main", "other")  # what /model does
        seen.append(model_of("main"))
        seen.append(model_of("compact"))

    repl.body = switch
    run_ncc(capsys, "--root", str(project))
    assert seen == ["claude-sonnet-5", "claude-haiku-4-5", "claude-sonnet-5"]


def test_the_ui_is_the_one_the_session_talks_through(
    ncc_home, project, serve, capsys, repl, interfaces
):
    serve(m=[])
    run_ncc(capsys, "--root", str(project))
    assert isinstance(ran(repl).ui, ConsoleUI)


def test_a_reply_in_the_repl_is_shown_with_the_model_named_in_the_banner(
    ncc_home, project, serve, capsys, monkeypatch
):
    serve(m=[says("It works.")])
    keyboard = ScriptedPrompter("hello")

    async def with_keyboard(
        session: Session, console: Console, history_path: str | None = None
    ) -> int:
        return await run_repl(session, console, history_path, prompt=keyboard)

    monkeypatch.setattr(ncc_main, "run_repl", with_keyboard)
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert code == EXIT_CODES["completed"] and err == ""
    assert "claude-sonnet-5" in out and "It works." in out
    assert "bye" in out


# --------------------------------------------------------------------------
# The same flags
# --------------------------------------------------------------------------


def test_the_flags_build_the_session_the_repl_runs(ncc_home, project, serve, capsys, repl):
    lib = project.parent / "lib"
    lib.mkdir()
    serve(m=[], other=[])
    run_ncc(
        capsys,
        "--root",
        str(project),
        "--add-dir",
        str(lib),
        "--model",
        "other",
        "--role",
        "compact=m",
        "--mode",
        "plan",
        "--max-turns",
        "3",
        "--allow-secrets",
    )
    session = ran(repl)
    assert session.policy.sandbox.roots == (str(project), str(lib))
    assert session.router.config.roles.main == "other"
    assert session.router.config.roles.compact == "m"
    assert session.policy.mode is PermissionMode.PLAN
    assert session.config.limits.max_turns == 3 and session.state.max_turns == 3
    assert session.policy.allow_secrets is True and session.redactor.enabled is False


def test_resume_hands_the_repl_the_stored_conversation(ncc_home, project, serve, capsys, repl):
    serve(m=[says("one")])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "first", "--output-format", "json")
    first = json.loads(out)["session_id"]
    seen: list[list[str]] = []
    repl.body = lambda session: seen.append([m.text() for m in session.state.transcript.messages])
    run_ncc(capsys, "--root", str(project), "--resume", first)
    assert ran(repl).session_id == first
    assert seen == [["first", "one"]]


def test_continue_takes_up_the_latest_session_here_in_the_repl_too(
    ncc_home, project, serve, capsys, repl
):
    serve(m=[says("one")])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "first", "--output-format", "json")
    first = json.loads(out)["session_id"]
    run_ncc(capsys, "--root", str(project), "-c")
    assert ran(repl).session_id == first


def test_turning_prompts_off_is_announced_before_the_repl_starts(
    ncc_home, project, serve, capsys, repl
):
    serve(m=[])
    _, out, err = run_ncc(capsys, "--root", str(project), "--dangerously-skip-permissions")
    assert out == "" and err.startswith("warning: permission prompts are off ")
    assert ran(repl).policy.mode is PermissionMode.BYPASS


# --------------------------------------------------------------------------
# However it ends
# --------------------------------------------------------------------------


def closed(store: Store) -> bool:
    try:
        store.db  # noqa: B018
    except RuntimeError:
        return True
    return False


def test_the_session_is_closed_when_the_repl_returns(
    ncc_home, project, serve, capsys, repl, stores
):
    clients = serve(m=[])
    run_ncc(capsys, "--root", str(project))
    assert closed(stores[0]) and clients["m"].closed


def test_a_repl_that_is_interrupted_exits_130_and_still_closes_the_session(
    ncc_home, project, serve, capsys, repl, stores
):
    # Ctrl+C in the instructions between a turn's handler being removed and the next prompt
    # is not caught by the REPL, and arrives here out of asyncio.run.
    clients = serve(m=[])

    def interrupt(_session: Session) -> None:
        raise KeyboardInterrupt

    repl.body = interrupt
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert code == EXIT_CODES["interrupted"]
    assert out == "" and err == "interrupted\n"
    assert closed(stores[0]) and clients["m"].closed


def test_a_session_changed_by_another_process_while_closing_is_reported(
    ncc_home, project, serve, capsys, repl, stores
):
    serve(m=[])

    async def refuse() -> None:
        raise SessionChangedError("session x was changed by another process — resume it again")

    def spoil(session: Session) -> None:
        session.aclose = refuse  # type: ignore[method-assign]

    repl.body = spoil
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert code == EXIT_CODES["stopped"] and out == ""
    assert err == "error: session x was changed by another process — resume it again\n"


@pytest.mark.parametrize(
    ("argv", "code", "needle"),
    [
        (["--mode", "bypass"], EXIT_CODES["usage"], "--dangerously-skip-permissions"),
        (["--model", "ghost"], EXIT_CODES["config"], "which is not defined"),
        (["--resume", "nope"], EXIT_CODES["usage"], 'no stored session "nope"'),
    ],
)
def test_what_goes_wrong_before_the_repl_starts_is_one_line_on_stderr(
    ncc_home, project, capsys, repl, argv, code, needle
):
    got, out, err = run_ncc(capsys, "--root", str(project), *argv)
    assert got == code
    assert out == "" and err.startswith("error: ") and needle in err and err.count("\n") == 1
    assert repl.session is None


def test_no_configuration_at_all_points_at_init_in_the_repl_too(
    tmp_path, monkeypatch, capsys, repl
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(tmp_path))
    code, out, err = run_ncc(capsys, "--root", str(tmp_path))
    assert code == EXIT_CODES["config"]
    assert out == "" and "ncc init" in err
