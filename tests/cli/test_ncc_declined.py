"""Tests for what ``ncc -p`` says about the calls it declined.

Headless mode has nobody to ask, so a call the policy wanted confirmed is declined. That is
the right default in CI, and it used to be invisible: a declined Write ended in exit 0, nothing
on stderr, and the model's apology on stdout, so a script that wanted the file could not tell
that it had not been written. Now each declined call is one line on stderr, in the form of
spec 17.9, with what to do about it. The exit code is what it was.
"""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from nanoclaude.agent.ui import Approval, AutoDecline
from nanoclaude.cli.headless import ReportingDecline, declined_line
from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.conversation.transcript import ToolUseBlock
from nanoclaude.permissions.policy import (
    Decision,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
)
from nanoclaude.providers.base import ModelReply
from nanoclaude.testing.scripted import calls, calls_many, says
from tests.cli.helpers import plain_console, run_ncc

ASKED = PermissionResult(Decision.ASK, "default.ask", "needs confirmation")


def write_call(path: str = "src/a.py", call_id: str = "w1") -> ModelReply:
    return calls("Write", {"path": path, "content": "x = 1\n"}, call_id=call_id)


def edit_request(root: str, path: str = "src/a.py") -> PermissionRequest:
    resolved = f"{root}/{path}"
    return PermissionRequest("Write", resolved, (resolved,), is_write=True)


# --------------------------------------------------------------------------
# The line
# --------------------------------------------------------------------------


def test_a_declined_file_edit_says_which_flag_lets_headless_edit_files():
    line = declined_line("Write", edit_request("/p"), root="/p", mode=PermissionMode.DEFAULT)
    assert line == "declined Write src/a.py — pass --mode accept-edits to let -p edit files"


@pytest.mark.parametrize("tool", ["Edit", "Write"])
def test_both_file_edits_get_the_same_advice(tool):
    request = PermissionRequest(tool, "/p/a.py", ("/p/a.py",), is_write=True)
    line = declined_line(tool, request, root="/p", mode=PermissionMode.DEFAULT)
    assert line == f"declined {tool} a.py — pass --mode accept-edits to let -p edit files"


def test_anything_else_is_to_be_allowed_by_a_rule_and_not_by_a_mode_that_would_not_help():
    # accept-edits lets file edits through and nothing else, so it is not the advice for a
    # call to another tool, nor for one declined with that mode already on.
    other = declined_line(
        "Git", PermissionRequest("Git", "status"), root="/p", mode=PermissionMode.DEFAULT
    )
    assert other == (
        'declined Git status — add "Git" to allow under [permissions] in your config '
        "to let -p run it"
    )
    again = declined_line("Write", edit_request("/p"), root="/p", mode=PermissionMode.ACCEPT_EDITS)
    assert "accept-edits" not in again and 'add "Write" to allow' in again


def test_a_path_outside_the_root_and_a_command_are_shown_as_they_are():
    outside = PermissionRequest("Write", "/etc/x", ("/etc/x",), is_write=True)
    assert declined_line("Write", outside, root="/p", mode=PermissionMode.DEFAULT).startswith(
        "declined Write /etc/x — "
    )
    command = PermissionRequest("Bash", "npm test -- --watch")
    assert declined_line("Bash", command, root="/p", mode=PermissionMode.DEFAULT).startswith(
        "declined Bash npm test -- --watch — "
    )


def test_what_the_model_wrote_cannot_break_the_line_or_reach_the_terminal():
    nasty = PermissionRequest("Bash", "echo hi\n\x1b[2J\x1b]0;owned\x07" + "x" * 500)
    line = declined_line("Bash", nasty, root="/p", mode=PermissionMode.DEFAULT)
    assert "\n" not in line and "\x1b" not in line and "\x07" not in line
    assert "\\n" in line and "\\x1b" in line  # shown, not acted on
    assert len(line) < 250 and "..." in line  # and not five hundred characters of it


# --------------------------------------------------------------------------
# The front end
# --------------------------------------------------------------------------


async def test_the_front_end_declines_and_says_so_once_per_call():
    console, written = plain_console(200)
    ui = ReportingDecline(console, root="/p", mode=PermissionMode.DEFAULT)
    call = ToolUseBlock("w1", "Write", {"path": "src/a.py", "content": "x"})
    answers = [await ui.confirm(call, edit_request("/p"), ASKED) for _ in range(2)]
    assert answers == [Approval.NO, Approval.NO]
    assert written.getvalue() == (
        "ncc: declined Write src/a.py — pass --mode accept-edits to let -p edit files\n" * 2
    )


def test_it_is_still_the_front_end_that_declines_everything_it_is_asked():
    # The session and the executor know it by the protocol, and tests know it by this.
    assert issubclass(ReportingDecline, AutoDecline)


async def test_the_line_is_not_read_as_markup_and_is_not_wrapped():
    err = Console(stderr=False, markup=False, highlight=False, soft_wrap=True, width=40)
    with err.capture() as captured:
        ui = ReportingDecline(err, root="/p", mode=PermissionMode.DEFAULT)
        request = PermissionRequest("Bash", "echo [bold]x[/bold] and a long, long, long command")
        await ui.confirm(ToolUseBlock("b1", "Bash", {}), request, ASKED)
    assert captured.get().count("\n") == 1 and "[bold]x[/bold]" in captured.get()


# --------------------------------------------------------------------------
# ncc -p
# --------------------------------------------------------------------------

DECLINED_WRITE = "ncc: declined Write a.txt — pass --mode accept-edits to let -p edit files\n"


def test_a_declined_write_is_one_line_on_stderr_and_the_exit_code_is_what_it_was(
    ncc_home, project, serve, capsys
):
    serve(m=[write_call("a.txt"), says("I could not write the file.")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "write a.txt")
    assert (code, out) == (EXIT_CODES["completed"], "I could not write the file.\n")
    assert err == DECLINED_WRITE
    assert not (project / "a.txt").exists()


def test_a_path_given_in_full_is_shown_as_it_is_inside_the_project(
    ncc_home, project, serve, capsys
):
    serve(m=[write_call(str(project / "src" / "b.py")), says("no")])
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert err == ("ncc: declined Write src/b.py — pass --mode accept-edits to let -p edit files\n")


def test_every_declined_call_gets_its_own_line_in_the_order_asked(ncc_home, project, serve, capsys):
    two = calls_many(
        ("Write", {"path": "one.txt", "content": "1"}, "w1"),
        ("Write", {"path": "two.txt", "content": "2"}, "w2"),
    )
    serve(m=[two, says("done")])
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "go")
    lines = err.splitlines()
    assert [line.split(" — ")[0] for line in lines] == [
        "ncc: declined Write one.txt",
        "ncc: declined Write two.txt",
    ]


def test_nothing_is_said_about_calls_that_ran(ncc_home, project, serve, capsys):
    (project / "a.txt").write_text("x\n")
    serve(m=[calls("Read", {"path": "a.txt"}, call_id="r1"), says("ok")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert (code, out, err) == (EXIT_CODES["completed"], "ok\n", "")


def test_with_the_mode_that_lets_edits_through_nothing_is_declined_and_nothing_is_said(
    ncc_home, project, serve, capsys
):
    serve(m=[write_call("a.txt"), says("written")])
    code, _, err = run_ncc(
        capsys, "--root", str(project), "--mode", "accept-edits", "-p", "write a.txt"
    )
    assert (code, err) == (EXIT_CODES["completed"], "")
    assert (project / "a.txt").exists()


def test_a_call_refused_by_the_policy_is_not_a_declined_one(ncc_home, project, serve, capsys):
    # plan mode refuses writes by rule: nobody was asked, and the model is told which rule.
    serve(m=[write_call("a.txt"), says("refused")])
    code, _, err = run_ncc(capsys, "--root", str(project), "--mode", "plan", "-p", "go")
    assert (code, err) == (EXIT_CODES["completed"], "")


def test_the_lines_are_there_in_json_mode_too_and_stdout_is_one_object(
    ncc_home, project, serve, capsys
):
    serve(m=[write_call("a.txt"), says("could not")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "go", "--output-format", "json")
    assert (code, err) == (EXIT_CODES["completed"], DECLINED_WRITE)
    assert json.loads(out)["result"] == "could not"
