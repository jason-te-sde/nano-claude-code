"""Tests for the mistakes a person can make in the command line, and how ncc says so.

Every one of them is a line in the form of spec 17.9 on stderr, with nothing on stdout, and
exit 2: the same contract as every other error. argparse's own way of ending a run (a block of
usage text, then ``ncc: error: ...``, with whatever was typed echoed raw) is not that, so the
parser reports through ``UsageError`` and the command prints it.
"""

from __future__ import annotations

import sys

import pytest

from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.main import EXIT_CODES, UsageError, build_parser
from nanoclaude.cli.main import _stdin_is_terminal as real_stdin_is_terminal
from nanoclaude.testing.scripted import says
from tests.cli.helpers import run_ncc

# --------------------------------------------------------------------------
# What argparse found wrong, in the form of spec 17.9
# --------------------------------------------------------------------------

FLAG_ERRORS = [
    (
        ["--mode", "foo", "-p", "x"],
        "error: --mode foo is not a mode — choose one of default, plan, accept-edits",
    ),
    (
        ["--output-format", "xml", "-p", "x"],
        "error: --output-format xml is not a format — choose one of text, json",
    ),
    (["-p"], "error: -p needs a prompt — pass the task as its argument"),
    (["--print"], "error: -p needs a prompt — pass the task as its argument"),
    (["--root"], "error: --root needs a value — pass one"),
    (["-p", "x", "--role"], "error: --role needs a value — pass one"),
    (["--max-turns"], "error: --max-turns needs a value — pass one"),
    (
        ["--max-turns", "many", "-p", "x"],
        "error: --max-turns needs a whole number of at least 1 — got 'many'",
    ),
    (
        ["-c", "-r", "abc", "-p", "x"],
        "error: --resume and --continue cannot be combined — pass only one",
    ),
    (
        ["-r", "abc", "-c", "-p", "x"],
        "error: --continue and --resume cannot be combined — pass only one",
    ),
    (
        ["--bogus", "-p", "x"],
        "error: unrecognized argument --bogus — run ncc --help for the flags there are",
    ),
    (
        ["--bogus", "--other", "-p", "x"],
        "error: unrecognized arguments --bogus --other — run ncc --help for the flags there are",
    ),
    (
        ["--allow-secrets=1", "-p", "x"],
        (
            "error: argument --allow-secrets: ignored explicit argument '1' "
            "— run ncc --help for the flags there are"
        ),
    ),
]


@pytest.mark.parametrize(("argv", "line"), FLAG_ERRORS)
def test_a_mistake_in_the_flags_is_one_line_in_the_form_of_spec_17_9(capsys, argv, line):
    code, out, err = run_ncc(capsys, *argv)
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == line + "\n"
    assert "usage:" not in err and "ncc: error" not in err


@pytest.mark.parametrize(
    ("argv", "line"),
    [
        (
            ["--bogus\x1b]0;owned\x07", "-p", "x"],
            "error: unrecognized argument --bogus — run ncc --help for the flags there are",
        ),
        (
            ["--bo\ngus\x1b[2J", "-p", "x"],
            "error: unrecognized argument --bo gus — run ncc --help for the flags there are",
        ),
        (
            ["--mode", "x\x1b[2Jy", "-p", "x"],
            "error: --mode xy is not a mode — choose one of default, plan, accept-edits",
        ),
        (
            ["--output-format", "a\nb", "-p", "x"],
            "error: --output-format a b is not a format — choose one of text, json",
        ),
    ],
)
def test_what_was_typed_is_not_echoed_with_anything_a_terminal_would_act_on(capsys, argv, line):
    # argparse puts the arguments into its message as they are. The line is one line, and
    # holds no escape sequence, however the arguments were made up.
    code, out, err = run_ncc(capsys, *argv)
    assert (code, out, err) == (EXIT_CODES["usage"], "", line + "\n")


def test_a_mistake_in_the_flags_ends_the_parser_with_the_command_s_own_error(capsys):
    # The parser used on its own, as the tests of a front end use it, does not print or exit.
    with pytest.raises(UsageError, match=r"^--mode foo is not a mode") as caught:
        build_parser().parse_args(["--mode", "foo"])
    assert str(caught.value) == (
        "--mode foo is not a mode — choose one of default, plan, accept-edits"
    )
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_help_and_version_are_for_stdout_and_are_not_errors(capsys):
    code, out, err = run_ncc(capsys, "--help")
    assert (code, err) == (0, "")
    assert out.startswith("usage: ncc") and "--dangerously-skip-permissions" in out
    code, out, err = run_ncc(capsys, "--version")
    assert (code, err) == (0, "")
    assert out.startswith("nano-claude-code ")


def test_a_mistake_in_the_flags_is_not_coloured_when_colour_is_off(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    _, _, coloured = run_ncc(capsys, "--bogus")
    assert "\x1b[" in coloured
    _, _, plain = run_ncc(capsys, "--no-color", "--bogus")
    assert "\x1b[" not in plain and plain.startswith("error: ")
    monkeypatch.setenv("NO_COLOR", "1")
    _, _, plain = run_ncc(capsys, "--bogus")
    assert "\x1b[" not in plain


# --------------------------------------------------------------------------
# A directory flag that holds nothing is not the current directory
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "   "])
@pytest.mark.parametrize("flag", ["--root", "--add-dir"])
def test_a_blank_directory_is_a_usage_error_and_never_the_current_directory(
    ncc_home, project, serve, capsys, monkeypatch, flag, value
):
    # What --add-dir "$UNSET" gives. It used to widen the sandbox to the directory ncc was
    # run in, and a Read there succeeded.
    monkeypatch.chdir(project)
    (project / "a.txt").write_text("secret\n")
    clients = serve(m=[])
    argv = ["--root", str(project), "-p", "hi"] if flag == "--add-dir" else ["-p", "hi"]
    code, out, err = run_ncc(capsys, *argv, flag, value)
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == f"error: {flag} needs a directory — pass one, or leave the flag out\n"
    assert clients["m"].requests == []


# --------------------------------------------------------------------------
# A flag that is given and holds nothing is not a flag that was left out
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_resume_is_a_usage_error_and_not_a_new_session(
    ncc_home, project, serve, capsys, stores, value
):
    clients = serve(m=[])
    code, out, err = run_ncc(capsys, "--root", str(project), "-r", value, "-p", "hi")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == "error: --resume needs a session id — pass one, or use --continue\n"
    assert clients["m"].requests == [] and stores == []  # nothing was started, or opened


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_model_is_a_usage_error_and_not_the_default(
    ncc_home, project, serve, capsys, stores, value
):
    clients = serve(m=[])
    code, out, err = run_ncc(capsys, "--root", str(project), "--model", value, "-p", "hi")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == "error: --model needs a model alias — pass one, or leave the flag out\n"
    assert clients["m"].requests == [] and stores == []


# --------------------------------------------------------------------------
# --output-format shapes the answer to one prompt
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_an_output_format_with_no_prompt_is_a_usage_error(ncc_home, project, capsys, fmt):
    code, out, err = run_ncc(capsys, "--root", str(project), "--output-format", fmt)
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        "error: --output-format needs -p — it shapes the answer to one prompt; "
        "pass a prompt with -p\n"
    )


# --------------------------------------------------------------------------
# --model and --role main= say the same thing, or they contradict each other
# --------------------------------------------------------------------------


def test_a_model_and_a_main_role_that_disagree_are_a_usage_error(
    ncc_home, project, serve, capsys, stores
):
    clients = serve(m=[], other=[])
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--model", "m", "--role", "main=other", "-p", "hi"
    )
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        "error: --model m and --role main=other say different things about the main role "
        "— pass only one of them\n"
    )
    assert clients["m"].requests == [] and clients["other"].requests == [] and stores == []


def test_a_model_and_a_main_role_that_agree_are_fine(ncc_home, project, serve, capsys):
    serve(m=[says("from m")], other=[says("from other")])
    code, out, _ = run_ncc(
        capsys, "--root", str(project), "--model", "other", "--role", "main=other", "-p", "hi"
    )
    assert (code, out) == (EXIT_CODES["completed"], "from other\n")


# --------------------------------------------------------------------------
# No prompt and nobody to type one
# --------------------------------------------------------------------------


def test_no_prompt_and_no_terminal_is_a_usage_error_and_not_a_quiet_success(
    ncc_home, project, serve, capsys, monkeypatch
):
    # A script that lost its -p would otherwise start the REPL on an empty pipe, read the end
    # of it, say goodbye and exit 0.
    monkeypatch.setattr(ncc_main, "_stdin_is_terminal", lambda: False)
    clients = serve(m=[])
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        "error: no prompt and no terminal — pass a prompt with -p, or run ncc in a terminal\n"
    )
    assert clients["m"].requests == []


def test_a_prompt_needs_no_terminal(ncc_home, project, serve, capsys, monkeypatch):
    monkeypatch.setattr(ncc_main, "_stdin_is_terminal", lambda: False)
    serve(m=[says("fine")])
    assert run_ncc(capsys, "--root", str(project), "-p", "hi")[:2] == (0, "fine\n")


class _Stdin:
    def __init__(self, answer: bool | Exception) -> None:
        self._answer = answer

    def isatty(self) -> bool:
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


@pytest.mark.parametrize(
    ("stdin", "terminal"),
    [
        (_Stdin(True), True),
        (_Stdin(False), False),
        (_Stdin(ValueError("I/O operation on closed file")), False),
        (None, False),
    ],
    ids=["terminal", "pipe", "closed", "no stdin at all"],
)
def test_the_terminal_question_is_asked_of_stdin_and_survives_what_stdin_can_be(
    monkeypatch, stdin, terminal
):
    monkeypatch.setattr(sys, "stdin", stdin)
    assert real_stdin_is_terminal() is terminal


@pytest.mark.parametrize(
    ("message", "line"),
    [
        # A Python that prints the value as it is, and not as its repr.
        (
            "argument --mode: invalid choice: foo (choose from default, plan)",
            "--mode foo is not a mode \u2014 choose one of default, plan, accept-edits",
        ),
        (
            "argument --output-format: invalid choice: 'yaml' (choose from 'text', 'json')",
            "--output-format yaml is not a format \u2014 choose one of text, json",
        ),
        # Something argparse can say that nothing here knows about.
        (
            "something new: argument --x is bad",
            "something new: argument --x is bad \u2014 run ncc --help for the flags there are",
        ),
    ],
)
def test_what_argparse_says_is_reworded_by_what_it_is_and_never_lost(message, line):
    assert ncc_main._flag_error(message) == line
