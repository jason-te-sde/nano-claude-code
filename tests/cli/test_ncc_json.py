"""Tests for ``--output-format json`` as a contract: every run ends in one object on stdout.

A script that asks for JSON parses stdout, and an empty stdout is not JSON: it was what a usage
error, a configuration error or a provider error left, so the script had to read the exit code
and stderr to learn what its parser had choked on. Now every run prints exactly one object, and
a failure's object says that it is one: ``is_error``, the line that went to stderr as ``error``
(without its ``error: `` prefix), and the process's exit code as ``exit_code``, beside the keys
a finished run has.

What is not a run prints what it always did: ``--help`` and ``--version`` are text, whatever the
format, and a reader that has closed stdout is not written to (exit 141).

Before the format is known, which is when the arguments cannot be parsed, it is read from what
was typed: the arguments are scanned for ``--output-format json`` (or ``=json``), the last one
winning as it does in the parser, and nothing else about them is trusted. Where the format was
not given, or was not one that exists, the mistake is the line on stderr and nothing on stdout.
"""

from __future__ import annotations

import io
import json
import re
import sys
from typing import Any

import pytest

from nanoclaude.cli.main import EXIT_CODES, main
from nanoclaude.providers.base import ModelError
from nanoclaude.testing.scripted import calls, says
from tests.cli.helpers import INTERRUPTED, run_ncc

#: What a failure's object holds that a run which got as far as a session may not have.
NOTHING_SPENT = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}


def one_object(out: str) -> dict[str, Any]:
    """The whole of stdout as one JSON object: a second one after it would not parse."""
    payload = json.loads(out)
    assert isinstance(payload, dict)
    return payload


def assert_failure(out: str, err: str, code: int) -> dict[str, Any]:
    """stdout is one object that says it is a failure, and says what stderr says."""
    payload = one_object(out)
    assert payload["is_error"] is True
    assert payload["exit_code"] == code
    assert err == f"error: {payload['error']}\n"
    assert payload["stop_reason"] == "error"
    return payload


# --------------------------------------------------------------------------
# A run that finished has the same two keys, empty
# --------------------------------------------------------------------------


def test_a_finished_run_says_it_was_not_an_error_and_exited_zero(ncc_home, project, serve, capsys):
    serve(m=[says("done")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "go", "--output-format", "json")
    payload = one_object(out)
    assert (code, err) == (0, "")
    assert payload["error"] is None and payload["exit_code"] == 0 and payload["is_error"] is False


def test_a_run_that_stopped_early_carries_the_stop_line_and_its_exit_code(
    ncc_home, project, serve, capsys
):
    (project / "a.txt").write_text("x\n")
    serve(m=[calls("Read", {"path": "a.txt"}, call_id="r1", preamble="reading")])
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--max-turns", "1", "-p", "go", "--output-format", "json"
    )
    payload = one_object(out)
    assert code == payload["exit_code"] == EXIT_CODES["stopped"]
    assert payload["is_error"] is True and payload["stop_reason"] == "turn_limit"
    assert err == f"error: {payload['error']}\n"
    assert payload["error"].startswith("the turn limit of 1 was reached")


# --------------------------------------------------------------------------
# A run that failed: an object, whatever the failure was
# --------------------------------------------------------------------------


def test_a_usage_error_found_after_parsing_is_an_object(ncc_home, project, capsys):
    code, out, err = run_ncc(capsys, "--root", str(project), "--output-format", "json")
    payload = assert_failure(out, err, EXIT_CODES["usage"])
    assert code == EXIT_CODES["usage"]
    assert payload == {
        "session_id": None,
        "result": "",
        "stop_reason": "error",
        "turns": 0,
        "usage": NOTHING_SPENT,
        "cost_usd": 0.0,
        "is_error": True,
        "error": (
            "--output-format needs -p — it shapes the answer to one prompt; pass a prompt with -p"
        ),
        "exit_code": 2,
    }


def test_the_refusal_of_a_mode_is_an_object(ncc_home, project, capsys):
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--mode", "bypass", "-p", "x", "--output-format", "json"
    )
    payload = assert_failure(out, err, EXIT_CODES["usage"])
    assert code == 2 and payload["error"].startswith("bypass is turned on by")


def test_a_configuration_error_is_an_object(ncc_home, project, capsys):
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--model", "ghost", "-p", "x", "--output-format", "json"
    )
    payload = assert_failure(out, err, EXIT_CODES["config"])
    assert code == 3 and "which is not defined" in payload["error"]


def test_a_missing_key_is_an_object_and_the_session_never_started(
    ncc_home, project, capsys, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "x", "--output-format", "json")
    payload = assert_failure(out, err, EXIT_CODES["config"])
    assert code == 3 and payload["error"].startswith("role 'main' uses model 'm'")
    assert payload["session_id"] is None and payload["usage"] == NOTHING_SPENT


def test_a_provider_error_is_an_object(ncc_home, project, serve, capsys):
    serve(m=[ModelError("the provider is down", retryable=False)])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "x", "--output-format", "json")
    payload = assert_failure(out, err, EXIT_CODES["provider"])
    assert code == 4 and payload["error"].startswith("the provider is down")


def test_an_error_nobody_mapped_is_an_object_too(ncc_home, project, serve, capsys):
    from nanoclaude.providers.base import ModelReply, StopKind, Usage

    serve(m=[ModelReply((), StopKind.END_TURN, Usage(), "scripted")])  # the loop refuses it
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "x", "--output-format", "json")
    payload = assert_failure(out, err, EXIT_CODES["stopped"])
    assert code == 1 and payload["error"].startswith("unexpected error (LoopError")


def test_a_run_that_was_interrupted_is_an_object(ncc_home, project, capsys, monkeypatch):
    from nanoclaude.cli import main as ncc_main

    async def interrupted(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(ncc_main, "_run_headless", interrupted)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "x", "--output-format", "json")
    assert_failure(out, err, EXIT_CODES["interrupted"])
    assert code == 130 and err == INTERRUPTED


def test_the_error_of_a_stop_that_brought_its_own_prefix_is_not_given_the_prefix_twice(
    ncc_home, project, serve, capsys
):
    # The loop's closing line for a model that cannot call tools is already "error: ...".
    from nanoclaude.providers.texttools import MAX_PARSE_RETRIES

    bad = '<tool name="Read">{"path": </tool>'
    serve(m=[], textual=[says(f"{bad} {n}") for n in range(MAX_PARSE_RETRIES + 1)])
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--model", "textual", "-p", "x", "--output-format", "json"
    )
    payload = one_object(out)
    assert code == payload["exit_code"] == EXIT_CODES["stopped"]
    assert payload["error"].startswith("claude-sonnet-5 did not produce a valid tool call")
    assert err == f"error: {payload['error']}\n"


def test_a_reply_cut_off_before_any_text_is_an_object_and_there_is_no_result(
    ncc_home, project, serve, capsys
):
    from nanoclaude.testing.scripted import cut_off

    serve(m=[cut_off("")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "q", "--output-format", "json")
    payload = assert_failure(out, err, EXIT_CODES["provider"])
    assert (
        code == 4 and payload["result"] == "" and payload["error"].startswith("the reply was cut")
    )


def test_what_was_spent_before_the_failure_is_reported_and_not_zero(
    ncc_home, project, serve, capsys
):
    # The first request was answered and paid for; the second failed. A script that reads
    # cost_usd to keep a budget must be told about the first.
    (project / "a.txt").write_text("x\n")
    serve(
        m=[
            calls("Read", {"path": "a.txt"}, call_id="r1", input_tokens=1000, output_tokens=50),
            ModelError("the provider is down", retryable=False),
        ]
    )
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "x", "--output-format", "json")
    payload = assert_failure(out, err, EXIT_CODES["provider"])
    assert code == 4
    assert payload["usage"] == {"input_tokens": 1000, "output_tokens": 50, "cache_read_tokens": 0}
    assert isinstance(payload["cost_usd"], float) and payload["cost_usd"] > 0
    assert re.fullmatch(r"[0-9a-f]{12}", payload["session_id"])
    assert payload["turns"] >= 1


# --------------------------------------------------------------------------
# Before the format is known
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--output-format", "json", "--bogus", "-p", "x"],
        ["--output-format=json", "--bogus", "-p", "x"],
        ["--bogus", "-p", "x", "--output-format", "json"],
        ["--bogus", "-p", "x", "--output-format=json"],
        ["--output-format", "text", "--output-format", "json", "--bogus", "-p", "x"],
    ],
    ids=["after", "equals", "before", "equals, before", "the last one wins"],
)
def test_a_mistake_in_the_flags_is_an_object_when_the_arguments_asked_for_json(capsys, argv):
    code, out, err = run_ncc(capsys, *argv)
    payload = assert_failure(out, err, EXIT_CODES["usage"])
    assert code == 2 and payload["error"].startswith("unrecognized argument --bogus")
    assert payload["session_id"] is None and payload["usage"] == NOTHING_SPENT


@pytest.mark.parametrize(
    "argv",
    [
        ["--bogus", "-p", "x"],
        ["--output-format", "text", "--bogus", "-p", "x"],
        ["--output-format", "xml", "-p", "x"],
        ["--output-format"],
        ["--output-format", "json", "--output-format", "text", "--bogus", "-p", "x"],
        ["--bogus", "--", "--output-format", "json"],
        ["-p", "--output-format=json-ish"],
    ],
    ids=[
        "none",
        "text",
        "not a format",
        "no value",
        "the last one is text",
        "after --",
        "json-ish",
    ],
)
def test_a_mistake_in_the_flags_is_the_line_on_stderr_alone_when_json_was_not_asked_for(
    capsys, argv
):
    code, out, err = run_ncc(capsys, *argv)
    assert (code, out) == (EXIT_CODES["usage"], "")
    assert err.startswith("error: ") and err.count("\n") == 1


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_help_and_version_are_text_whatever_the_format(capsys, flag):
    code, out, err = run_ncc(capsys, "--output-format", "json", flag)
    assert (code, err) == (0, "")
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_a_reader_that_has_closed_stdout_is_not_written_to_when_the_run_failed_either(
    ncc_home, project, capsys, monkeypatch
):
    class Closed(io.StringIO):
        def write(self, _text: str) -> int:
            raise BrokenPipeError

    monkeypatch.setattr(sys, "stdout", Closed())
    code = main(["--root", str(project), "--model", "ghost", "-p", "x", "--output-format", "json"])
    # What the shell calls a process that was killed by SIGPIPE, and nothing more is said.
    assert code == EXIT_CODES["output_closed"] == 141
    assert capsys.readouterr().err == ""
