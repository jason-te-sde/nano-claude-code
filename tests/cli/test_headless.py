"""Tests for ``ncc -p``: one prompt, one answer, and an exit code a script can read.

Headless mode is the same session as the REPL's with a front end that never asks, so what
is tested here is what that front end owes a script: the answer on stdout and nothing else
there, every complaint on stderr in the form of spec 17.9, and an exit code that says which
of "finished", "stopped early", "typed wrong", "misconfigured" and "the provider failed"
happened. The models are scripted; the configuration, the router and the store are real.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys

import pytest

from nanoclaude.agent.router import Router
from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.main import EXIT_CODES, build_parser, main
from nanoclaude.conversation.store import Store
from nanoclaude.conversation.transcript import TextBlock
from nanoclaude.providers.base import ModelError, ModelReply, StopKind, Usage
from nanoclaude.providers.texttools import MAX_PARSE_RETRIES
from nanoclaude.testing.scripted import cut_off, says
from nanoclaude.testing.session import ScriptedClient
from tests.cli.helpers import run_ncc

BAD_CALL = '<tool name="Read">{"path": </tool>'

#: Everything in a reply that Rich would do something to: a subscript, a closing tag with
#: nothing open, markup that would be styling, and a line longer than the 80 columns Rich
#: wraps at when its output is not a terminal.
AWKWARD = "total = arr[0] + arr[1]\n[/]\n[bold]not bold[/bold]\n" + "x" * 200 + "\nlast line"


# --------------------------------------------------------------------------
# The contract: exit codes and flags
# --------------------------------------------------------------------------


def test_the_exit_codes_are_documented_and_distinct():
    assert len(set(EXIT_CODES.values())) == len(EXIT_CODES)
    assert EXIT_CODES == {
        "completed": 0,
        "stopped": 1,
        "usage": 2,
        "config": 3,
        "provider": 4,
        "interrupted": 130,
    }


def test_role_overrides_parse_into_pairs():
    args = build_parser().parse_args(["-p", "x", "--role", "explore=local", "--role", "plan=big"])
    assert args.role == ["explore=local", "plan=big"]


def test_print_mode_with_no_prompt_is_a_usage_error(capsys):
    code, out, err = run_ncc(capsys, "-p")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert "expected one argument" in err


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_print_mode_with_a_blank_prompt_is_a_usage_error_in_the_form_of_spec_17_9(capsys, prompt):
    code, out, err = run_ncc(capsys, "--print", prompt)
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err.startswith("error: --print needs a prompt — ")
    assert err.count("error:") == 1 and err.endswith("\n") and err.count("\n") == 1


# --------------------------------------------------------------------------
# The answer is written exactly, and only the answer goes to stdout
# --------------------------------------------------------------------------


def test_the_reply_comes_out_byte_for_byte(ncc_home, project, serve, capsys):
    serve(m=[says(AWKWARD)])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "write it")
    assert code == EXIT_CODES["completed"]
    # Not parsed as markup, not wrapped at 80 columns, and a final newline that was not there.
    assert out == AWKWARD + "\n"
    assert err == ""


def test_a_reply_is_not_wrapped_when_piped_to_a_file(ncc_home, project, serve, capfd):
    # The 200-character line is the one that came out as three lines through Rich. capfd
    # watches the real file descriptor, where "> out.py" would put it.
    serve(m=[says("x" * 211)])
    main(["--root", str(project), "-p", "write it"])
    assert capfd.readouterr().out == "x" * 211 + "\n"


@pytest.mark.parametrize(
    ("text", "written"), [("a", "a\n"), ("a\n", "a\n"), ("a\n\n", "a\n\n"), ("", "")]
)
def test_the_newline_is_added_only_where_there_is_none(capsys, text, written):
    ncc_main._write_result(text)
    assert capsys.readouterr().out == written


def test_a_terminal_is_not_sent_the_escape_sequences_a_reply_holds(
    ncc_home, project, serve, capsys, monkeypatch
):
    # Piped to a file the reply is what the model wrote. Printed on a terminal it is text
    # the person is shown, and a terminal acts on the controls inside it.
    hostile = "fine\x1b]0;owned\x07\x1b[2Jstill fine"
    serve(m=[says(hostile), says(hostile)])
    _, piped, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert piped == hostile + "\n"
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    _, shown, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert shown == "finestill fine\n"


# --------------------------------------------------------------------------
# --output-format json
# --------------------------------------------------------------------------


def test_json_output_has_a_stable_shape(ncc_home, project, serve, capsys):
    """CI consumes this. Adding a key is fine; renaming one is a breaking change."""
    serve(m=[says("done", input_tokens=1000, output_tokens=2000)])
    code, out, err = run_ncc(capsys, "-p", "hi", "--output-format", "json", "--root", str(project))
    payload = json.loads(out)
    assert code == EXIT_CODES["completed"]
    assert err == ""
    assert set(payload) >= {
        "session_id",
        "result",
        "stop_reason",
        "turns",
        "usage",
        "cost_usd",
        "is_error",
    }
    assert re.fullmatch(r"[0-9a-f]{12}", payload["session_id"])
    assert payload["result"] == "done"
    assert payload["stop_reason"] == "completed"
    assert payload["is_error"] is False
    assert payload["usage"] == {"input_tokens": 1000, "output_tokens": 2000, "cache_read_tokens": 0}
    assert isinstance(payload["cost_usd"], float) and payload["cost_usd"] > 0
    assert isinstance(payload["turns"], int)
    store = Store(ncc_home / ".nanoclaude" / "sessions.db")
    store.open()
    assert store.session_row(payload["session_id"]) is not None


def test_json_carries_the_reply_exactly_too(ncc_home, project, serve, capsys):
    serve(m=[says(AWKWARD)])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go", "--output-format", "json")
    assert json.loads(out)["result"] == AWKWARD


def test_a_cost_that_is_not_known_is_null_and_not_zero(ncc_home, project, serve, capsys):
    # A model with no price: a zero would read as free.
    config = ncc_home / ".nanoclaude" / "config.toml"
    config.write_text(
        '[models.m]\nadapter = "anthropic"\nmodel = "a-model-nobody-priced"\n'
        "context_window = 100000\n"
    )
    serve(m=[says("ok")])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go", "--output-format", "json")
    assert json.loads(out)["cost_usd"] is None


# --------------------------------------------------------------------------
# A run that stops early: exit 1, and an error that says why
# --------------------------------------------------------------------------


def unsuitable() -> list[ModelReply]:
    return [says(f"{BAD_CALL} attempt {n}") for n in range(MAX_PARSE_RETRIES + 1)]


def test_a_model_without_working_tool_calls_is_a_stop_and_not_a_completion(
    ncc_home, project, serve, capsys
):
    serve(textual=unsuitable())
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--model", "textual", "-p", "read a.py"
    )
    assert code == EXIT_CODES["stopped"]
    # What went wrong is the program's own sentence and not the model's words: it is not
    # the result, so it is not on stdout.
    assert out == ""
    assert err == (
        "error: claude-sonnet-5 did not produce a valid tool call in "
        f"{MAX_PARSE_RETRIES + 1} attempts — choose a model with native tool calling "
        "(--model)\n"
    )


def test_a_model_without_working_tool_calls_is_an_error_in_the_json(
    ncc_home, project, serve, capsys
):
    serve(textual=unsuitable())
    code, out, err = run_ncc(
        capsys,
        "--root",
        str(project),
        "--model",
        "textual",
        "-p",
        "read a.py",
        "--output-format",
        "json",
    )
    payload = json.loads(out)
    assert code == EXIT_CODES["stopped"]
    assert payload["stop_reason"] == "model_unsuitable"
    assert payload["is_error"] is True
    assert "choose a model with native tool calling" in payload["result"]
    assert err.startswith("error: claude-sonnet-5 did not produce a valid tool call")


def declined() -> ModelReply:
    return ModelReply((TextBlock("I will not do that."),), StopKind.REFUSAL, Usage(), "scripted")


def too_long() -> ModelReply:
    return ModelReply(
        (TextBlock("def f():\n    return"),), StopKind.MAX_TOKENS, Usage(), "scripted"
    )


def test_a_refusal_keeps_the_models_words_on_stdout_and_says_why_it_stopped(
    ncc_home, project, serve, capsys
):
    serve(m=[declined()])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "do it")
    assert code == EXIT_CODES["stopped"]
    assert out == "I will not do that.\n"
    assert err == (
        "error: the model declined to continue — rephrase the request, or try another "
        "model with --model\n"
    )


def test_a_reply_cut_off_at_the_output_limit_keeps_its_text_and_says_so(
    ncc_home, project, serve, capsys
):
    serve(m=[too_long()])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "write f")
    assert code == EXIT_CODES["stopped"]
    assert out == "def f():\n    return\n"
    assert err.startswith("error: the reply was cut off at the model's output limit — ")


# --------------------------------------------------------------------------
# A stream that broke midway: what arrived is kept, and the run still failed
# --------------------------------------------------------------------------


def test_a_reply_cut_off_midway_writes_what_arrived_and_fails_with_the_provider_code(
    ncc_home, project, serve, capsys
):
    serve(m=[cut_off("The first half of an ans")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "question")
    assert code == EXIT_CODES["provider"]
    # The kept text and nothing of the session's marker, which is for the model.
    assert out == "The first half of an ans\n"
    assert err == (
        "error: the reply was cut off (the connection to the provider was lost: connection "
        "reset by peer) — what arrived is kept; ask the model to continue\n"
    )


def test_a_reply_cut_off_midway_is_the_result_of_the_json_and_an_error(
    ncc_home, project, serve, capsys
):
    serve(m=[cut_off("The first half of an ans", input_tokens=120, output_tokens=7)])
    code, out, err = run_ncc(
        capsys, "--root", str(project), "-p", "question", "--output-format", "json"
    )
    payload = json.loads(out)
    assert code == EXIT_CODES["provider"]
    assert payload["result"] == "The first half of an ans"
    assert payload["is_error"] is True
    assert payload["stop_reason"] == "cut_off"
    assert payload["usage"]["input_tokens"] == 120 and payload["usage"]["output_tokens"] == 7
    assert err.startswith("error: the reply was cut off")


def test_a_reply_cut_off_before_any_text_has_nothing_to_write(ncc_home, project, serve, capsys):
    serve(m=[cut_off("")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "question")
    assert code == EXIT_CODES["provider"]
    assert out == ""
    assert err.startswith("error: the reply was cut off") and "what arrived is kept" not in err


def test_a_reply_cut_off_before_any_text_has_no_json_either(ncc_home, project, serve, capsys):
    serve(m=[cut_off("")])
    code, out, _ = run_ncc(capsys, "--root", str(project), "-p", "q", "--output-format", "json")
    assert code == EXIT_CODES["provider"]
    assert out == ""


# --------------------------------------------------------------------------
# Every error goes to stderr, whole, and stdout stays empty
# --------------------------------------------------------------------------


def test_missing_configuration_exits_three_and_points_at_init(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(tmp_path))
    code, out, err = run_ncc(capsys, "--root", str(tmp_path), "-p", "hello")
    assert code == EXIT_CODES["config"]
    assert out == ""
    assert "ncc init" in err and err.startswith("error: no configuration found — ")


def test_a_config_error_naming_a_section_reaches_stderr_verbatim(ncc_home, project, capsys):
    # [models.<alias>] is what Rich took for markup and dropped, and the path that follows it
    # is long enough to be wrapped at 80 columns.
    config = ncc_home / ".nanoclaude" / "config.toml"
    config.write_text("[limits]\nmax_turns = 3\n")
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["config"]
    assert out == ""
    assert err == (
        f"error: no [models.<alias>] sections found in {config} — add one, or run: ncc init\n"
    )


def test_a_model_that_is_not_defined_is_a_config_error_even_with_markup_in_its_name(
    ncc_home, project, capsys
):
    code, out, err = run_ncc(capsys, "--root", str(project), "--model", "bad[/]", "-p", "hello")
    assert code == EXIT_CODES["config"]
    assert out == ""
    assert "which is not defined — define [models.bad[/]] or" in err
    assert err.count("\n") == 1


def test_a_provider_error_is_reported_on_stderr_and_stdout_stays_empty(
    ncc_home, project, serve, capsys
):
    serve(m=[ModelError("the provider refused the key", retryable=False, status=401)])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["provider"]
    assert out == ""
    # Spec 17.9: what happened, then what to do.
    assert err == (
        "error: the provider refused the key — try again, or choose another model with --model\n"
    )


def test_a_provider_error_that_already_says_what_to_do_is_left_as_it_is(
    ncc_home, project, serve, capsys
):
    serve(m=[ModelError("no key — set ANTHROPIC_API_KEY", retryable=False)])
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert err == "error: no key — set ANTHROPIC_API_KEY\n"


def test_a_provider_error_is_printed_whole_whatever_it_holds(ncc_home, project, serve, capsys):
    awkward = "bad [/] and [bold]x[/bold] " + "y" * 200 + " \x1b]0;t\x07 end"
    serve(m=[ModelError(awkward, retryable=False)])
    _, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert out == ""
    assert err.count("\n") == 1  # not wrapped
    assert "bad [/] and [bold]x[/bold] " + "y" * 200 in err  # not markup
    assert "\x1b" not in err and "\x07" not in err  # nothing for a terminal to act on


def test_a_missing_key_is_a_provider_error_that_names_the_variable(
    ncc_home, project, capsys, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["provider"]
    assert out == ""
    assert "ANTHROPIC_API_KEY" in err


def test_a_model_window_too_small_for_the_prompt_is_a_configuration_error(
    ncc_home, project, capsys
):
    config = ncc_home / ".nanoclaude" / "config.toml"
    config.write_text(
        '[models.m]\nadapter = "anthropic"\nmodel = "claude-sonnet-5"\ncontext_window = 1000\n'
    )
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["config"]
    assert out == ""
    assert err.startswith("error: the system prompt and tools alone need about ")
    assert "(--model)" in err


def test_no_color_leaves_an_error_line_without_colour(ncc_home, project, capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    _, _, coloured = run_ncc(capsys, "--root", str(project), "--model", "ghost", "-p", "hello")
    assert "\x1b[" in coloured
    _, _, plain = run_ncc(
        capsys, "--no-color", "--root", str(project), "--model", "ghost", "-p", "hello"
    )
    assert "\x1b[" not in plain and plain.startswith("error: ")


# --------------------------------------------------------------------------
# Ctrl+C
# --------------------------------------------------------------------------


def test_a_run_that_is_interrupted_exits_130_and_writes_nothing_to_stdout(
    ncc_home, project, capsys, monkeypatch
):
    async def interrupted(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(ncc_main, "_run_headless", interrupted)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["interrupted"]
    assert out == ""
    assert err == "interrupted\n"


class Interrupting(ScriptedClient):
    """A model that is waiting for its answer when the person presses Ctrl+C."""

    async def complete(self, _request, *, on_text=None):  # noqa: ARG002
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(10)
        raise AssertionError("the interrupt did not end the run")


def test_ctrl_c_during_a_request_ends_the_run_and_closes_the_session(
    ncc_home, project, capsys, monkeypatch
):
    clients = {"m": Interrupting([])}
    monkeypatch.setattr(ncc_main, "Router", lambda config, cache: Router(config, cache, clients))
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["interrupted"]
    assert out == "" and err == "interrupted\n"
    assert clients["m"].closed  # the session was closed on the way out


# --------------------------------------------------------------------------
# Where ncc keeps its files, and how a bare provider message becomes a line of 17.9
# --------------------------------------------------------------------------


def test_ncc_keeps_its_files_in_the_home_directory_unless_told_to_use_another(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NANOCLAUDE_HOME", raising=False)
    assert ncc_main.nanoclaude_home() == tmp_path / "home"
    monkeypatch.setenv("NANOCLAUDE_HOME", str(tmp_path / "elsewhere"))
    assert ncc_main.nanoclaude_home() == tmp_path / "elsewhere"


@pytest.mark.parametrize(
    ("message", "line"),
    [
        ("the provider is down.", "the provider is down — try again"),
        ("two\nlines  of   text", "two lines of text — try again"),
        ("", "the model request failed — try again"),
        ("already — says what to do", "already — says what to do"),
    ],
)
def test_a_message_becomes_what_happened_a_dash_and_what_to_do(message, line):
    assert ncc_main._with_advice(message, "try again") == line
