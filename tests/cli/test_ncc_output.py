"""Tests for what ``ncc -p`` writes to stdout, byte for byte, and for where it is written.

The result of a prompt is often a file (``ncc -p "..." > out.py``), so it is the reply as the
model wrote it: indentation included, written as bytes of UTF-8 whatever the locale, and a
reader that closes the pipe early ends the run quietly.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nanoclaude.agent.loop import Done, LoopState, StopReason
from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    Transcript,
    assistant_text,
    user_text,
)
from nanoclaude.providers.base import ModelError, ModelReply, StopKind, Usage
from nanoclaude.testing.scripted import calls, cut_off, says
from tests.cli.helpers import run_ncc

# --------------------------------------------------------------------------
# The reply as the model wrote it
# --------------------------------------------------------------------------

INDENTED = "    indented = 1\n\n\n"


def test_a_reply_that_begins_with_indentation_keeps_it(ncc_home, project, serve, capsys):
    serve(m=[says(INDENTED)])
    code, out, _ = run_ncc(capsys, "--root", str(project), "-p", "write it")
    # Not "indented = 1\n", which is what stripping the reply gave, and which would have
    # put the first line of a file written with > out.py one level out.
    assert (code, out) == (EXIT_CODES["completed"], INDENTED)


@pytest.mark.parametrize(
    ("reply", "written"),
    [
        ("\n\n  two blank lines first", "\n\n  two blank lines first\n"),
        ("trailing space   ", "trailing space   \n"),
        ("one\n", "one\n"),
        ("\tdef f():\n\t\treturn 1", "\tdef f():\n\t\treturn 1\n"),
    ],
)
def test_nothing_of_the_reply_is_trimmed_and_a_newline_is_added_only_where_there_is_none(
    ncc_home, project, serve, capsys, reply, written
):
    serve(m=[says(reply)])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert out == written


def _done(*messages: Message, text: str) -> Done:
    state = LoopState(Transcript(messages), turn=0, max_turns=40)
    return Done(state, text, StopReason.COMPLETED)


def test_the_text_as_written_is_the_last_assistant_message_and_otherwise_what_the_loop_said():
    said = _done(user_text("hi"), assistant_text("  x\n"), text="x")
    assert ncc_main._as_written(said) == "  x\n"
    # Nothing a loop that finished leaves looks like these two, and the loop's words are
    # better than none if something does.
    assert ncc_main._as_written(_done(user_text("hi"), text="fallback")) == "fallback"
    assert ncc_main._as_written(_done(text="fallback")) == "fallback"


def test_the_json_result_is_the_same_text(ncc_home, project, serve, capsys):
    serve(m=[says(INDENTED)])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go", "--output-format", "json")
    assert json.loads(out)["result"] == INDENTED


def test_a_reply_in_several_blocks_is_its_blocks_one_to_a_line(ncc_home, project, serve, capsys):
    two = ModelReply(
        (TextBlock("first  "), TextBlock("    second")), StopKind.END_TURN, Usage(), "scripted"
    )
    serve(m=[two])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert out == "first  \n    second\n"


def test_it_is_the_last_reply_that_is_the_result_after_the_tools_have_run(
    ncc_home, project, serve, capsys
):
    (project / "a.txt").write_text("x\n")
    serve(
        m=[
            calls("Read", {"path": "a.txt"}, call_id="r1", preamble="  reading it"),
            says("    done = True"),
        ]
    )
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert out == "    done = True\n"


def test_the_words_of_a_refusal_and_of_the_output_limit_keep_their_indentation_too(
    ncc_home, project, serve, capsys
):
    limit = ModelReply((TextBlock("    def f():\n"),), StopKind.MAX_TOKENS, Usage(), "scripted")
    serve(m=[limit])
    code, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert (code, out) == (EXIT_CODES["stopped"], "    def f():\n")


def test_what_the_program_says_of_a_turn_limit_is_not_made_into_a_result(
    ncc_home, project, serve, capsys
):
    (project / "a.txt").write_text("x\n")
    serve(m=[calls("Read", {"path": "a.txt"}, call_id="r1", preamble="  reading")])
    code, out, _ = run_ncc(capsys, "--root", str(project), "--max-turns", "1", "-p", "go")
    assert (code, out) == (EXIT_CODES["stopped"], "")


def test_a_reply_that_was_cut_off_keeps_its_indentation_in_what_is_written(
    ncc_home, project, serve, capsys
):
    serve(m=[cut_off("    indented = 1\n    more")])
    code, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert (code, out) == (EXIT_CODES["provider"], "    indented = 1\n    more\n")


# --------------------------------------------------------------------------
# What a script can do about a reply that was cut off
# --------------------------------------------------------------------------

CAUSE = "the connection to the provider was lost: connection reset by peer"
CONTINUE = 'run: ncc --continue -p "continue"'


@pytest.mark.parametrize("fmt", [[], ["--output-format", "json"]], ids=["text", "json"])
def test_the_advice_for_a_reply_cut_off_is_a_command_a_script_can_run(
    ncc_home, project, serve, capsys, fmt
):
    serve(m=[cut_off("The first half")])
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "q", *fmt)
    assert err == f"error: the reply was cut off ({CAUSE}) — what arrived is kept; {CONTINUE}\n"


def test_the_advice_when_nothing_arrived_is_the_same_command_without_what_is_kept(
    ncc_home, project, serve, capsys
):
    serve(m=[cut_off("")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "q")
    assert (code, out) == (EXIT_CODES["provider"], "")
    assert err == f"error: the reply was cut off ({CAUSE}) — {CONTINUE}\n"


def test_a_cause_that_has_a_dash_in_it_does_not_cut_the_line_in_the_wrong_place(
    ncc_home, project, serve, capsys
):
    serve(m=[cut_off("half", message="the link — the whole of it — dropped")])
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "q")
    assert err == (
        "error: the reply was cut off (the link — the whole of it — dropped) "
        f"— what arrived is kept; {CONTINUE}\n"
    )


def test_an_error_that_was_raised_from_another_is_not_taken_for_a_reply_that_broke_off(
    ncc_home, project, serve, capsys
):
    # The session raises "the conversation does not fit" from the provider's own overflow
    # error, which is a ModelError that holds no reply. It is not advised to continue.
    overflow = ModelError("too long", retryable=False, context_overflow=True)
    serve(m=[overflow])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "q")
    assert (code, out) == (EXIT_CODES["provider"], "")
    assert err.startswith("error: the conversation does not fit the context window of ")
    assert "ncc --continue" not in err


def test_an_ordinary_provider_error_keeps_its_own_words(ncc_home, project, serve, capsys):
    serve(m=[ModelError("the provider is down", retryable=False)])
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "q")
    assert "continue" not in err and err.startswith("error: the provider is down — ")


# --------------------------------------------------------------------------
# Bytes, whatever the locale
# --------------------------------------------------------------------------


def test_the_result_is_utf_8_bytes_and_a_locale_that_cannot_write_it_does_not_matter(
    ncc_home, project, serve, capsys, monkeypatch
):
    ascii_only = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", ascii_only)
    serve(m=[says("café — ✓ naïve")])
    code = ncc_main.main(["--root", str(project), "-p", "go"])
    assert code == EXIT_CODES["completed"]
    assert ascii_only.buffer.getvalue() == "café — ✓ naïve\n".encode()


def test_the_json_and_the_result_arrive_in_the_order_they_were_written(
    ncc_home, project, serve, capsys, monkeypatch
):
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", stream)
    serve(m=[says("café")])
    ncc_main.main(["--root", str(project), "-p", "go", "--output-format", "json"])
    payload = json.loads(stream.buffer.getvalue())
    assert payload["result"] == "café"


def test_a_character_that_cannot_be_written_in_utf_8_is_replaced_and_the_rest_arrives(
    ncc_home, project, serve, capsys
):
    # A lone surrogate is not text. The reply is not lost for it.
    serve(m=[says("before \ud800 after")])
    _, out, _ = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert out == "before ? after\n"


# --------------------------------------------------------------------------
# A reader that has gone
# --------------------------------------------------------------------------


class _Gone(io.RawIOBase):
    """What stdout is when the other end of the pipe has been closed: until the test is over.

    A wrapper that is still holding what it could not write would try again as it is
    collected, and say so; once the test is done the pipe is as good as open.
    """

    def __init__(self) -> None:
        super().__init__()
        self.closed_at_the_other_end = True

    def writable(self) -> bool:
        return True

    def write(self, data: object) -> int:
        if self.closed_at_the_other_end:
            raise BrokenPipeError(32, "Broken pipe")
        return len(data)  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt", [[], ["--output-format", "json"]], ids=["text", "json"])
def test_a_closed_stdout_ends_the_run_quietly_with_141(
    ncc_home, project, serve, capsys, monkeypatch, fmt
):
    pipe = _Gone()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BufferedWriter(pipe), "utf-8"))
    serve(m=[says("a reply nobody will read")])
    try:
        code = ncc_main.main(["--root", str(project), "-p", "go", *fmt])
    finally:
        pipe.closed_at_the_other_end = False
    captured = capsys.readouterr()
    assert code == EXIT_CODES["output_closed"] == 141
    assert captured.err == ""  # not an error: the person that closed it knew


DRIVER = """
import sys
from nanoclaude.agent.router import Router
from nanoclaude.cli import main as ncc
from nanoclaude.testing.scripted import says
from nanoclaude.testing.session import ScriptedClient

client = ScriptedClient([says("a reply nobody will read")])
ncc.Router = lambda config, cache: Router(config, cache, {"m": client})
sys.exit(ncc.main(sys.argv[1:]))
"""


@pytest.mark.parametrize("fmt", [[], ["--output-format", "json"]], ids=["text", "json"])
def test_ncc_piped_into_a_reader_that_has_quit_exits_141_and_says_nothing(ncc_home, project, fmt):
    # The real thing: a pipe whose read end is closed. Python flushes stdout once more as it
    # exits, and unless that is dealt with it prints a complaint of its own to stderr.
    read, write = os.pipe()
    os.close(read)
    env = dict(os.environ, HOME=str(ncc_home), NANOCLAUDE_HOME=str(ncc_home))
    try:
        done = subprocess.run(  # noqa: S603
            [sys.executable, "-c", DRIVER, "--root", str(project), "-p", "go", *fmt],
            stdout=write,
            stderr=subprocess.PIPE,
            env=env,
            cwd=Path(__file__).parents[2],
            timeout=60,
            check=False,
        )
    finally:
        os.close(write)
    assert done.returncode == 141
    assert done.stderr == b""
