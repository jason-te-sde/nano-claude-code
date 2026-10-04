"""Tests for nanoclaude.cli.repl: the loop a person lives in.

A prompt is answered from a script (``ScriptedPrompter``), so these tests run a whole
conversation with no terminal: the lines typed, Ctrl+C and Ctrl+D are the script's
entries. Time is a ``FakeClock`` that moves when the test says so, and a turn that has to
be interrupted blocks on an event, so nothing here waits for time to pass.
"""

from __future__ import annotations

import asyncio
import functools
import signal
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from nanoclaude.agent.loop import Done, LoopError
from nanoclaude.cli import repl
from nanoclaude.cli.commands import COMMANDS
from nanoclaude.cli.prompt import new_prompter
from nanoclaude.cli.render import ConsoleUI
from nanoclaude.cli.repl import PROMPT, error_message, parse_command, run_repl
from nanoclaude.conversation.budget import ContextTooSmallError
from nanoclaude.conversation.transcript import TextBlock, TranscriptError
from nanoclaude.providers.base import ModelError, ModelReply, ModelRequest, StopKind, Usage
from nanoclaude.providers.capabilities import Capabilities
from nanoclaude.providers.texttools import MAX_PARSE_RETRIES
from nanoclaude.testing.scripted import calls, says
from nanoclaude.testing.session import ScriptedSession, build_session
from tests.cli.helpers import FakeClock, ScriptedPrompter, plain_console


class Harness:
    """A REPL, the session it drives, and the screen it writes to.

    ``lines`` are what the person types, in order: a string is a line, an exception is a
    key (``KeyboardInterrupt`` for Ctrl+C). When they run out the person presses Ctrl+D.
    One script feeds the prompt and the confirmation questions alike, as one keyboard does.
    """

    def __init__(
        self,
        root: Path,
        script: list[Any],
        *lines: str | BaseException,
        width: int = 100,
        **options: Any,
    ) -> None:
        self.console, self._buffer = plain_console(width)
        self.prompter = ScriptedPrompter(*lines)
        self.clock = FakeClock()
        self.session: ScriptedSession = build_session(
            root, script, ui=ConsoleUI(self.console, prompter=self.prompter), **options
        )

    @property
    def text(self) -> str:
        return self._buffer.getvalue()

    def start(self) -> asyncio.Task[int]:
        return asyncio.create_task(
            run_repl(self.session, self.console, prompt=self.prompter, clock=self.clock)
        )

    async def run(self) -> int:
        return await asyncio.wait_for(self.start(), timeout=10)


class Sigint:
    """The SIGINT handler the REPL installs on the loop, so a test can press Ctrl+C.

    A real signal would also reach pytest. This records the handler as it is installed and
    calls it, which is what the loop does when the signal arrives.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop = asyncio.get_running_loop()
        self._handler: Callable[[], None] | None = None
        self.installed = 0
        self.removed = 0
        add, remove = loop.add_signal_handler, loop.remove_signal_handler

        def spy_add(sig: int, callback: Callable[..., Any], *args: Any) -> None:
            if sig == signal.SIGINT:
                self._handler = functools.partial(callback, *args)
                self.installed += 1
            add(sig, callback, *args)

        def spy_remove(sig: int) -> bool:
            if sig == signal.SIGINT:
                self._handler = None
                self.removed += 1
            return remove(sig)

        monkeypatch.setattr(loop, "add_signal_handler", spy_add)
        monkeypatch.setattr(loop, "remove_signal_handler", spy_remove)

    def press(self) -> None:
        assert self._handler is not None, "the REPL has no SIGINT handler installed right now"
        self._handler()


def block_the_answer(
    session: ScriptedSession, monkeypatch: pytest.MonkeyPatch, nth: int = 1
) -> asyncio.Event:
    """Make the model's ``nth`` answer never come, and say when it was asked for.

    The answers before it come from the script as usual, and so do the ones after.
    """
    started = asyncio.Event()
    answer = session.model.complete
    asked = 0

    async def complete(request: ModelRequest) -> ModelReply:
        nonlocal asked
        asked += 1
        if asked == nth:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("the model was allowed to answer")
        return await answer(request)

    monkeypatch.setattr(session.model, "complete", complete)
    return started


async def reached(event: asyncio.Event) -> None:
    """Wait for something the code under test must do, and fail if it never does."""
    await asyncio.wait_for(event.wait(), timeout=5)


def sent_messages(session: ScriptedSession, request: int = 0) -> list[str]:
    return [m.text() for m in session.model.requests[request].transcript.messages]


# --------------------------------------------------------------------------
# Reading and answering
# --------------------------------------------------------------------------


async def test_a_prompt_is_answered_and_the_reply_shown(tmp_repo):
    harness = Harness(tmp_repo, [says("hi there")], "hello")
    assert await harness.run() == 0
    assert "hi there" in harness.text
    assert sent_messages(harness.session) == ["hello"]


async def test_the_banner_names_the_main_model_and_how_to_get_help_and_leave(tmp_repo):
    harness = Harness(tmp_repo, [])
    await harness.run()
    first = harness.text.splitlines()[0]
    assert "claude-sonnet-5" in first and "/help" in first and "Ctrl+D" in first


async def test_ctrl_d_leaves_and_says_goodbye(tmp_repo):
    harness = Harness(tmp_repo, [])
    assert await harness.run() == 0
    assert harness.text.splitlines()[-1] == "bye"


async def test_a_blank_line_asks_nobody_anything(tmp_repo):
    harness = Harness(tmp_repo, [says("answer")], "", "   ", "hello")
    await harness.run()
    assert len(harness.session.model.requests) == 1


async def test_exit_leaves_without_reading_another_line(tmp_repo):
    harness = Harness(tmp_repo, [], "/exit", "never read")
    assert await harness.run() == 0
    assert len(harness.prompter.asked) == 1


async def test_the_indentation_of_a_pasted_block_reaches_the_model_intact(tmp_repo):
    # The first line of a pasted function is indented relative to the ones below it, and
    # stripping the whole text would take that indentation off the one line.
    pasted = "\n\n    if x:\n        return 1  \n\n"
    harness = Harness(tmp_repo, [says("ok")], pasted)
    await harness.run()
    assert sent_messages(harness.session) == ["    if x:\n        return 1"]


async def test_a_command_may_be_typed_with_spaces_in_front_of_it(tmp_repo):
    harness = Harness(tmp_repo, [], "   /exit  ", "never read")
    assert await harness.run() == 0
    assert len(harness.prompter.asked) == 1


async def test_a_prompt_of_several_lines_reaches_the_model_whole(tmp_repo):
    harness = Harness(tmp_repo, [says("ok")], "fix this:\n    x = 1\nplease")
    await harness.run()
    assert sent_messages(harness.session) == ["fix this:\n    x = 1\nplease"]


async def test_a_slash_command_is_run_and_then_the_next_line_is_read(tmp_repo):
    harness = Harness(tmp_repo, [says("answer")], "/tools", "hello", width=200)
    await harness.run()
    assert "Read (read-only)" in harness.text and "answer" in harness.text


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        ("/help", ("help", [])),
        ("/model  c ", ("model", ["c"])),
        ("/compact keep the auth module", ("compact", ["keep", "the", "auth", "module"])),
        ("/Unknown-thing x", ("Unknown-thing", ["x"])),
        ("/etc/hosts has a bug", None),
        ("/", None),
        ("//x", None),
        ("/ x", None),
        ("hello /help", None),
        ("", None),
    ],
)
def test_only_a_word_after_a_slash_is_a_command(text, parsed):
    # A prompt that begins with a path is a prompt.
    assert parse_command(text) == parsed


async def test_a_prompt_that_begins_with_a_path_goes_to_the_model(tmp_repo):
    harness = Harness(tmp_repo, [says("looking")], "/etc/hosts has a typo")
    await harness.run()
    assert "unknown command" not in harness.text
    assert sent_messages(harness.session) == ["/etc/hosts has a typo"]


# --------------------------------------------------------------------------
# Every prompt goes through follow_up
# --------------------------------------------------------------------------


async def test_the_first_prompt_of_a_resumed_session_sees_the_history_that_was_stored(tmp_repo):
    # run() starts a conversation, and on a resumed session that would archive the one
    # just loaded. follow_up() continues it, and starts one when nothing has been said.
    earlier = build_session(tmp_repo, [says("first answer")])
    await earlier.run("first question")
    harness = Harness(
        tmp_repo,
        [says("second answer")],
        "second question",
        session_id=earlier.session_id,
        resume=True,
    )
    await harness.run()
    assert sent_messages(harness.session) == ["first question", "first answer", "second question"]
    assert harness.session.store is not None
    archived = harness.session.store.db.execute(
        "SELECT COUNT(*) FROM messages_archive WHERE session_id = ?", (earlier.session_id,)
    ).fetchone()[0]
    assert archived == 0


async def test_every_prompt_of_a_conversation_continues_it(tmp_repo):
    harness = Harness(tmp_repo, [says("one"), says("two")], "first", "second")
    await harness.run()
    assert sent_messages(harness.session, 1) == ["first", "one", "second"]


# --------------------------------------------------------------------------
# A failed command or turn does not end the REPL
# --------------------------------------------------------------------------


async def test_a_compaction_that_cannot_be_summarised_is_reported_and_the_repl_goes_on(tmp_repo):
    harness = Harness(
        tmp_repo,
        [says(f"answer {n} " + "word " * 400) for n in range(5)],
        "q1",
        "q2",
        "q3",
        "q4",
        "/compact",
        "q5",
        compact_script=[ModelError("summary refused", retryable=False, status=400)],
        keep_recent_turns=1,
        width=300,
    )
    assert await harness.run() == 0
    lines = harness.text.splitlines()
    [failure] = [line for line in lines if line.startswith("error:")]
    assert "could not summarise the conversation: summary refused" in failure
    assert " — " in failure  # what happened, then what to do
    # The prompt after it was answered, by the same session.
    assert harness.text.index("error:") < harness.text.index("answer 4 ")
    # And the history is where it was: nothing was thrown away to carry on.
    assert len(harness.session.state.transcript.messages) == 10


FAILURES: list[Exception] = [
    ModelError("overloaded", retryable=True, status=529),
    ContextTooSmallError("the system prompt and tools alone need about 9000 tokens"),
    LoopError("cannot resume while tool calls are unanswered"),
    TranscriptError("message 3 has no blocks"),
    sqlite3.OperationalError("database is locked"),
    sqlite3.IntegrityError("UNIQUE constraint failed: messages.seq"),
    RuntimeError("boom"),
]


@pytest.mark.parametrize("failure", FAILURES, ids=lambda e: type(e).__name__)
async def test_a_turn_that_fails_is_reported_and_the_next_prompt_is_answered(
    tmp_repo, monkeypatch, failure
):
    harness = Harness(tmp_repo, [says("the answer")], "first", "second", width=300)
    follow_up = harness.session.follow_up
    attempts = 0

    async def flaky(prompt: str) -> Done:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise failure
        return await follow_up(prompt)

    monkeypatch.setattr(harness.session, "follow_up", flaky)
    assert await harness.run() == 0
    [reported] = [ln for ln in harness.text.splitlines() if ln.startswith("error:")]
    assert " — " in reported
    assert "the answer" in harness.text


@pytest.mark.parametrize("failure", FAILURES, ids=lambda e: type(e).__name__)
async def test_a_command_that_fails_is_reported_and_the_next_prompt_is_answered(
    tmp_repo, monkeypatch, failure
):
    harness = Harness(tmp_repo, [says("the answer")], "/status", "hello", width=300)

    async def broken() -> tuple[int, int]:
        raise failure

    monkeypatch.setattr(harness.session, "context_usage", broken)
    assert await harness.run() == 0
    [reported] = [ln for ln in harness.text.splitlines() if ln.startswith("error:")]
    assert " — " in reported
    assert "the answer" in harness.text


@pytest.mark.parametrize(
    ("failure", "line"),
    [
        (
            ModelError("overloaded"),
            "overloaded — try again, or switch models with /model",
        ),
        (
            ModelError("rejected your credentials (HTTP 401): bad key — check the key"),
            "rejected your credentials (HTTP 401): bad key — check the key",
        ),
        (
            ContextTooSmallError("the prompt alone needs 9000 tokens. Use a larger window."),
            (
                "the prompt alone needs 9000 tokens. Use a larger window"
                " — /model switches to a model with a larger window"
            ),
        ),
        (
            LoopError("model returned an empty reply"),
            (
                "the conversation is in an unexpected state (model returned an empty reply)"
                " — /clear starts it over"
            ),
        ),
        (
            TranscriptError("message 3 has no blocks"),
            (
                "the conversation is in an unexpected state (message 3 has no blocks)"
                " — /clear starts it over"
            ),
        ),
        (
            sqlite3.OperationalError("database is locked"),
            (
                "the session store failed (database is locked) — check that the disk has room "
                "and the database file is writable, then try again"
            ),
        ),
        (
            RuntimeError("boom"),
            "RuntimeError: boom — this is a bug; /clear starts the conversation over",
        ),
        (RuntimeError(""), "RuntimeError — this is a bug; /clear starts the conversation over"),
        (ModelError(""), "ModelError — try again, or switch models with /model"),
    ],
)
def test_a_failure_is_described_in_the_form_the_spec_gives_for_a_person(failure, line):
    assert error_message(failure) == line


async def test_what_a_failure_quotes_is_data_and_cannot_act_on_the_terminal(tmp_repo, monkeypatch):
    harness = Harness(tmp_repo, [], "hello", width=300)

    async def boom(prompt: str) -> Done:
        raise ModelError("bad [/] value \x1b[2J here")

    monkeypatch.setattr(harness.session, "follow_up", boom)
    await harness.run()
    assert "error: bad [/] value  here" in harness.text and "\x1b" not in harness.text


# --------------------------------------------------------------------------
# Why a prompt stopped
# --------------------------------------------------------------------------


async def test_a_prompt_that_ran_out_of_turns_says_so(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    harness = Harness(tmp_repo, [calls("Read", {"path": "a.py"}, call_id="r1")], "go", max_turns=1)
    await harness.run()
    assert "Stopped after 1 turn without finishing the task." in harness.text


async def test_a_model_that_cannot_make_tool_calls_is_named_and_the_fix_given(tmp_repo):
    text_only = Capabilities(
        native_tools=False,
        parallel_tools=False,
        cache="none",
        context_window=32_000,
        max_output=4_000,
    )
    bad = '<tool name="Read">{"path": </tool>'
    harness = Harness(
        tmp_repo,
        [says(bad) for _ in range(MAX_PARSE_RETRIES + 1)],
        "go",
        capabilities=text_only,
        width=200,
    )
    await harness.run()
    attempts = MAX_PARSE_RETRIES + 1
    assert (
        f"error: scripted did not produce a valid tool call in {attempts} attempts "
        "— choose a model with native tool calling (--model)"
    ) in harness.text


@pytest.mark.parametrize(
    ("stop", "said"),
    [(StopKind.REFUSAL, "declined to continue"), (StopKind.MAX_TOKENS, "output limit")],
)
async def test_a_reply_that_was_refused_or_cut_off_says_why_the_prompt_stopped(
    tmp_repo, stop, said
):
    cut = ModelReply((TextBlock("partial"),), stop, Usage(), "scripted")
    harness = Harness(tmp_repo, [cut], "go")
    await harness.run()
    assert "stopped: " in harness.text and said in harness.text
    # What the model did say was shown once, by the reply, and is not repeated.
    assert harness.text.count("partial") == 1


async def test_a_prompt_that_completed_prints_nothing_about_stopping(tmp_repo):
    harness = Harness(tmp_repo, [says("done")], "go")
    await harness.run()
    assert "stopped" not in harness.text.lower()


# --------------------------------------------------------------------------
# Ctrl+C: it cancels the turn, not the REPL
# --------------------------------------------------------------------------


async def test_ctrl_c_during_a_turn_cancels_the_turn_and_the_next_prompt_is_answered(
    tmp_repo, monkeypatch
):
    sigint = Sigint(monkeypatch)
    harness = Harness(tmp_repo, [says("an answer, this time")], "first", "second")
    started = block_the_answer(harness.session, monkeypatch)
    repl_task = harness.start()
    await reached(started)
    sigint.press()
    assert await asyncio.wait_for(repl_task, timeout=5) == 0
    assert "cancelled" in harness.text and "an answer, this time" in harness.text
    # The session mended the interrupted turn: the model's next request is a valid one,
    # holding the first question, the note that it was interrupted, and the second.
    assert sent_messages(harness.session) == [
        "first",
        "[this turn was interrupted before it finished]",
        "second",
    ]
    # The handler lived for the turn and no longer.
    assert (sigint.installed, sigint.removed) == (2, 2)


async def test_the_sigint_handler_is_removed_after_a_turn_that_finished_and_one_that_failed(
    tmp_repo, monkeypatch
):
    sigint = Sigint(monkeypatch)
    harness = Harness(tmp_repo, [says("fine"), ModelError("boom")], "first", "second", width=200)
    await harness.run()
    assert (sigint.installed, sigint.removed) == (2, 2)


async def test_ctrl_c_still_cancels_a_turn_after_a_confirmation_was_answered_at_a_real_prompt(
    tmp_repo, monkeypatch
):
    # The question is asked through prompt_toolkit, which takes over SIGINT while it is up and
    # removes whatever handler is installed when it goes. The turn goes on after the answer,
    # and Ctrl+C has to still cancel it: it did not, and the REPL could not be interrupted.
    sigint = Sigint(monkeypatch)
    with create_pipe_input() as keyboard:
        console, buffer = plain_console()
        ui = ConsoleUI(console, prompter=new_prompter(input=keyboard, output=DummyOutput()))
        write = calls("Write", {"path": "w.txt", "content": "w"}, call_id="w1")
        session = build_session(tmp_repo, [write, says("after the write")], ui=ui)
        # The model's second answer, the one after the write, never comes.
        started = block_the_answer(session, monkeypatch, nth=2)
        keyboard.send_text("y\r")
        typed = ScriptedPrompter("write it", "and again")
        repl_task = asyncio.create_task(run_repl(session, console, prompt=typed, clock=FakeClock()))
        await reached(started)
        assert (tmp_repo / "w.txt").read_text() == "w"  # it was approved, and it ran
        sigint.press()
        assert await asyncio.wait_for(repl_task, timeout=5) == 0
    assert "cancelled" in buffer.getvalue()
    assert typed.asked == [PROMPT, PROMPT, PROMPT]  # and the next prompt was read


async def test_ctrl_c_at_a_confirmation_cancels_the_turn_and_writes_nothing(tmp_repo):
    harness = Harness(
        tmp_repo,
        [calls("Write", {"path": "w.txt", "content": "w"}, call_id="w1"), says("next answer")],
        "write it",
        KeyboardInterrupt(),
        "next",
    )
    assert await harness.run() == 0
    assert "cancelled" in harness.text and "next answer" in harness.text
    assert not (tmp_repo / "w.txt").exists()


async def test_a_ctrl_c_at_a_confirmation_counts_towards_leaving_too(tmp_repo):
    harness = Harness(
        tmp_repo,
        [calls("Write", {"path": "w.txt", "content": "w"}, call_id="w1")],
        "write it",
        KeyboardInterrupt(),  # at the question: cancels the turn
        KeyboardInterrupt(),  # at the prompt, straight after: leaves
        "never read",
    )
    assert await harness.run() == 0
    assert harness.text.splitlines()[-1] == "bye"
    assert len(harness.prompter.asked) == 3


async def test_two_confirmations_cancelled_within_a_second_leave(tmp_repo):
    write = calls("Write", {"path": "w.txt", "content": "w"}, call_id="w1")
    harness = Harness(
        tmp_repo,
        [write, calls("Write", {"path": "w.txt", "content": "w"}, call_id="w2")],
        "write it",
        KeyboardInterrupt(),  # at the first question
        "write it again",
        KeyboardInterrupt(),  # at the second, with no time having passed
        "never read",
    )
    assert await harness.run() == 0
    assert harness.text.splitlines()[-1] == "bye"
    assert len(harness.prompter.asked) == 4


async def test_ctrl_c_during_a_command_cancels_the_command_and_not_the_repl(tmp_repo, monkeypatch):
    sigint = Sigint(monkeypatch)
    harness = Harness(tmp_repo, [says("still here")], "/compact", "hello")
    started = asyncio.Event()

    async def slow_compact(instructions: str | None = None) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(harness.session, "compact", slow_compact)
    repl_task = harness.start()
    await reached(started)
    sigint.press()
    assert await asyncio.wait_for(repl_task, timeout=5) == 0
    assert "cancelled" in harness.text and "still here" in harness.text


async def test_cancelling_the_repl_itself_is_not_taken_for_ctrl_c(tmp_repo, monkeypatch):
    # The turn's CancelledError is the person's Ctrl+C only if nothing cancelled the REPL
    # as well. When it was the REPL that was cancelled, swallowing it would leave a loop
    # that cannot be stopped.
    harness = Harness(tmp_repo, [says("never")], "first", "second")
    started = block_the_answer(harness.session, monkeypatch)
    repl_task = harness.start()
    await reached(started)
    repl_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(repl_task, timeout=5)
    assert "cancelled" not in harness.text
    assert harness.prompter.asked == [PROMPT]  # the second prompt was never asked for


# --------------------------------------------------------------------------
# Spec 8: a second Ctrl+C within a second leaves
# --------------------------------------------------------------------------


async def test_a_single_ctrl_c_at_the_prompt_says_how_to_leave_and_goes_on(tmp_repo):
    harness = Harness(tmp_repo, [says("answer")], KeyboardInterrupt(), "hello")
    assert await harness.run() == 0
    assert "press Ctrl+C again to exit" in harness.text and "answer" in harness.text


async def test_two_ctrl_c_within_a_second_leave(tmp_repo):
    harness = Harness(tmp_repo, [], KeyboardInterrupt(), KeyboardInterrupt(), "never read")
    assert await harness.run() == 0
    assert harness.text.splitlines()[-1] == "bye"
    assert len(harness.prompter.asked) == 2


@pytest.mark.parametrize(
    ("gap", "leaves"), [(0.0, True), (1.0, True), (1.01, False), (30.0, False)]
)
async def test_the_second_ctrl_c_leaves_only_if_it_comes_within_a_second(tmp_repo, gap, leaves):
    harness = Harness(tmp_repo, [says("answer")], KeyboardInterrupt(), KeyboardInterrupt(), "hello")

    def a_moment_passes(_message: str) -> None:
        if len(harness.prompter.asked) == 2:  # between the two presses
            harness.clock.now += gap

    harness.prompter.on_ask = a_moment_passes
    assert await harness.run() == 0
    assert (len(harness.prompter.asked) == 2) is leaves
    assert ("answer" in harness.text) is not leaves


async def test_a_ctrl_c_that_cancelled_a_turn_counts_towards_leaving(tmp_repo, monkeypatch):
    sigint = Sigint(monkeypatch)
    harness = Harness(tmp_repo, [], "first", KeyboardInterrupt(), "never read")
    started = block_the_answer(harness.session, monkeypatch)
    repl_task = harness.start()
    await reached(started)
    sigint.press()
    assert await asyncio.wait_for(repl_task, timeout=5) == 0
    assert "cancelled" in harness.text and harness.text.splitlines()[-1] == "bye"
    assert len(harness.prompter.asked) == 2


async def test_two_ctrl_c_during_one_turn_leave(tmp_repo, monkeypatch):
    sigint = Sigint(monkeypatch)
    harness = Harness(tmp_repo, [], "first", "never read")
    started = block_the_answer(harness.session, monkeypatch)
    repl_task = harness.start()
    await reached(started)
    sigint.press()
    sigint.press()
    assert await asyncio.wait_for(repl_task, timeout=5) == 0
    assert harness.text.splitlines()[-1] == "bye"
    assert len(harness.prompter.asked) == 1


# --------------------------------------------------------------------------
# A whole conversation through the real session
# --------------------------------------------------------------------------


async def test_an_edit_is_previewed_asked_about_and_applied(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    harness = Harness(
        tmp_repo,
        [
            calls("Read", {"path": "a.py"}, call_id="r1"),
            calls(
                "Edit",
                {"path": "a.py", "edits": [{"old_string": "x = 1", "new_string": "x = 2"}]},
                call_id="e1",
            ),
            says("changed it"),
        ],
        "make x two",
        "y",
        width=300,
    )
    await harness.run()
    assert target.read_text() == "x = 2\n"
    assert harness.text.count("-x = 1") == 1  # shown once, at the confirmation, and not again
    assert "Read a.py" in harness.text and "changed it" in harness.text
    assert len(harness.prompter.asked) == 3  # the prompt, the confirmation, the next prompt


async def test_a_declined_edit_is_not_applied_and_the_model_is_told(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    harness = Harness(
        tmp_repo,
        [
            calls("Read", {"path": "a.py"}, call_id="r1"),
            calls(
                "Edit",
                {"path": "a.py", "edits": [{"old_string": "x = 1", "new_string": "x = 2"}]},
                call_id="e1",
            ),
            says("understood"),
        ],
        "make x two",
        "n",
    )
    await harness.run()
    assert target.read_text() == "x = 1\n"
    declined = harness.session.model.requests[2].transcript.messages[-1].tool_results()[0]
    assert declined.is_error and "declined" in declined.content


async def test_a_refusal_by_the_real_policy_is_shown_with_its_rule_and_reason(tmp_repo):
    harness = Harness(
        tmp_repo,
        [calls("Read", {"path": "/etc/passwd"}, call_id="r1"), says("I cannot read that")],
        "read the password file",
        width=300,
    )
    await harness.run()
    [refusal] = [ln for ln in harness.text.splitlines() if ln.startswith("refused")]
    assert refusal.startswith("refused (sandbox.outside-root): ") and "/etc/passwd" in refusal


# --------------------------------------------------------------------------
# How the REPL builds its prompt
# --------------------------------------------------------------------------


async def test_the_repl_builds_its_prompt_from_the_session_and_the_environment(
    tmp_repo, monkeypatch
):
    asked: dict[str, Any] = {}

    def spy(root: str, history_path: str | None, **options: Any) -> ScriptedPrompter:
        asked.update(root=root, history_path=history_path, **options)
        return ScriptedPrompter()

    monkeypatch.setattr(repl, "build_prompt_session", spy)
    monkeypatch.setenv("NANOCLAUDE_EDITING_MODE", "vi")
    harness = Harness(tmp_repo, [])
    await run_repl(harness.session, harness.console, str(tmp_repo / "history"), clock=harness.clock)
    assert asked["root"] == harness.session.root
    assert asked["history_path"] == str(tmp_repo / "history")
    assert asked["vi_mode"] is True
    assert asked["commands"] == {name: c.summary for name, c in COMMANDS.items()}
