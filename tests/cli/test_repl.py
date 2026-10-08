"""Tests for nanoclaude.cli.repl: the loop a person lives in.

A prompt is answered from a script (``ScriptedPrompter``), so these tests run a whole
conversation with no terminal: the lines typed, Ctrl+C and Ctrl+D are the script's
entries. Time is a ``FakeClock`` that moves when the test says so, and a turn that has to
be interrupted blocks on an event, so nothing here waits for time to pass.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
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
from nanoclaude.providers.base import (
    CredentialsError,
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    Usage,
)
from nanoclaude.providers.capabilities import Capabilities
from nanoclaude.providers.texttools import MAX_PARSE_RETRIES
from nanoclaude.testing.scripted import calls, cut_off, says
from nanoclaude.testing.session import ScriptedSession, build_session
from tests.cli.helpers import FakeClock, ScriptedPrompter, plain_console, terminal_console
from tests.cli.screen import cursor_is_hidden, foreign_controls, on_screen


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
        terminal: bool = False,
        height: int = 30,
        **options: Any,
    ) -> None:
        """``terminal`` makes the console a terminal of ``height`` lines, which draws a live
        display for a reply that is arriving; the default console is not one."""
        if terminal:
            self.console, self._buffer = terminal_console(width, height=height, no_color=False)
        else:
            self.console, self._buffer = plain_console(width)
        self._terminal = terminal
        self._height = height
        self.prompter = ScriptedPrompter(*lines)
        self.clock = FakeClock()
        self.session: ScriptedSession = build_session(
            root,
            script,
            ui=ConsoleUI(
                self.console, prompter=self.prompter, root=str(root), model_of=self._model_of
            ),
            **options,
        )

    def _model_of(self, role: str) -> str:
        """The model that plays ``role``, as the live router says: it changes with /model."""
        config = self.session.router.config
        return config.models[config.roles.alias_for(role)].model

    @property
    def text(self) -> str:
        return self._buffer.getvalue()

    @property
    def rows(self) -> list[str]:
        """The lines a person would read: what a terminal shows, or a plain console printed."""
        if self._terminal:
            return on_screen(self.text, self._height).splitlines()
        return [line.rstrip() for line in self.text.splitlines()]

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
        self._last: Callable[[], None] | None = None
        self.installed = 0
        self.removed = 0
        add, remove = loop.add_signal_handler, loop.remove_signal_handler

        def spy_add(sig: int, callback: Callable[..., Any], *args: Any) -> None:
            if sig == signal.SIGINT:
                self._handler = self._last = functools.partial(callback, *args)
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

    def press_late(self) -> None:
        """Deliver a press to the handler of the turn that has just ended.

        The loop schedules the handler when the signal arrives, and removing it does not
        take back a call that is already scheduled: it can run after the turn is over.
        """
        assert self._last is not None, "no handler was ever installed"
        self._last()


def block_the_answer(
    session: ScriptedSession, monkeypatch: pytest.MonkeyPatch, nth: int = 1
) -> asyncio.Event:
    """Make the model's ``nth`` answer never come, and say when it was asked for.

    The answers before it come from the script as usual, and so do the ones after.
    """
    started = asyncio.Event()
    answer = session.model.complete
    asked = 0

    async def complete(
        request: ModelRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelReply:
        nonlocal asked
        asked += 1
        if asked == nth:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("the model was allowed to answer")
        return await answer(request, on_text=on_text)

    monkeypatch.setattr(session.model, "complete", complete)
    return started


def stream_then_stall(
    session: ScriptedSession, monkeypatch: pytest.MonkeyPatch, text: str, nth: int = 1
) -> asyncio.Event:
    """Make the model's ``nth`` answer stream ``text`` and then never finish, and say when.

    The answers before it come from the script as usual, and so do the ones after.
    """
    streaming = asyncio.Event()
    answer = session.model.complete
    asked = 0

    async def complete(
        request: ModelRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelReply:
        nonlocal asked
        asked += 1
        if asked == nth:
            assert on_text is not None, "the session did not ask for the text as it arrives"
            on_text(text)
            streaming.set()
            await asyncio.Event().wait()
            raise AssertionError("the model was allowed to finish")
        return await answer(request, on_text=on_text)

    monkeypatch.setattr(session.model, "complete", complete)
    return streaming


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


TEXT_TOOL_BANNER = "\u26a0 text-tool mode (model lacks native tool calling)"  # spec 4.4, verbatim


async def test_a_main_model_without_native_tool_calling_is_flagged_at_start(tmp_repo):
    harness = Harness(tmp_repo, [], native_tools=False)
    await harness.run()
    assert harness.text.splitlines()[1] == TEXT_TOOL_BANNER


async def test_a_main_model_with_native_tool_calling_gets_no_such_warning(tmp_repo):
    harness = Harness(tmp_repo, [], native_tools=True)
    await harness.run()
    assert "text-tool mode" not in harness.text


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
            "error: overloaded — try again, or switch models with /model",
        ),
        (
            ModelError("rejected your credentials (HTTP 401): bad key — check the key"),
            "error: rejected your credentials (HTTP 401): bad key — check the key",
        ),
        (
            CredentialsError("the provider rejected your credentials (HTTP 401) — check the key"),
            "error: the provider rejected your credentials (HTTP 401) — check the key",
        ),
        (
            ContextTooSmallError("the prompt alone needs 9000 tokens. Use a larger window."),
            (
                "error: the prompt alone needs 9000 tokens. Use a larger window"
                " — /model switches to a model with a larger window"
            ),
        ),
        (
            LoopError("model returned an empty reply"),
            (
                "error: the conversation is in an unexpected state (model returned an empty reply)"
                " — /clear starts it over"
            ),
        ),
        (
            TranscriptError("message 3 has no blocks"),
            (
                "error: the conversation is in an unexpected state (message 3 has no blocks)"
                " — /clear starts it over"
            ),
        ),
        (
            sqlite3.OperationalError("database is locked"),
            (
                "error: the session store failed (database is locked) "
                "— check that the disk has room and the database file is writable, then try again"
            ),
        ),
        (
            RuntimeError("boom"),
            (
                "error: unexpected error (RuntimeError: boom) — this is a bug; "
                "/clear starts the conversation over"
            ),
        ),
        (
            RuntimeError(""),
            (
                "error: unexpected error (RuntimeError) — this is a bug; "
                "/clear starts the conversation over"
            ),
        ),
        (
            ModelError(""),
            "error: the model request failed — try again, or switch models with /model",
        ),
        (
            ContextTooSmallError(""),
            (
                "error: the context window is too small for this conversation"
                " — /model switches to a model with a larger window"
            ),
        ),
    ],
)
def test_a_failure_is_described_in_the_form_the_spec_gives_for_a_person(failure, line):
    assert error_message(failure) == line


@pytest.mark.parametrize(
    "failure",
    [
        KeyError("missing"),
        OSError(28, "No space left on device"),
        UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed"),
        TimeoutError(),
        ValueError(""),
        AttributeError("'NoneType' object has no attribute 'x'"),
        *FAILURES,
    ],
    ids=lambda e: f"{type(e).__name__}-{str(e)[:12]}",
)
def test_every_failure_is_reported_as_one_error_line_that_says_what_to_do(failure):
    # Spec 17.9: "error: <what happened> \u2014 <what to do>", whatever failed, including
    # what nobody has seen before.
    line = error_message(failure)
    assert line.startswith("error: ") and " \u2014 " in line and "\n" not in line
    assert line.count("error: ") == 1


@pytest.mark.parametrize(
    ("failure", "line"),
    [
        (
            ModelError("overloaded\nerror: second"),
            "error: overloaded error: second \u2014 try again, or switch models with /model",
        ),
        (
            RuntimeError("first\r\n\n  second\tthird "),
            (
                "error: unexpected error (RuntimeError: first second third) \u2014 this is a bug; "
                "/clear starts the conversation over"
            ),
        ),
        (
            ModelError("tried\nagain \u2014 wait a moment\nthen retry."),
            "error: tried again \u2014 wait a moment then retry",
        ),
    ],
    ids=["model-error", "runtime-error", "already-says-what-to-do"],
)
def test_a_failure_whose_text_runs_over_several_lines_is_still_one_error_line(failure, line):
    # A second line of an error line is a line nobody wrote the prefix of, and the text is
    # the provider's: it can begin with "error:" and be taken for a second error.
    assert error_message(failure) == line


async def test_a_failure_that_quotes_several_lines_prints_one_error_line(tmp_repo, monkeypatch):
    harness = Harness(tmp_repo, [], "hello", width=300)

    async def boom(prompt: str) -> Done:
        raise ModelError("overloaded\nerror: second")

    monkeypatch.setattr(harness.session, "follow_up", boom)
    await harness.run()
    assert [ln for ln in harness.text.splitlines() if "overloaded" in ln or "second" in ln] == [
        "error: overloaded error: second \u2014 try again, or switch models with /model"
    ]


async def test_a_reported_failure_carries_the_error_prefix_exactly_once(tmp_repo, monkeypatch):
    harness = Harness(tmp_repo, [], "hello", width=300)

    async def boom(prompt: str) -> Done:
        raise RuntimeError("boom")

    monkeypatch.setattr(harness.session, "follow_up", boom)
    await harness.run()
    assert "error: unexpected error (RuntimeError: boom) \u2014 this is a bug" in harness.text
    assert "error: error:" not in harness.text


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


async def test_a_turn_is_not_left_running_when_the_sigint_handler_cannot_be_installed(
    tmp_repo, monkeypatch
):
    # Off the main thread, or where the loop has no signal handlers, installing one raises.
    # The turn must not be started and forgotten: a task nobody awaits goes on talking to
    # the model behind the REPL's back.
    loop = asyncio.get_running_loop()

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("set_wakeup_fd only works in main thread of the main interpreter")

    monkeypatch.setattr(loop, "add_signal_handler", refuse)
    harness = Harness(tmp_repo, [says("the turn ran")], "hello", "again", width=300)
    assert await harness.run() == 0
    await asyncio.sleep(0)  # give anything left running a turn of the loop
    await asyncio.sleep(0)
    assert harness.session.model.requests == []
    assert "the turn ran" not in harness.text
    assert asyncio.all_tasks() == {asyncio.current_task()}
    reported = [ln for ln in harness.text.splitlines() if ln.startswith("error:")]
    assert len(reported) == 2 and all("set_wakeup_fd" in ln for ln in reported)


async def test_the_sigint_handler_is_removed_when_the_task_cannot_be_created(monkeypatch):
    # Nothing that can raise may sit between installing the handler and the try that removes it:
    # a handler left behind would cancel whatever task is the REPL's next, and nobody would
    # know why. The work that was never started is closed, not left to be reported unawaited.
    sigint = Sigint(monkeypatch)
    ran: list[int] = []

    async def work() -> None:
        ran.append(1)

    def refuse(_coroutine: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no task for you")

    coroutine = work()
    monkeypatch.setattr(asyncio, "create_task", refuse)
    with pytest.raises(RuntimeError, match="no task for you"):
        await repl._cancellable(coroutine, repl._DoubleTap(FakeClock()))
    assert (sigint.installed, sigint.removed) == (1, 1)
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED and ran == []


async def test_a_press_that_arrives_after_its_turn_cannot_turn_the_next_cancel_into_an_exit(
    tmp_repo, monkeypatch
):
    sigint = Sigint(monkeypatch)
    harness = Harness(
        tmp_repo, [says("first answer"), says("third answer")], "first", "second", "third"
    )
    started = block_the_answer(harness.session, monkeypatch, nth=2)
    asked_for_second = 0

    def two_presses_straggle_in(_message: str) -> None:
        # The question for the second prompt: the first turn is over. Two presses that were
        # scheduled before its handler was removed arrive now, within a second of each other.
        nonlocal asked_for_second
        asked_for_second += 1
        if asked_for_second == 2:
            sigint.press_late()
            sigint.press_late()

    harness.prompter.on_ask = two_presses_straggle_in
    repl_task = harness.start()
    await reached(started)
    sigint.press()  # one press, during the second turn: it is cancelled, and nothing more
    assert await asyncio.wait_for(repl_task, timeout=5) == 0
    assert "cancelled" in harness.text and "third answer" in harness.text
    assert len(harness.prompter.asked) == 4  # the third prompt was read, and then Ctrl+D


async def test_compact_after_a_cancelled_turn_says_there_was_nothing_to_compact(
    tmp_repo, monkeypatch
):
    # /compact closes what the cancelled turn left open before it looks for anything to
    # summarise. That adds a note and so makes the conversation larger, which is not a
    # compaction, and must not be reported as one.
    sigint = Sigint(monkeypatch)
    harness = Harness(tmp_repo, [], "first", "/compact", width=300)
    started = block_the_answer(harness.session, monkeypatch)
    repl_task = harness.start()
    await reached(started)
    sigint.press()
    assert await asyncio.wait_for(repl_task, timeout=5) == 0
    assert "nothing to compact" in harness.text and "compacted" not in harness.text


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


async def test_a_real_sigint_cancels_the_turn_and_the_next_prompt_is_answered(
    tmp_repo, monkeypatch
):
    # The signal itself, through the loop's own plumbing and not a call to the handler. If
    # the REPL had installed none, the signal would reach this test, and the stand-in below
    # turns that into a failure of this test and not an interruption of the whole run.
    def nobody_handles_this(_signum: int, _frame: object) -> None:
        raise AssertionError("a SIGINT reached the process: the REPL's handler was not installed")

    previous = signal.signal(signal.SIGINT, nobody_handles_this)
    try:
        harness = Harness(tmp_repo, [says("an answer, this time")], "first", "second")
        started = block_the_answer(harness.session, monkeypatch)
        repl_task = harness.start()
        await reached(started)
        asyncio.get_running_loop().call_later(0.1, signal.raise_signal, signal.SIGINT)
        assert await asyncio.wait_for(repl_task, timeout=5) == 0
    finally:
        signal.signal(signal.SIGINT, previous)
    assert "cancelled" in harness.text and "an answer, this time" in harness.text
    assert harness.text.splitlines()[-1] == "bye"


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


async def test_a_write_is_previewed_asked_about_and_written(tmp_repo):
    harness = Harness(
        tmp_repo,
        [
            calls("Write", {"path": "new.txt", "content": "hello\nworld\n"}, call_id="w1"),
            says("written"),
        ],
        "write it",
        "y",
        width=300,
    )
    await harness.run()
    assert (tmp_repo / "new.txt").read_text() == "hello\nworld\n"
    lines = harness.text.splitlines()
    # The path inside the project is relative to it, and what will be written comes before
    # the question, which is the third thing the keyboard was asked for.
    assert "Write  new.txt" in lines
    assert lines[lines.index("Write  new.txt") + 1 : lines.index("Write  new.txt") + 4] == [
        "  new file: 2 lines, 12 bytes",
        "    1  hello",
        "    2  world",
    ]
    assert len(harness.prompter.asked) == 3


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


# --------------------------------------------------------------------------
# A reply that is arriving when the person cancels, or when the stream breaks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("terminal", [False, True], ids=["plain console", "terminal"])
async def test_ctrl_c_while_a_reply_is_streaming_leaves_what_arrived_on_the_screen_only(
    tmp_repo, monkeypatch, terminal
):
    sigint = Sigint(monkeypatch)
    harness = Harness(tmp_repo, [says("a second answer")], "first", "second", terminal=terminal)
    streaming = stream_then_stall(harness.session, monkeypatch, "Once upon a time")
    repl_task = harness.start()
    await reached(streaming)
    sigint.press()
    assert await asyncio.wait_for(repl_task, timeout=5) == 0
    # On the screen once, and before the notice that the turn was cancelled, whichever
    # kind of console it was.
    rows = harness.rows
    assert rows.count("Once upon a time") == 1
    assert rows.index("Once upon a time") < rows.index("cancelled") < rows.index("a second answer")
    # Not in the conversation: the repair at the next prompt is what the model is told.
    assert sent_messages(harness.session) == [
        "first",
        "[this turn was interrupted before it finished]",
        "second",
    ]
    # Nothing is left drawing, and the cursor is the person's again.
    assert cursor_is_hidden(harness.text) is False


async def test_a_stream_that_breaks_keeps_what_arrived_and_the_person_can_ask_for_the_rest(
    tmp_repo,
):
    harness = Harness(
        tmp_repo,
        [cut_off("The first half of"), says("an answer, and here is the rest")],
        "write it",
        "continue",
        width=200,
        terminal=True,
    )
    assert await harness.run() == 0
    rows = harness.rows
    assert rows.count("The first half of") == 1
    [error] = [row for row in rows if row.startswith("error:")]
    assert error == (
        "error: the reply was cut off (the connection to the provider was lost: connection reset "
        "by peer) \u2014 what arrived is kept; ask the model to continue"
    )
    assert rows.index("The first half of") < rows.index(error)
    # Asked to continue, the model was shown what it had said, and that it was cut off.
    assert sent_messages(harness.session, 1) == [
        "write it",
        "The first half of\n\n[this reply was cut off before it finished]",
        "continue",
    ]
    assert rows[-2] == "an answer, and here is the rest"
    assert foreign_controls(harness.text) == []


async def test_a_question_after_streamed_text_is_asked_when_nothing_is_drawing(tmp_repo):
    harness = Harness(
        tmp_repo,
        [
            calls(
                "Write",
                {"path": "new.txt", "content": "hello\n"},
                call_id="w1",
                preamble="Writing it now.",
            ),
            says("written"),
        ],
        "write it",
        "y",
        terminal=True,
        width=100,
    )
    when_asked: list[tuple[bool, list[str]]] = []
    harness.prompter.on_ask = lambda _message: when_asked.append(
        (cursor_is_hidden(harness.text), harness.rows)
    )
    assert await harness.run() == 0
    assert (tmp_repo / "new.txt").read_text() == "hello\n"
    _prompt, (hidden, rows), _next = when_asked
    assert hidden is False
    assert rows.count("Writing it now.") == 1  # on the screen above the question, once
