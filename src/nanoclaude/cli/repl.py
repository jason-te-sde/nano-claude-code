# src/nanoclaude/cli/repl.py
"""The interactive loop.

Input goes through prompt_toolkit (see :mod:`nanoclaude.cli.prompt`) because a terminal
agent people live in needs real line editing, history and completion; ``input()`` gets
abandoned within a day.

**Ctrl+C cancels what is running, not the REPL.** The common case is "not like that",
not "I want to leave". Under ``asyncio.run`` a Ctrl+C is not an exception that can be
caught around an ``await``: it cancels the main task and ends the process with a
``KeyboardInterrupt`` out of ``asyncio.run``. So for as long as a turn or a command runs
a handler is installed that cancels just that, and it is removed when the work ends. The
session mends what the interruption left half done when the next prompt arrives. Signal
handlers are Unix-only, which is the platform this supports. Spec 8 asks for a second
Ctrl+C within a second to leave, and Ctrl+D to leave at any time the prompt is up.

**Every prompt goes through ``Session.follow_up``**, the first included. ``run()`` starts
a conversation, and on a resumed session that would archive the history just loaded;
``follow_up()`` continues it, and starts one when nothing has been said yet.

**A failure does not end the session.** A command or turn that raises is reported, as spec
17.9 words an error for a person, and the next prompt is read. Only Ctrl+D, a second
Ctrl+C, and ``/exit`` end it. Cancellation is not a failure and is not caught as one.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sqlite3
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, assert_never

from rich.console import Console

from nanoclaude.agent.loop import Done, LoopError, StopReason
from nanoclaude.agent.session import Session
from nanoclaude.cli.commands import COMMANDS, dispatch
from nanoclaude.cli.prompt import Prompter, build_prompt_session, wants_vi_mode
from nanoclaude.cli.render import ERROR_PREFIX, plain, show_error, show_notice
from nanoclaude.conversation.budget import ContextTooSmallError
from nanoclaude.conversation.transcript import TranscriptError
from nanoclaude.providers.base import ModelError

PROMPT = "> "

#: Spec 8: two Ctrl+C within this many seconds leave.
LEAVE_WITHIN_S = 1.0

#: A slash command is a slash and a word. ``/etc/hosts has a typo`` is a prompt that
#: begins with a path, and is not looked up in the command table.
_COMMAND = re.compile(r"/([A-Za-z][\w-]*)(?:\s|$)")

#: Blank lines before the first line with something on it.
_LEADING_BLANK_LINES = re.compile(r"\A(?:[ \t]*\r?\n)+")

_T = TypeVar("_T")


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """``("model", ["c"])`` for ``/model c``; None for anything that is not a command."""
    match = _COMMAND.match(text)
    if match is None:
        return None
    return match.group(1), text[match.end() :].split()


def error_message(exc: Exception) -> str:
    """The line a person is shown when a command or a turn failed, in the form of spec 17.9.

    ``error: <what happened> — <what to do>``, whatever failed, including what nobody has
    seen before. A message that already says what to do (the provider and session errors
    are written that way) is left as it is; the others get the step that fits what failed.
    """
    return f"{ERROR_PREFIX}{_what_to_say(exc)}"


def _what_to_say(exc: Exception) -> str:
    """``<what happened> — <what to do>`` for ``exc``, without the prefix.

    What happened starts in lower case, as spec 17.9 words it, and is never a bare class
    name. It is one line: the exception's text is somebody else's, and a second line of it
    would be a line of the error that nobody wrote the prefix of.
    """
    detail = " ".join(str(exc).split()).rstrip(".")
    if " — " in detail:
        return detail
    name = type(exc).__name__
    if isinstance(exc, sqlite3.Error):
        return (
            f"the session store failed ({detail or name}) — check that the disk has room "
            "and the database file is writable, then try again"
        )
    if isinstance(exc, LoopError | TranscriptError):
        return (
            f"the conversation is in an unexpected state ({detail or name}) — /clear starts it over"
        )
    if isinstance(exc, ContextTooSmallError):
        what = detail or "the context window is too small for this conversation"
        return f"{what} — /model switches to a model with a larger window"
    if isinstance(exc, ModelError):
        return f"{detail or 'the model request failed'} — try again, or switch models with /model"
    what = f"{name}: {detail}" if detail else name
    return f"unexpected error ({what}) — this is a bug; /clear starts the conversation over"


class _Cancelled(Exception):
    """The person pressed Ctrl+C while something ran, and it was cancelled."""


class _DoubleTap:
    """Spec 8: Ctrl+C cancels, and a second one within a second leaves."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._last: float | None = None
        #: Set when a press was the second within the window while something was running,
        #: so that the REPL leaves once that has been cancelled.
        self.leave = False

    def pressed(self) -> bool:
        """Record a Ctrl+C. True if it is the second within :data:`LEAVE_WITHIN_S`."""
        now = self._clock()
        second = self._last is not None and now - self._last <= LEAVE_WITHIN_S
        self._last = None if second else now
        return second


async def _cancellable(work: Coroutine[Any, Any, _T], taps: _DoubleTap) -> _T:
    """Run ``work`` so that Ctrl+C cancels it and nothing else.

    Raises :class:`_Cancelled` when the person cancelled it. A cancellation that is not
    theirs, the REPL task's own, is not turned into one: it passes through, because
    swallowing it would leave a loop that nothing can stop.

    The handler is installed before the work is started. Installing it can fail (off the
    main thread, or where the loop has no signal handlers), and work that had been started
    by then would be left running with nobody awaiting it. Once it is installed, the ``try``
    that removes it begins at once: nothing that can raise may sit between the two, or a
    handler is left behind for whatever runs next. And what an earlier press left in
    ``taps.leave`` is not this work's: a press that arrives after its own turn has ended
    must not turn the next single Ctrl+C into an exit.
    """
    loop = asyncio.get_running_loop()
    taps.leave = False
    task: asyncio.Task[_T] | None = None
    by_signal = False

    def interrupt() -> None:
        nonlocal by_signal
        by_signal = True
        if taps.pressed():
            taps.leave = True
        if task is not None:
            task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, interrupt)
    except BaseException:
        work.close()  # it never started, and must not be reported as never awaited
        raise
    try:
        task = asyncio.create_task(work)
        return await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
        if not by_signal and taps.pressed():
            # The Ctrl+C arrived as a key, at a question the turn was asking, and not as a
            # signal. It is a press all the same.
            taps.leave = True
        raise _Cancelled from None
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        if task is None:
            work.close()  # no task was made for it, and it must not be reported as never awaited


def _report_stop(console: Console, done: Done) -> None:
    """Say why a prompt stopped, when it did not finish."""
    match done.reason:
        case StopReason.COMPLETED:
            return
        case StopReason.TURN_LIMIT | StopReason.MODEL_UNSUITABLE:
            # The loop's own closing line, and the spec 17.9 message that names the model.
            # Nothing else has said either: they were not the model's words.
            line = done.text
        case StopReason.REFUSAL:
            line = (
                "stopped: the model declined to continue "
                "— rephrase the request, or try another model with /model"
            )
        case StopReason.MAX_TOKENS:
            line = "stopped: the reply was cut off at the model's output limit — ask it to continue"
        case _:
            assert_never(done.reason)
    console.print(plain(line, "yellow"))


def _leave(console: Console) -> int:
    show_notice(console, "bye")
    return 0


async def run_repl(
    session: Session,
    console: Console,
    history_path: str | None = None,
    *,
    prompt: Prompter | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Read lines until the person leaves, and answer each. Returns the exit code, 0.

    ``history_path`` is where what was typed is kept between sessions. ``prompt`` and
    ``clock`` are for tests: a scripted prompt in place of the terminal, and time that
    moves when the test says so. The session is not closed here: whoever made it does.
    """
    if prompt is None:
        prompt = build_prompt_session(
            session.root,
            history_path,
            commands={name: command.summary for name, command in COMMANDS.items()},
            vi_mode=wants_vi_mode(os.environ),
        )
    config = session.router.config
    show_notice(
        console,
        f"nano-claude-code · {config.models[config.roles.main].model} "
        "· /help for commands · Ctrl+D to exit",
    )
    taps = _DoubleTap(clock)
    while True:
        try:
            line = await prompt.prompt_async(PROMPT)
        except EOFError:
            return _leave(console)
        except KeyboardInterrupt:
            if taps.pressed():
                return _leave(console)
            show_notice(console, "press Ctrl+C again to exit")
            continue
        # Only the ends are trimmed. The first line of a pasted block is indented relative to
        # the lines below it, and that indentation is code.
        text = _LEADING_BLANK_LINES.sub("", line.rstrip())
        if not text.strip():
            continue
        command = parse_command(text.strip())
        try:
            if command is not None:
                name, args = command
                if not await _cancellable(dispatch(name, args, session, console), taps):
                    return _leave(console)
            else:
                _report_stop(console, await _cancellable(session.follow_up(text), taps))
        except _Cancelled:
            show_notice(console, "cancelled")
            if taps.leave:
                return _leave(console)
        except Exception as exc:
            show_error(console, error_message(exc))
