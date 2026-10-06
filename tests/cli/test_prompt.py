"""Tests for nanoclaude.cli.prompt: what a person types into.

A real prompt_toolkit session is driven here through a pipe, with the screen swallowed,
so the keys, the completions and the history are the ones a terminal would deliver.
"""

from __future__ import annotations

import asyncio
import os
import pty
import select
import signal
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from nanoclaude.cli.prompt import (
    CommandCompleter,
    MentionCompleter,
    build_prompt_session,
    discard_pending_input,
    new_prompter,
    wants_vi_mode,
)

COMMANDS = {
    "help": "list these commands",
    "clear": "start a fresh conversation",
    "compact": "summarise the conversation so far",
    "model": "show or switch the model",
    "mode": "show or switch the permission mode",
}


@pytest.fixture
def keyboard() -> Iterator[PipeInput]:
    """The keys a person presses, as bytes on a pipe, and the pipe closed afterwards."""
    with create_pipe_input() as pipe:
        yield pipe


def session_on(
    keyboard: PipeInput, root: Path, history: Path | None = None, *, vi_mode: bool = False
) -> PromptSession[str]:
    return build_prompt_session(
        str(root),
        str(history) if history else None,
        commands=COMMANDS,
        vi_mode=vi_mode,
        input=keyboard,
        output=DummyOutput(),
    )


async def typed(keyboard: PipeInput, root: Path, keys: str, *, vi_mode: bool = False) -> str:
    """Press ``keys`` at a fresh prompt and return what the prompt gives back."""
    keyboard.send_text(keys)
    return await session_on(keyboard, root, vi_mode=vi_mode).prompt_async("> ")


def completions(completer: Completer, text: str) -> list[Completion]:
    return list(completer.get_completions(Document(text), CompleteEvent()))


async def until(condition: Callable[[], bool], what: str, *, seconds: float = 5.0) -> None:
    """Wait for something the prompt does in the background, and fail if it never does."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"gave up waiting for {what}")
        await asyncio.sleep(0)


# --------------------------------------------------------------------------
# A bare prompt, for a single question
# --------------------------------------------------------------------------


async def test_a_bare_prompt_returns_the_line_that_was_typed(keyboard):
    keyboard.send_text("yes\r")
    prompter = new_prompter(input=keyboard, output=DummyOutput())
    assert await prompter.prompt_async("allow? ") == "yes"


async def test_ctrl_c_at_a_bare_prompt_raises_keyboard_interrupt(keyboard):
    keyboard.send_text("half an answer\x03")
    prompter = new_prompter(input=keyboard, output=DummyOutput())
    with pytest.raises(KeyboardInterrupt):
        await prompter.prompt_async("allow? ")


async def test_ctrl_d_at_a_bare_prompt_raises_end_of_file(keyboard):
    keyboard.send_text("\x04")
    prompter = new_prompter(input=keyboard, output=DummyOutput())
    with pytest.raises(EOFError):
        await prompter.prompt_async("allow? ")


async def test_a_bare_prompt_leaves_the_sigint_handler_of_the_turn_it_is_asked_in_alone(keyboard):
    # prompt_toolkit installs a SIGINT handler of its own while a prompt is up and removes
    # whatever is there when it is done, which would take the REPL's handler for the turn
    # with it. A question asked in the middle of a turn must not do that.
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: None)
    try:
        keyboard.send_text("yes\r")
        prompter = new_prompter(input=keyboard, output=DummyOutput())
        assert await prompter.prompt_async("allow? ") == "yes"
        # remove_signal_handler says whether there was a handler to remove.
        assert loop.remove_signal_handler(signal.SIGINT) is True
    finally:
        loop.remove_signal_handler(signal.SIGINT)


# --------------------------------------------------------------------------
# Keys: spec 8 asks for several lines by Esc+Enter or a trailing backslash
# --------------------------------------------------------------------------


async def test_enter_submits_the_line(keyboard, tmp_path):
    assert await typed(keyboard, tmp_path, "fix the parser\r") == "fix the parser"


async def test_escape_then_enter_starts_another_line(keyboard, tmp_path):
    assert await typed(keyboard, tmp_path, "first\x1b\rsecond\r") == "first\nsecond"


async def test_a_trailing_backslash_continues_the_line_and_is_not_kept(keyboard, tmp_path):
    assert await typed(keyboard, tmp_path, "first\\\rsecond\r") == "first\nsecond"


async def test_a_backslash_that_is_not_last_does_not_continue_anything(keyboard, tmp_path):
    assert await typed(keyboard, tmp_path, "C:\\dir\\file\r") == "C:\\dir\\file"


async def test_a_space_after_the_backslash_submits_it_as_typed(keyboard, tmp_path):
    # The way to send a line that really does end in a backslash.
    assert await typed(keyboard, tmp_path, "path\\ \r") == "path\\ "


async def test_a_pasted_block_arrives_as_one_prompt_with_its_line_breaks(keyboard, tmp_path):
    # Bracketed paste: the newline inside the paste is text, and does not submit.
    pasted = "\x1b[200~def f():\n    return 1\x1b[201~\r"
    assert await typed(keyboard, tmp_path, pasted) == "def f():\n    return 1"


async def test_ctrl_c_interrupts_the_prompt(keyboard, tmp_path):
    with pytest.raises(KeyboardInterrupt):
        await typed(keyboard, tmp_path, "half a thought\x03")


async def test_ctrl_d_on_an_empty_line_ends_the_input(keyboard, tmp_path):
    with pytest.raises(EOFError):
        await typed(keyboard, tmp_path, "\x04")


async def test_vi_keys_are_available_and_escape_then_enter_still_submits_there(keyboard, tmp_path):
    # Esc leaves insert mode in vi, so Esc+Enter is "stop typing, submit": making it a
    # newline there would break the muscle memory of anyone who picks vi.
    assert session_on(keyboard, tmp_path, vi_mode=True).editing_mode is EditingMode.VI
    assert await typed(keyboard, tmp_path, "ab\x1b\r", vi_mode=True) == "ab"


async def test_the_default_is_emacs_keys(keyboard, tmp_path):
    assert session_on(keyboard, tmp_path).editing_mode is EditingMode.EMACS


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"NANOCLAUDE_EDITING_MODE": "vi"}, True),
        ({"NANOCLAUDE_EDITING_MODE": " VIM "}, True),
        ({"NANOCLAUDE_EDITING_MODE": "emacs"}, False),
        ({"NANOCLAUDE_EDITING_MODE": "nonsense"}, False),
        ({}, False),
    ],
)
def test_vi_keys_are_chosen_by_an_environment_variable(environ, expected):
    assert wants_vi_mode(environ) is expected


# --------------------------------------------------------------------------
# History: kept across sessions, and searchable
# --------------------------------------------------------------------------


async def test_what_is_typed_is_kept_for_the_next_session(keyboard, tmp_path):
    history = tmp_path / "state" / "history"  # its directory does not exist yet
    keyboard.send_text("fix the parser\r")
    await session_on(keyboard, tmp_path, history).prompt_async("> ")
    later = session_on(keyboard, tmp_path, history)
    assert [line async for line in later.history.load()] == ["fix the parser"]


async def test_an_earlier_line_can_be_found_by_searching_backwards(keyboard, tmp_path):
    session = session_on(keyboard, tmp_path)
    keyboard.send_text("alpha one\r")
    await session.prompt_async("> ")
    keyboard.send_text("beta two\r")
    await session.prompt_async("> ")
    # Ctrl+R, the letters to look for, Enter to take the match and Enter to submit it.
    keyboard.send_text("\x12alp\r\r")
    assert await session.prompt_async("> ") == "alpha one"


# --------------------------------------------------------------------------
# Completion: "/" completes a command, "@" completes a path in the project
# --------------------------------------------------------------------------


def test_a_slash_completes_to_the_commands_that_begin_that_way():
    [found] = completions(CommandCompleter(COMMANDS), "/co")
    assert found.text == "/compact" and found.start_position == -3
    assert found.display_meta_text == "summarise the conversation so far"


def test_a_bare_slash_offers_every_command_in_order():
    offered = [c.text for c in completions(CommandCompleter(COMMANDS), "/")]
    assert offered == ["/clear", "/compact", "/help", "/mode", "/model"]


@pytest.mark.parametrize("text", ["/model ", "/model m", "hello /he", "he", ""])
def test_a_command_is_completed_only_at_the_start_of_the_line_and_only_its_name(text):
    assert completions(CommandCompleter(COMMANDS), text) == []


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("")
    (tmp_path / "src" / "lib" / "util.py").write_text("")
    (tmp_path / "README.md").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env.example").write_text("")
    return tmp_path


def test_an_at_sign_completes_a_path_in_the_project(project):
    found = completions(MentionCompleter(str(project)), "fix @sr")
    assert [(c.text, c.start_position) for c in found] == [("src/", -2)]


def test_completion_goes_on_inside_a_directory(project):
    found = completions(MentionCompleter(str(project)), "look at @src/")
    assert [(c.text, c.start_position) for c in found] == [("app.py", 0), ("lib/", 0)]


def test_a_partly_typed_name_inside_a_directory_is_completed(project):
    found = completions(MentionCompleter(str(project)), "@src/lib/ut")
    assert [(c.text, c.start_position) for c in found] == [("util.py", -2)]


def test_hidden_entries_are_offered_only_to_someone_who_types_the_dot(project):
    completer = MentionCompleter(str(project))
    assert [c.text for c in completions(completer, "@")] == ["README.md", "src/"]
    assert [c.text for c in completions(completer, "@.")] == [".env.example", ".git/"]


def test_an_entry_that_cannot_be_examined_is_offered_as_a_file_and_does_not_stop_completion(
    tmp_path,
):
    # A symlink that points at itself cannot be stat'ed: asking whether it is a directory
    # raises, and an exception out of a completer is an error on the person's screen.
    (tmp_path / "loop").symlink_to("loop")
    (tmp_path / "next.py").write_text("")
    found = completions(MentionCompleter(str(tmp_path)), "@")
    assert [c.text for c in found] == ["loop", "next.py"]


@pytest.mark.parametrize(
    "text", ["fix src/ap", "me@sr", "@../", "@/etc/", "@src/../", "@~/", "@nope/x", "@README.md/x"]
)
def test_only_a_mention_of_something_in_the_project_is_completed(project, text):
    assert completions(MentionCompleter(str(project)), text) == []


@pytest.mark.parametrize(
    ("typing", "completed"),
    [("/hel", "/help"), ("see @READ", "see @README.md"), ("see @sr", "see @src/")],
)
async def test_tab_at_the_prompt_takes_the_first_offer(keyboard, project, typing, completed):
    # Completion runs in the background, so the line is submitted once it has landed:
    # keys sent in one burst would submit it before the offer arrived.
    session = session_on(keyboard, project)
    prompt = asyncio.create_task(session.prompt_async("> "))
    keyboard.send_text(typing + "\t")
    await until(lambda: session.default_buffer.text == completed, f"{typing!r} to complete")
    keyboard.send_text("\r")
    assert await prompt == completed


# --------------------------------------------------------------------------
# Discarding what was typed before a question
# --------------------------------------------------------------------------


def readable(fd: int) -> bool:
    return bool(select.select([fd], [], [], 0)[0])


def test_what_was_typed_and_not_yet_read_on_a_terminal_is_thrown_away():
    master, slave = pty.openpty()
    stream = os.fdopen(slave, "rb", buffering=0, closefd=False)
    try:
        os.write(master, b"y\n")  # typed while nobody was reading
        assert readable(slave)
        discard_pending_input(stream)
        assert not readable(slave)
        os.write(master, b"n\n")  # and what is typed afterwards is still read
        assert readable(slave)
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def test_where_there_is_no_termios_nothing_is_discarded_and_nothing_fails(monkeypatch):
    # Not every platform has it, and a question is still asked there.
    monkeypatch.setitem(sys.modules, "termios", None)
    master, slave = pty.openpty()
    stream = os.fdopen(slave, "rb", buffering=0, closefd=False)
    try:
        os.write(master, b"y\n")
        discard_pending_input(stream)
        assert readable(slave)
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def test_what_is_waiting_on_something_that_is_not_a_terminal_is_left_alone():
    read_end, write_end = os.pipe()
    stream = os.fdopen(read_end, "rb", buffering=0, closefd=False)
    try:
        os.write(write_end, b"data\n")
        discard_pending_input(stream)
        assert readable(read_end)
    finally:
        stream.close()
        os.close(read_end)
        os.close(write_end)


class TerminalThatCannotBeFlushed:
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("not a real descriptor")


def test_a_terminal_that_cannot_be_flushed_is_not_an_error():
    discard_pending_input(TerminalThatCannotBeFlushed())


def test_there_is_nothing_to_discard_without_a_standard_input(monkeypatch):
    monkeypatch.setattr("sys.stdin", None)
    discard_pending_input()
