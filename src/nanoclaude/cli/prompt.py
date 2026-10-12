# src/nanoclaude/cli/prompt.py
"""The input layer: what a person types into.

Everything here is about reading a line from a terminal. What to do with the line is
the REPL's business and how to show the answer is the renderer's, so this module
imports neither.

A line is read by prompt_toolkit, because an agent people live in for hours needs real
editing, history and completion, and ``input()`` gets abandoned within a day. What it
gives for free (history search with Ctrl+R, bracketed paste, emacs keys) is left as it
is. What this module adds is what spec 8 asks for on top:

* several lines, by ``Esc`` then ``Enter`` or by ending a line with a backslash;
* ``/`` completes a slash command, and ``@`` completes a path in the project, which is
  the syntax a mention is written in (see ``nanoclaude.context.mentions``);
* vi keys, chosen by ``NANOCLAUDE_EDITING_MODE=vi``;
* history kept between sessions.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, merge_completers
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, emacs_insert_mode, emacs_mode, vi_insert_mode
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.input.typeahead import clear_typeahead
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.output import Output

from nanoclaude.private import (
    create_private_file,
    make_private_directories,
    narrow_directory,
    narrow_file,
)

#: The environment variable that picks vi keys: ``vi`` (or ``vim``), anything else is emacs.
EDITING_MODE_VARIABLE = "NANOCLAUDE_EDITING_MODE"


class Prompter(Protocol):
    """Something that asks for a line and waits for it.

    ``prompt_toolkit.PromptSession`` is one. The REPL and the confirmation prompt ask
    for a ``Prompter`` and not for a session, so a test can answer them from a script,
    and neither depends on how the line is read.

    ``EOFError`` means the person ended their input (Ctrl+D) and ``KeyboardInterrupt``
    that they pressed Ctrl+C, as they do for ``input()``.
    """

    async def prompt_async(self, message: str = "") -> str: ...


class Questioner(Prompter, Protocol):
    """A :class:`Prompter` for a question asked in the middle of a turn.

    What a person typed before a question was asked is not its answer, and some of it is
    out of reach of the terminal by then: a prompt reads what is waiting in one chunk, and
    what it did not use waits for the prompt that comes next. ``forget_typed_ahead`` drops it.
    """

    def forget_typed_ahead(self) -> None: ...


class _Question:
    """A prompt for a question asked in the middle of a turn, which leaves SIGINT alone.

    While a prompt is up, prompt_toolkit installs a SIGINT handler of its own on the event
    loop, and when the prompt closes it removes whatever handler is there. The REPL has
    one installed for the turn, so a question asked in the middle of a turn would end with
    it gone, and Ctrl+C would do nothing for the rest of that turn. Ctrl+C at the question
    itself still works: the terminal is in raw mode there, and the key reaches the prompt
    as a key and not as a signal.
    """

    def __init__(self, session: PromptSession[str]) -> None:
        self._session = session

    async def prompt_async(self, message: str = "") -> str:
        return await self._session.prompt_async(message, handle_sigint=False)

    def forget_typed_ahead(self) -> None:
        # prompt_toolkit keeps it per input, and every prompt on the terminal's input shares
        # one store: the REPL's prompt leaves keys there, and this one would be fed them.
        clear_typeahead(self._session.input)


def new_prompter(*, input: Input | None = None, output: Output | None = None) -> Questioner:
    """A bare prompt on the terminal, for a question asked in the middle of a turn.

    It has no history and no completion, and it leaves SIGINT to the turn it is asked in.
    ``input`` and ``output`` are the terminal's unless given: a test hands it a pipe.
    """
    return _Question(PromptSession[str](input=input, output=output))


class _Descriptor(Protocol):
    """What the terminal helpers ask of a stream: whether it is a terminal, and where."""

    def isatty(self) -> bool: ...

    def fileno(self) -> int: ...


def discard_pending_input(stream: _Descriptor | None = None) -> None:
    """Throw away what was typed on the terminal and has not been read.

    Called before a question is asked in the middle of a turn. A person who types ``y`` and
    Enter while the model is still thinking has not answered a question nobody had asked,
    and the prompt would read it as the answer to the one that comes: a write approved
    without being seen. Only a terminal has anything to discard, so on anything else, a
    pipe, a file, no standard input at all, this does nothing. ``stream`` is standard input
    unless given.
    """
    try:
        import termios  # not on every platform: only where there is a terminal to flush
    except ImportError:
        return
    stream = sys.stdin if stream is None else stream
    try:
        if stream.isatty():
            termios.tcflush(stream.fileno(), termios.TCIFLUSH)
    except (AttributeError, OSError, ValueError, termios.error):
        return


def _nothing_to_put_back() -> None:
    """What ``silence_echo`` returns where it changed nothing."""


def silence_echo(stream: _Descriptor | None = None) -> Callable[[], None]:
    """Stop the terminal echoing what is typed, and return what puts the echo back.

    For as long as a live display is on the screen the echo is a second writer. A key typed
    ahead, or the ``^C`` of Ctrl+C, is printed at the cursor, and the cursor is at the right
    edge of the display's last line, so it lands on the line below. The display counts its
    lines from where it believes the cursor to be, so every frame after that erases one line
    too few and leaves a copy of itself behind: a cancelled reply is on the screen twice.

    Without the echo what is typed is still queued, for the next prompt to read, and Ctrl+C
    still signals. Only the printing stops. Only the echo is touched, and only the echo is put
    back: a setting somebody else changes in between is not written over, and a terminal that
    was already silent stays so. Only a terminal echoes, so on anything else, a pipe, a file,
    no standard input at all, this changes nothing and what it returns does nothing.
    ``stream`` is standard input unless given.
    """
    try:
        import termios  # not on every platform: only where there is a terminal to change
    except ImportError:
        return _nothing_to_put_back
    stream = sys.stdin if stream is None else stream
    try:
        if not stream.isatty():
            return _nothing_to_put_back
        fd = stream.fileno()
        attributes = termios.tcgetattr(fd)
        if not attributes[3] & termios.ECHO:
            return _nothing_to_put_back
        attributes[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, attributes)
    except (AttributeError, OSError, ValueError, termios.error):
        return _nothing_to_put_back

    def put_back() -> None:
        try:
            now = termios.tcgetattr(fd)
            now[3] |= termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, now)
        except (OSError, ValueError, termios.error):
            return  # the terminal has gone, and there is nothing left to put back

    return put_back


def wants_vi_mode(environ: Mapping[str, str]) -> bool:
    """True when ``NANOCLAUDE_EDITING_MODE`` asks for vi keys."""
    return environ.get(EDITING_MODE_VARIABLE, "").strip().lower() in ("vi", "vim")


class CommandCompleter(Completer):
    """Completes a slash command's name, at the start of the line and nowhere else.

    ``commands`` maps each name (without the slash) to a line saying what it does,
    which is shown beside the completion. It is a mapping and not the command table
    itself so that this module does not depend on the commands.
    """

    def __init__(self, commands: Mapping[str, str]) -> None:
        self._commands = dict(commands)

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,  # noqa: ARG002 - the base class's signature
    ) -> Iterator[Completion]:
        typed = document.text_before_cursor
        if not typed.startswith("/") or any(char.isspace() for char in typed):
            return
        for name in sorted(self._commands):
            if f"/{name}".startswith(typed):
                yield Completion(
                    f"/{name}", start_position=-len(typed), display_meta=self._commands[name]
                )


def _is_directory(entry: os.DirEntry[str]) -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


class MentionCompleter(Completer):
    """Completes ``@path``, the way a mention is written, relative to the project root.

    Only what a mention can name is offered: a path inside the project. An absolute
    path, ``~`` and ``..`` are not expanded by ``expand_mentions`` (they would leave the
    project, or are not resolved at all), and offering them would suggest otherwise.
    Entries that begin with a dot are offered only to someone who has typed one, so that
    ``@<Tab>`` is not a list of caches.
    """

    def __init__(self, root: str) -> None:
        self._root = root

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,  # noqa: ARG002 - the base class's signature
    ) -> Iterator[Completion]:
        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith("@"):
            return
        typed = word[1:]
        if typed.startswith(("/", "~")) or ".." in typed.split("/"):
            return
        folder, _, prefix = typed.rpartition("/")
        try:
            with os.scandir(Path(self._root) / folder) as listing:
                entries = sorted(listing, key=lambda entry: entry.name)
        except OSError:
            return
        for entry in entries:
            if not entry.name.startswith(prefix):
                continue
            if entry.name.startswith(".") and not prefix.startswith("."):
                continue
            name = entry.name + ("/" if _is_directory(entry) else "")
            yield Completion(name, start_position=-len(prefix))


def input_bindings() -> KeyBindings:
    """The keys spec 8 adds to prompt_toolkit's own: a second line without submitting."""
    bindings = KeyBindings()
    ends_with_backslash = Condition(
        lambda: get_app().current_buffer.document.text_before_cursor.endswith("\\")
    )

    # Emacs keys only: in vi, Esc leaves insert mode and Enter then submits, which is
    # what a person who chose vi expects Esc and Enter to do.
    @bindings.add("escape", "enter", filter=emacs_mode)
    def _another_line(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    # A line that ends in a backslash goes on, as in a shell. To send one that really
    # ends in a backslash, put a space after it.
    @bindings.add("enter", filter=(emacs_insert_mode | vi_insert_mode) & ends_with_backslash)
    def _continue_line(event: KeyPressEvent) -> None:
        event.current_buffer.delete_before_cursor(1)
        event.current_buffer.insert_text("\n")

    return bindings


def build_prompt_session(
    root: str,
    history_path: str | None = None,
    *,
    commands: Mapping[str, str] | None = None,
    vi_mode: bool = False,
    input: Input | None = None,
    output: Output | None = None,
) -> PromptSession[str]:
    """The prompt the REPL reads its lines from.

    ``root`` is the project, which ``@`` completes inside. ``history_path`` is a file to
    keep what was typed in, created with its directory when it is not there, and a
    history that lasts only as long as the session when it is not given. ``commands``
    are the names and one-line summaries ``/`` completes. ``input`` and ``output`` are
    the terminal's unless given.
    """
    history: History
    if history_path:
        # What was typed is as private as the conversation it was typed into. The library
        # opens the file with the default mode the first time something is stored, so it is
        # made here, private, and it only ever appends to a file that is already there.
        make_private_directories(Path(history_path).parent)
        create_private_file(Path(history_path))
        # A history an earlier version made with the default mode is as open as it was.
        narrow_directory(Path(history_path).parent)
        narrow_file(Path(history_path))
        history = FileHistory(history_path)
    else:
        history = InMemoryHistory()
    return PromptSession[str](
        history=history,
        completer=merge_completers([CommandCompleter(commands or {}), MentionCompleter(root)]),
        key_bindings=input_bindings(),
        enable_history_search=True,
        vi_mode=vi_mode,
        input=input,
        output=output,
    )
