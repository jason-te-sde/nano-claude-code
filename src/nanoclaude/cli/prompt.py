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
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, merge_completers
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, emacs_insert_mode, emacs_mode, vi_insert_mode
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.output import Output

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


def new_prompter(*, input: Input | None = None, output: Output | None = None) -> Prompter:
    """A bare prompt on the terminal, for a question asked in the middle of a turn.

    It has no history and no completion, and it leaves SIGINT to the turn it is asked in.
    ``input`` and ``output`` are the terminal's unless given: a test hands it a pipe.
    """
    return _Question(PromptSession[str](input=input, output=output))


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
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
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
