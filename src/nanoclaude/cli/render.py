# src/nanoclaude/cli/render.py
"""Turning what the agent does into something readable.

The confirmation prompt is the one piece of UI that has to be right. Somebody
approving a write needs to see what will change, in the two seconds they will
actually spend on it, so an Edit shows its diff before it is applied, a command is
shown whole and not summarised, and every refusal names the rule that fired:
"refused" on its own teaches people to stop reading.

Almost everything printed here was not written by this code: tool names, arguments
and paths, tool output, what the model said, a provider's error, a model name from a
config file. It is data, and two things can go wrong when data is printed.

* **Markup.** Rich reads ``[...]`` in every string it prints and every table cell, so a
  route segment such as ``src/app/[id]/page.tsx`` loses its ``[id]`` and a ``[/]`` raises.
  Outside text therefore goes in as a :class:`~rich.text.Text`, which is never parsed,
  and is never put inside a markup template.
* **Control sequences.** A terminal acts on what it is sent: an escape sequence can clear
  the screen, retitle the window or rewrite the line above. Rich removes a few controls
  and passes the escape through. Text that only informs (model prose, tool output, an
  error) goes through :func:`~nanoclaude.tools.base.sanitize`. Text a person is deciding
  about (the request a confirmation is for, the call's name and arguments) is not
  stripped, because a prompt that quietly shows less than will run is no better than one
  that obeys it: each control character is shown by name, ``\\x1b`` and ``\\r``, by
  :func:`visible`.

Colour is whatever the console allows. This module never writes an escape sequence of
its own, so ``NO_COLOR`` and a console that is not a terminal both give plain text.

A reply that takes tens of seconds is shown as it arrives (spec 8): a waiting indicator that
names the model, then the text so far as Markdown in a live region that is drawn again at most
ten times a second, and left on the screen when the request ends. Nothing live is drawn on a
console that is not a terminal, that cannot move the cursor, or that has ``NO_COLOR`` set: the
text is printed once, when the request ends. A question to the person is never asked with
anything live on the screen. While a request is on the screen the terminal's own echo is off:
a key typed ahead, or the ``^C`` of Ctrl+C, would be printed at the cursor and throw the
display off by a line, which leaves a copy of its last frame behind.
"""

from __future__ import annotations

import asyncio
import os
import re
import unicodedata
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import ClassVar

from markdown_it import MarkdownIt
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import ImageItem, Markdown, MarkdownElement
from rich.segment import Segment
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from nanoclaude.agent.router import Router
from nanoclaude.agent.session import Session
from nanoclaude.agent.ui import Approval
from nanoclaude.cli.prompt import Questioner, discard_pending_input, new_prompter, silence_echo
from nanoclaude.config.schema import ROLES
from nanoclaude.conversation.transcript import TextBlock, ToolUseBlock
from nanoclaude.permissions.policy import Decision, PermissionRequest, PermissionResult
from nanoclaude.providers.base import ModelReply
from nanoclaude.providers.pricing import PriceBook
from nanoclaude.tools.base import ToolArgumentError, ToolOutcome, sanitize
from nanoclaude.tools.edit import EditError, preview_edit
from nanoclaude.tools.fs import FileTooLargeError
from nanoclaude.tools.write import preview_write

MAX_SUMMARY_CHARS = 100

#: How much of what a Write would write a confirmation shows: the first lines of a file it
#: creates, and the first lines of the diff when it replaces one. The rest is counted, not
#: left out without a word.
WRITE_PREVIEW_NEW_LINES = 20
WRITE_PREVIEW_DIFF_LINES = 40

#: How much of one line a preview shows. The cuts above bound how many lines there are and
#: not how wide: a file that is one line of two hundred thousand characters is one line of
#: the listing, two thousand rows on the screen, and the subject line is somewhere above them.
MAX_PREVIEW_LINE_CHARS = 200

#: The width of the bar ``/status`` draws for how much of the context window is in use.
BAR_WIDTH = 20

#: What a call is summarised by, most informative first.
_SUMMARY_KEYS = ("path", "command", "pattern", "subcommand")

#: The characters that can change what a person believes they are reading: the ones a
#: terminal acts on (category Cc: the C0 and C1 controls and DEL), the ones it draws as
#: nothing (Cf, "format": zero-width spaces and joiners, left-to-right and right-to-left
#: marks and overrides, the soft hyphen, the byte order mark), and a lone surrogate (Cs),
#: which cannot be printed at all. A subject that holds one is not the subject without it,
#: so each is shown by name and two different subjects never print alike.
_UNSAFE_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})

#: Where an unsafe character can be: anything but printable ASCII. Most text has none, and
#: is never looked at one character at a time.
_NOT_PLAIN_ASCII = re.compile(r"[^\x20-\x7e]")

_NAMED = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _spelled(char: str) -> str:
    """The escape that names ``char``: ``\\n``, ``\\x1b``, ``\\u200b``, ``\\U000e0001``."""
    named = _NAMED.get(char)
    if named is not None:
        return named
    code = ord(char)
    if code <= 0xFF:
        return f"\\x{code:02x}"
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"


def visible_text(raw: str, *, keep_newlines: bool = False) -> Text:
    """``raw`` with every control character shown as the escape that names it.

    The escapes are styled apart from the text around them, so that a real escape
    character can be told from the four characters ``\\x1b`` written out. With
    ``keep_newlines`` line breaks and tabs stay what they are, for text that is laid
    out over several lines (a diff) and not for one line (a path, a command): in a
    single line a newline would let one subject pose as two.
    """
    shown = Text()
    start = 0
    for match in _NOT_PLAIN_ASCII.finditer(raw):
        char = match.group()
        if unicodedata.category(char) not in _UNSAFE_CATEGORIES:
            continue
        if keep_newlines and char in "\n\t":
            continue
        shown.append(raw[start : match.start()])
        shown.append(_spelled(char), style="reverse")
        start = match.end()
    shown.append(raw[start:])
    return shown


def visible(raw: str, *, keep_newlines: bool = False) -> str:
    """:func:`visible_text`, as a plain string."""
    return visible_text(raw, keep_newlines=keep_newlines).plain


def plain(text: str, style: str = "") -> Text:
    """Outside text, stripped of terminal control sequences and never parsed as markup."""
    return Text(sanitize(text), style=style)


def show_notice(console: Console, message: str) -> None:
    """A line of information, dimmed. ``message`` is data: never read as markup."""
    console.print(plain(message, "dim"))


#: What every error line for a person begins with (spec 17.9).
ERROR_PREFIX = "error: "

#: A line break, and the blanks around it: what makes one line two.
_LINE_BREAKS = re.compile(r"[ \t]*\n\s*")


def show_error(console: Console, message: str) -> None:
    """``error: <what happened> — <what to do>``: spec 17.9's form for a person.

    ``message`` is what follows the prefix, or the whole line: either way the prefix is on
    the line exactly once, so no caller can print an error line without it, and it is one
    line: a line break in ``message`` becomes a space, so that nothing it quotes can start a
    line of its own, with a second ``error:`` in front of it, say. ``message`` is data:
    never read as markup.
    """
    body = _LINE_BREAKS.sub(" ", sanitize(message.removeprefix(ERROR_PREFIX))).strip()
    console.print(Text.assemble((ERROR_PREFIX, "red"), body))


def _shorten(text: str) -> str:
    if len(text) <= MAX_SUMMARY_CHARS:
        return text
    return text[: MAX_SUMMARY_CHARS - 3] + "..."


def _primary_argument(call: ToolUseBlock) -> str | None:
    """The argument that says most about the call, as the model wrote it."""
    for key in _SUMMARY_KEYS:
        if key in call.arguments:
            return str(call.arguments[key])
    return None


def summarise_call(call: ToolUseBlock) -> str:
    """One short line about what a call is for, with its control characters shown."""
    primary = _primary_argument(call)
    if primary is None:
        primary = ", ".join(sorted(call.arguments))
    return _shorten(visible(primary))


def _shown_line(line: str, style: str = "") -> Text:
    """One line of a preview, with its controls shown by name and cut where it is too wide.

    What was cut off is counted in a dim note after the line, so that nobody takes a line
    that goes on for what it shows. Tabs stay what they are. The cut is made on the line as it
    is in the file, before its controls are spelled out, so that no spelling is cut in two.
    """
    shown = visible_text(line[:MAX_PREVIEW_LINE_CHARS], keep_newlines=True)
    if style:
        shown.stylize(style)
    if len(line) > MAX_PREVIEW_LINE_CHARS:
        more = _plural(len(line) - MAX_PREVIEW_LINE_CHARS, "more character")
        shown.append(f"... ({more})", style="dim")
    return shown


def _diff_line_style(line: str, in_hunk: bool) -> str:
    """How a line of a unified diff is drawn, from what kind of line it is."""
    if line.startswith("@@"):
        return "cyan"
    if not in_hunk and line.startswith(("--- ", "+++ ")):
        return "bold"
    if in_hunk and line.startswith("+"):
        return "green"
    if in_hunk and line.startswith("-"):
        return "red"
    if in_hunk and line.startswith("\\"):
        return "dim"
    return ""


def render_diff(diff: str) -> RenderableType:
    """A unified diff, coloured by the kind of each line, with its controls shown by name.

    Drawn here and not by a syntax highlighter, which would hand a made-visible escape back
    to the terminal as plain text: in a diff that is about to be approved, ``\\x1b`` the
    escape character and ``\\x1b`` the four characters must not look alike, so the first is
    styled apart, as it is on a confirmation's subject line. A line that begins with ``--``
    inside a hunk is a removed line and not a file header, and is drawn as one. A line wider
    than ``MAX_PREVIEW_LINE_CHARS`` is cut, and says how much of it is not shown.
    """
    lines = diff.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    drawn: list[Text] = []
    in_hunk = False
    for line in lines:
        drawn.append(_shown_line(line, _diff_line_style(line, in_hunk)))
        in_hunk = in_hunk or line.startswith("@@")
    return Text("\n").join(drawn)


class _ImageWithItsAddress(ImageItem):
    """An image as text: that it is one, what it is said to show, and where it points.

    Rich draws a placeholder and the description, and with hyperlinks off nothing says where
    the image is. A link shows its address, and an image, which a terminal cannot show, is
    no less in need of one.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        described = self.text.plain.strip()
        yield Text(
            f"image: {described} ({self.destination})"
            if described
            else f"image: {self.destination}",
            end="",
        )


class _ModelMarkdown(Markdown):
    """Markdown as a model writes it, shown as it was written.

    Two things Rich leaves out are put back. HTML is text here, parsed as text and drawn
    as it was typed: Rich draws no HTML at all, so a reply that is a tag, or whose point is
    one, would print as nothing. An image shows its address (see
    :class:`_ImageWithItsAddress`). Code is as it was: nothing in it is HTML, and nothing in
    it is escaped.
    """

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "image": _ImageWithItsAddress,
    }

    def __init__(self, markup: str) -> None:
        super().__init__("", hyperlinks=False)
        self.markup = markup
        parser = MarkdownIt("commonmark", {"html": False}).enable("strikethrough").enable("table")
        self.parsed = parser.parse(markup)


def render_markdown(text: str) -> RenderableType:
    # Hyperlinks off: a terminal link shows its label and hides where it goes, and the
    # label is the model's to choose. Without them the address is printed beside it.
    return _ModelMarkdown(sanitize(text))


#: How often, at most, a reply that is arriving is drawn again. Drawn after every piece, a
#: reply written in hundreds of pieces would be parsed and painted as Markdown hundreds of
#: times a second, and every one of them would be a frame to redraw.
STREAM_REFRESH_PER_SECOND = 10


class _Arriving:
    """The text of a reply as far as it has arrived, drawn as Markdown whenever it is drawn.

    Pieces are added from the loop's thread and drawn from the live display's own, so they
    are kept as they come and joined and parsed only when drawn. Adding to a list and
    joining one are each safe to do from two threads.

    A reply taller than the screen is drawn as its last screenful while it grows. Left to
    itself Rich draws the top of it and hides the end, which is the part being written, and
    drawn without limit it leaves a copy of its top in the scrollback with every frame: the
    cursor cannot go back up past the first line of the screen to erase it. ``settle`` makes
    the next frame draw all of the reply, which is the one the display leaves behind.
    """

    def __init__(self) -> None:
        self._pieces: list[str] = []
        self._settled = False

    def add(self, piece: str) -> None:
        self._pieces.append(piece)

    def settle(self) -> None:
        self._settled = True

    @property
    def text(self) -> str:
        return "".join(self._pieces)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        drawn = render_markdown(self.text.strip())
        if self._settled:
            yield drawn
            return
        # One line short of the screen, so that drawing it scrolls nothing.
        lines = console.render_lines(drawn, options, pad=False)[-max(1, options.max_height - 1) :]
        for number, line in enumerate(lines):
            if number:
                yield Segment.line()
            yield from line


def _call_line(call: ToolUseBlock) -> Text:
    """A call as one collapsed line: its name, and what it is about."""
    parts: list[str | tuple[str, str]] = [(visible(call.name), "dim")]
    summary = summarise_call(call)
    if summary:
        parts.extend([" ", summary])
    return Text.assemble(*parts)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _inside(root: str, path: str) -> str | None:
    """``path`` relative to ``root``, or None when it is not inside it."""
    try:
        return str(PurePosixPath(path).relative_to(root))
    except ValueError:
        return None


def _lines_of(content: str) -> list[str]:
    """``content`` as its lines: a final newline ends the last line and does not start another."""
    lines = content.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def _refusal_line(result: PermissionResult) -> Text:
    """Spec 17.9's form, built from the decision: ``refused (<rule>): <reason>``.

    Not the string the model is told, which is capitalised and written for it.
    """
    return Text.assemble((f"refused ({sanitize(result.rule)}): ", "red"), sanitize(result.reason))


class ConsoleUI:
    """The interactive implementation of the UI protocol."""

    def __init__(
        self,
        console: Console,
        *,
        auto_approve: bool = False,
        prompter: Questioner | None = None,
        root: str | None = None,
        discard_input: Callable[[], None] | None = None,
        model_of: Callable[[str], str] | None = None,
        silence: Callable[[], Callable[[], None]] | None = None,
    ) -> None:
        self._console = console
        self._auto = auto_approve
        # What the waiting indicator calls the model that answers a role. The UI is handed a
        # role and not a model, and which model plays a role changes with /model, so it asks
        # when a request starts. Without it the indicator names the role.
        self._model_of = model_of
        # What stops the terminal echoing what is typed while a live display is on the screen,
        # and returns what puts the echo back: the echo moves the cursor the display counts
        # its lines from.
        self._silence = silence if silence is not None else silence_echo
        self._put_echo_back: Callable[[], None] | None = None
        # The project, so that a path inside it can be shown relative to it: "src/a.py" is
        # shorter than the resolved path and says the same. Resolved, like the paths of a
        # request are. Without it a path is shown whole.
        self._root = os.path.realpath(root) if root else None
        # What clears the terminal's unread input before a question is asked.
        self._discard_input = discard_input if discard_input is not None else discard_pending_input
        # Built when the first question is asked: a UI that is never asked anything has
        # no reason to take hold of the terminal.
        self._prompter: Questioner | None = prompter
        # Edits whose diff was shown at the confirmation, so that it is not shown again.
        self._previewed: set[str] = set()
        # What is on the screen for the request that is running: the indicator while it
        # waits, and a live region once its text arrives. Never both.
        self._waiting: Live | None = None
        self._live: Live | None = None
        # The text of the request that is running, not parsed until it is drawn.
        self._arriving: _Arriving | None = None
        # Whether the last request showed its text, so that the reply that follows it is not
        # shown as well. Taken by that reply, and cleared when the next request starts.
        self._streamed = False

    async def confirm(
        self, call: ToolUseBlock, request: PermissionRequest, _result: PermissionResult
    ) -> Approval:
        # The session ends a request before it runs a tool, so this is not needed. It is here
        # because a question asked over a live display cannot be read, and a rule that is only
        # kept by whoever calls is a rule that gets broken.
        self.on_request_end()
        if self._auto:
            # The question is not asked, so the confirmation names the call: on_decision
            # stays quiet for a call that is asked about, and it would be named twice.
            self._console.print(self._announcement(call, request))
            return Approval.ONCE
        self._console.print()
        self._console.print(
            Text.assemble(
                (visible(call.name), "bold yellow"),
                "  ",
                visible_text(self._subject_shown(request)),
            )
        )
        called_with = self._spelling_beside(call, request)
        if called_with is not None:
            # The policy judged the resolved path, and that is what is shown above. What
            # the model wrote may differ from it, and a person is owed both.
            self._console.print(Text.assemble(("  called with ", "dim"), visible_text(called_with)))
        if call.name == "Edit":
            await self._preview_edit(call, request)
        elif call.name == "Write":
            await self._preview_write(call, request)
        question = (
            f"  allow? [y]es / [n]o / [a]lways (every {visible(call.name)} call this session) "
        )
        while True:
            # Whatever was typed before now was not typed for this question: not what the
            # terminal has not delivered yet, and not what the prompt has read and kept.
            self._discard_input()
            prompter = self._prompt()
            prompter.forget_typed_ahead()
            try:
                answer = await prompter.prompt_async(question)
            except EOFError:
                return Approval.NO
            except KeyboardInterrupt:
                # Ctrl+C means stop, not no: a no goes back to the model, which answers it
                # and costs another round. And a KeyboardInterrupt raised inside the turn's
                # task would take the whole event loop with it, so it becomes the
                # cancellation that the REPL already knows how to take.
                raise asyncio.CancelledError from None
            choice = answer.strip().lower()
            if choice in ("y", "yes"):
                return Approval.ONCE
            if choice in ("a", "always"):
                return Approval.ALWAYS
            if choice in ("", "n", "no"):
                return Approval.NO
            self._console.print(plain("answer y, n or a", "dim"))

    def _prompt(self) -> Questioner:
        if self._prompter is None:
            self._prompter = new_prompter()
        return self._prompter

    def _subject_shown(self, request: PermissionRequest) -> str:
        """What the request is about, as it is shown: a resolved path inside the root is
        shown relative to it, anything else, a command or a path outside it, as it is."""
        subject = request.subject
        if self._root is not None and subject in request.resolved_paths:
            relative = _inside(self._root, subject)
            if relative is not None:
                return relative
        return subject

    def _spelling_beside(self, call: ToolUseBlock, request: PermissionRequest) -> str | None:
        """What the model called the subject, when that is not what is shown for it."""
        called_with = _primary_argument(call)
        if called_with is None:
            return None
        shown = (visible(self._subject_shown(request)), visible(request.subject))
        return None if visible(called_with) in shown else called_with

    def _announcement(self, call: ToolUseBlock, request: PermissionRequest) -> Text:
        """A call as one collapsed line: its name, and what it is about.

        For a write that is the path the policy judged and the tool will write, with what
        the model called it beside it when that differs: the model's own spelling can name
        a link, or a path with ``..`` in it, for a file somewhere else. Anything else is
        named by its most informative argument.
        """
        if not (request.is_write and request.subject in request.resolved_paths):
            return _call_line(call)
        parts: list[str | tuple[str, str] | Text] = [
            (visible(call.name), "dim"),
            " ",
            visible_text(self._subject_shown(request)),
        ]
        called_with = self._spelling_beside(call, request)
        if called_with is not None:
            parts.extend([(" (called with ", "dim"), visible_text(called_with), (")", "dim")])
        return Text.assemble(*parts)

    async def _preview_write(self, call: ToolUseBlock, request: PermissionRequest) -> None:
        """Show what a Write would write, before anything is written.

        The diff, when it replaces a file it can compare with. The first lines, when it
        creates one, or replaces one that cannot be compared (and why not). How much more
        there is is said, so that nobody approves what they have not been told is there.
        """
        if not request.resolved_paths:
            return
        path = request.resolved_paths[0]
        label = _primary_argument(call) or path
        try:
            preview = await asyncio.to_thread(preview_write, path, label, call.arguments)
        except (ToolArgumentError, OSError) as exc:
            self._console.print(Text.assemble(("  no preview: ", "dim"), visible_text(str(exc))))
            return
        totals = f"{_plural(preview.lines, 'line')}, {_plural(preview.size, 'byte')}"
        if preview.kind == "unchanged":
            self._console.print(
                plain("  the file already holds this content: nothing will change", "dim")
            )
        elif preview.kind == "replace":
            diff = _lines_of(preview.diff)
            # What the cut below leaves out is counted by its size too: a rewrite shows its
            # removed lines first, and the person should know how many are added.
            hunks = diff[2:]  # past the two file header lines
            added = sum(line.startswith("+") for line in hunks)
            removed = sum(line.startswith("-") for line in hunks)
            self._console.print(
                plain(f"  replaces the file: {totals} after the write (+{added} -{removed})", "dim")
            )
            self._console.print(render_diff("\n".join(diff[:WRITE_PREVIEW_DIFF_LINES])))
            self._say_how_many_more(len(diff) - WRITE_PREVIEW_DIFF_LINES, WRITE_PREVIEW_DIFF_LINES)
        else:
            if preview.kind == "create":
                self._console.print(plain(f"  new file: {totals}", "dim"))
            else:
                self._console.print(
                    Text.assemble(("  no diff: ", "dim"), visible_text(preview.why))
                )
                self._console.print(plain(f"  will write: {totals}", "dim"))
            lines = _lines_of(preview.content)
            for number, line in enumerate(lines[:WRITE_PREVIEW_NEW_LINES], start=1):
                # The file's own text: shown by name where it holds a control, with its tabs
                # as they are, and cut where it is very wide.
                self._console.print(Text.assemble((f"  {number:>3}  ", "dim"), _shown_line(line)))
            self._say_how_many_more(len(lines) - WRITE_PREVIEW_NEW_LINES, WRITE_PREVIEW_NEW_LINES)
            # As a diff says it, and only where the end of the file is on the screen.
            ends_bare = preview.content != "" and not preview.content.endswith("\n")
            if ends_bare and len(lines) <= WRITE_PREVIEW_NEW_LINES:
                self._console.print(plain("  \\ No newline at end of file", "dim"))

    def _say_how_many_more(self, more: int, shown: int) -> None:
        if more > 0:
            self._console.print(
                plain(f"  ... {_plural(more, 'more line')} (showing the first {shown})", "dim")
            )

    async def _preview_edit(self, call: ToolUseBlock, request: PermissionRequest) -> None:
        """Show what an Edit would change, before anything is written."""
        if not request.resolved_paths:
            return
        path = request.resolved_paths[0]
        label = _primary_argument(call) or path
        try:
            diff = await asyncio.to_thread(preview_edit, path, label, call.arguments)
        except (ToolArgumentError, EditError, OSError) as exc:
            # A file that is too large has a reason for a person, and a message for the model.
            why = exc.reason if isinstance(exc, FileTooLargeError) else str(exc)
            self._console.print(Text.assemble(("  no preview: ", "dim"), visible_text(why)))
            return
        # The new text is the model's and is going into a file: its controls are shown, not
        # removed, so that nobody approves a cleaner edit than the one that is written.
        self._console.print(render_diff(diff))
        self._previewed.add(call.id)

    def on_reply(self, reply: ModelReply) -> None:
        streamed, self._streamed = self._streamed, False
        if streamed:
            return  # its text is on the screen already, once
        for block in reply.blocks:
            if isinstance(block, TextBlock) and block.text.strip():
                self._console.print(render_markdown(block.text.strip()))

    def _animated(self) -> bool:
        """Whether this console can draw over what it drew, and may.

        A terminal that can move the cursor, with no ``NO_COLOR`` asking for plain output:
        anywhere else a live display is a stream of escape sequences, or lines repeated.
        """
        console = self._console
        return console.is_terminal and not console.is_dumb_terminal and not console.no_color

    def _display(self, renderable: RenderableType, *, transient: bool) -> Live:
        """A live display of ``renderable`` on this console, started and drawn at once."""
        live = Live(
            renderable,
            console=self._console,
            refresh_per_second=STREAM_REFRESH_PER_SECOND,
            transient=transient,
            # Rich would otherwise swap sys.stdout and sys.stderr for as long as it runs.
            # Everything this UI writes goes through its console, and two process-wide
            # streams are state that a request that fails must not be able to leave behind.
            redirect_stdout=False,
            redirect_stderr=False,
        )
        live.start(refresh=True)
        return live

    def _waiting_for(self, role: str) -> Text:
        """What the indicator says: the model that is answering, and for what if not the main."""
        if self._model_of is None:
            return plain(f"waiting for the {role} model", "dim")
        model = self._model_of(role)
        return plain(f"waiting for {model}" + ("" if role == "main" else f" ({role})"), "dim")

    def _spinner(self, role: str) -> Spinner:
        """The indicator: braille dots where the terminal can write them, and ASCII where not.

        Its first frame is the first thing a request writes, and a terminal whose encoding has
        no braille would raise at once. The request would then fail with an encoding error
        where, before there was an indicator, its reply came out in any encoding it could be
        written in.
        """
        braille = self._console.encoding.startswith("utf")
        return Spinner("dots" if braille else "line", text=self._waiting_for(role))

    def on_request_start(self, role: str) -> None:
        self.on_request_end()  # a request that never ended is over now
        self._streamed = False
        if self._animated():
            restore = self._silence()
            self._put_echo_back = restore
            try:
                self._waiting = self._display(self._spinner(role), transient=True)
            except BaseException:
                # The display could not be started: nothing is on the screen to throw off,
                # and the person must be able to see what they type.
                self._put_echo_back = None
                restore()
                raise

    def on_text(self, delta: str) -> None:
        if self._arriving is None:
            self._arriving = _Arriving()
        self._arriving.add(delta)
        if self._streamed or not delta.strip():
            return  # blank lines the model opens with are nothing to read yet
        self._streamed = True
        if not self._animated():
            return  # printed once, when the request ends
        if self._waiting is not None:
            self._waiting.stop()
            self._waiting = None
        self._live = self._display(self._arriving, transient=False)

    def on_request_end(self) -> None:
        waiting, self._waiting = self._waiting, None
        live, self._live = self._live, None
        arriving, self._arriving = self._arriving, None
        put_echo_back, self._put_echo_back = self._put_echo_back, None
        # Each thing that has to be undone is undone whatever the one before it did: a
        # display that fails to stop must not leave the text unprinted, nor the echo off.
        # The echo goes back last, once nothing is drawing, and in a finally of its own.
        try:
            try:
                if waiting is not None:
                    waiting.stop()  # transient: its line is erased
            finally:
                if live is not None:
                    if arriving is not None:
                        arriving.settle()
                    live.stop()  # draws all of the reply, once, and leaves it there
                elif arriving is not None and self._streamed:
                    self._console.print(render_markdown(arriving.text.strip()))
        finally:
            if put_echo_back is not None:
                put_echo_back()

    def on_decision(
        self, call: ToolUseBlock, request: PermissionRequest, result: PermissionResult
    ) -> None:
        if result.decision is Decision.DENY:
            # A refused call never reaches on_outcome, so this line is everything the
            # person learns about it.
            self._console.print(_refusal_line(result))
        elif result.decision is Decision.ALLOW:
            self._console.print(self._announcement(call, request))
        # A call that is to be asked about is announced by the confirmation, in full.

    def on_outcome(self, call: ToolUseBlock, outcome: ToolOutcome) -> None:
        previewed = call.id in self._previewed
        self._previewed.discard(call.id)
        if outcome.is_error:
            lines = sanitize(outcome.content).splitlines()
            self._console.print(Text.assemble(("  ! ", "red"), lines[0] if lines else ""))
        elif call.name == "TodoWrite":
            # The tool's own output is the list, written for the person to read: it is how
            # they see what the model thinks it is doing without reading the calls.
            self._console.print(plain(outcome.content))
        elif call.name == "Edit" and not previewed:
            # No confirmation showed it (accept-edits mode), so this is the one place the
            # person sees what changed. The tool reports "Edited <file>" and then the diff.
            _, _, diff = outcome.content.partition("\n")
            if diff.strip():
                self._console.print(render_diff(diff))

    def on_output(self, text: str) -> None:
        self._console.print(plain(text))

    def on_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        self._console.print(
            Text.assemble(
                (f"provider busy, retry {attempt} in {delay_s:.1f}s ", "yellow"),
                (sanitize(reason)[:80], "dim"),
            )
        )


def context_bar(used: int, available: int) -> str:
    """How full the context window is, as ``[####....]``. Empty when the size is not known."""
    filled = int(BAR_WIDTH * min(1.0, used / available)) if available > 0 else 0
    return f"[{'#' * filled}{'.' * (BAR_WIDTH - filled)}]"


def status_panel(session: Session, used: int, available: int) -> RenderableType:
    """Where the session stands, on one screen.

    ``used`` and ``available`` are what ``Session.context_usage()`` returns. That is a
    coroutine, so the caller awaits it and this stays a function of its arguments. Which
    model plays each role is read from the router, which is what decides.
    """
    config = session.router.config
    table = Table(title="status", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    for role in ROLES:
        alias = config.roles.alias_for(role)
        table.add_row(role, plain(f"{config.models[alias].model} ({alias})"))
    table.add_row("", "")
    table.add_row("mode", plain(str(session.policy.mode)))
    table.add_row("sandbox", plain(", ".join(session.policy.sandbox.roots)))
    table.add_row("tools", str(len(session.registry)))
    table.add_row("session", plain(session.session_id))
    table.add_row("turns", f"{session.state.turn}/{session.config.limits.max_turns}")
    table.add_row("context", Text(f"{context_bar(used, available)} {used}/{available} tokens"))
    return table


def _dollars(cost: float | None) -> str:
    """A cost, or a dash for one that is not known: zero would say the model is free."""
    return "-" if cost is None else f"${cost:.4f}"


def cost_panel(router: Router, prices: PriceBook) -> RenderableType:
    """What this run has used and spent, by role.

    "This run": a resumed session's stored row holds the running total, but the router
    only knows what was spent since this process started.
    """
    table = Table(title="cost (this run)", show_header=True, header_style="bold")
    # In a narrow terminal the model name gives way first: a role and its numbers say
    # what was spent, and a cost of "$4.…" reads as a different amount.
    table.add_column("role", no_wrap=True)
    table.add_column("model")
    for column in ("in", "out", "cached", "cost"):
        table.add_column(column, justify="right", no_wrap=True)
    for role, entry in router.by_role().items():
        table.add_row(
            role,
            plain(entry.model),
            str(entry.usage.input_tokens),
            str(entry.usage.output_tokens),
            str(entry.usage.cache_read_tokens),
            _dollars(entry.cost),
        )
    table.add_row("total", "", "", "", "", _dollars(router.total_cost()), style="bold")
    if prices.is_stale():
        table.caption = f"price table last checked {prices.last_updated}; it may be out of date"
    return table
