"""Doubles and helpers the CLI tests share."""

from __future__ import annotations

import io
import re
from collections.abc import Callable

from rich.console import Console, RenderableType

from nanoclaude.conversation.transcript import ToolUseBlock
from nanoclaude.permissions.policy import Decision, PermissionRequest, PermissionResult

#: What the policy answers for a call that has to be confirmed.
ASK = PermissionResult(Decision.ASK, "default.ask", "Edit needs confirmation")


def plain_console(width: int = 100) -> tuple[Console, io.StringIO]:
    """A console that is not a terminal and has no colour, and the text it has written."""
    buffer = io.StringIO()
    return Console(file=buffer, width=width, no_color=True, force_terminal=False), buffer


def terminal_console(width: int = 100) -> tuple[Console, io.StringIO]:
    """A console that believes it is a colour terminal, and the bytes it has written."""
    buffer = io.StringIO()
    console = Console(
        file=buffer, width=width, force_terminal=True, color_system="standard", legacy_windows=False
    )
    return console, buffer


def capture(renderable: RenderableType, width: int = 100) -> str:
    """What ``renderable`` prints on a plain console."""
    console, buffer = plain_console(width)
    console.print(renderable)
    return buffer.getvalue()


#: Every control a terminal acts on, other than styling: an escape that does not begin a
#: styling (SGR) sequence, and every other C0 control except tab and newline, DEL, and the
#: C1 controls (U+0080 to U+009F), which are the 8-bit forms of the escape sequences:
#: U+009B is a CSI and U+009D an OSC, and either of them clears the screen on its own.
#: The styling Rich adds, and line breaks, are all that output should ever contain.
_STRAY_ESCAPE = re.compile(r"\x1b(?!\[[0-9;]*m)|[\x00-\x08\x0b-\x1a\x1c-\x1f\x7f-\x9f]")

#: The colour parameters of every styling sequence in a piece of output.
_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_COLOUR_PARAMETER = re.compile(r"3[0-7]|4[0-7]|9[0-7]|10[0-7]|38|48")


def stray_escapes(output: str) -> list[str]:
    """The controls in ``output`` that are not styling, each with what follows it."""
    return [output[m.start() : m.start() + 12] for m in _STRAY_ESCAPE.finditer(output)]


def colour_codes(output: str) -> list[str]:
    """The colour parameters in ``output``: empty when it carries no colour at all."""
    found: list[str] = []
    for match in _SGR.finditer(output):
        found.extend(p for p in match.group(1).split(";") if _COLOUR_PARAMETER.fullmatch(p))
    return found


class ScriptedPrompter:
    """A prompt that answers from a script and remembers what it was asked.

    An answer that is an exception is raised instead of returned, which is how a test
    presses Ctrl+C or Ctrl+D. When the script runs out the prompt raises ``EOFError``,
    so a REPL driven by one ends by itself instead of waiting for a person.
    """

    def __init__(
        self, *answers: str | BaseException, on_ask: Callable[[str], None] | None = None
    ) -> None:
        self._answers = list(answers)
        #: Called with the message of each question as it is asked, before it is answered.
        self.on_ask = on_ask
        #: The message of every question asked, in order.
        self.asked: list[str] = []

    async def prompt_async(self, message: str = "") -> str:
        self.asked.append(message)
        if self.on_ask is not None:
            self.on_ask(message)
        if not self._answers:
            raise EOFError
        answer = self._answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeClock:
    """A clock that moves only when a test says so."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def edit_call(path: str, old: str, new: str, call_id: str = "e1") -> ToolUseBlock:
    return ToolUseBlock(
        call_id, "Edit", {"path": path, "edits": [{"old_string": old, "new_string": new}]}
    )


def write_request(tool: str, subject: str) -> PermissionRequest:
    """What an Edit, a Write or a Bash call asks the policy about."""
    if tool == "Bash":
        return PermissionRequest(tool, subject)
    return PermissionRequest(tool, subject, (subject,), is_write=True)
