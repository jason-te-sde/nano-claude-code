"""What a person sees on a terminal after some output was written to it.

The streaming tests are about the screen, not about the bytes. A live display draws its
text again for every frame, so "the reply appears once" is true of the screen and false
of the output, which holds every frame. This lays the output out as a terminal does, for
the few sequences Rich draws a live display with: carriage return, line feed (taken as
carriage return and line feed, as a terminal in its usual mode does), erase line, cursor up,
and the sequences that change nothing on the screen (styling, and hiding or showing the
cursor).

It is strict on purpose. A sequence it does not model, a screen clear say, raises: skipped,
it would leave on the "screen" the text it was meant to clear, and the test would pass for
the wrong reason.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\x1b\[(?P<args>[0-9;?]*)(?P<final>[ -/]*[@-~])|(?P<char>.)", re.DOTALL)
_CURSOR_VISIBILITY = re.compile(r"\x1b\[\?25([lh])")

#: What a live display writes of its own: styling, erasing a line, going up a line, hiding
#: and showing the cursor, and returning to the start of the line.
_OWN = re.compile(r"\x1b\[[0-9;]*m|\x1b\[2K|\x1b\[[0-9]*A|\x1b\[\?25[lh]|\r")

#: Every control a terminal can act on, but the line feed: the C0 controls, DEL and the C1 controls.
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def on_screen(output: str, height: int | None = None) -> str:
    """The lines a terminal would show after ``output``, trailing blanks left off.

    A line feed at the end leaves the cursor on a line of its own, and that empty line is
    in the result: ``"ab\\n"`` is ``"ab\\n"``, and ``"ab"`` is ``"ab"``. Empty lines below
    the cursor are not: a display that erased itself and went back up shows nothing.

    With a ``height`` the terminal is that many lines tall and scrolls, and the cursor cannot
    go above its top line, as on a real one. The result is then what scrolled off followed by
    what is on the screen, so that a line shown twice shows twice: a display taller than the
    screen cannot erase the lines it has pushed off the top, and leaves copies of them.
    """
    rows = [""]
    row = col = top = 0
    for token in _TOKEN.finditer(output):
        char = token.group("char")
        if char is not None:
            if char == "\r":
                col = 0
            elif char == "\n":
                row, col = row + 1, 0
                if row == len(rows):
                    rows.append("")
                if height is not None and row - top >= height:
                    top = row - height + 1
            elif char < " " or char == "\x7f":
                raise ValueError(f"the screen helper does not model the control {char!r}")
            else:
                line = rows[row].ljust(col)
                rows[row] = line[:col] + char + line[col + 1 :]
                col += 1
            continue
        final, args = token.group("final"), token.group("args")
        if final == "m" or (final in ("l", "h") and args == "?25"):
            continue  # styling; the cursor shown or hidden
        if final == "K" and args == "2":
            rows[row] = ""
        elif final == "A":
            row = max(top, row - int(args or 1))
        else:
            raise ValueError(
                f"the screen helper does not model the escape sequence {token.group(0)!r}"
            )
    while len(rows) - 1 > row and not rows[-1]:
        rows.pop()
    return "\n".join(line.rstrip() for line in rows)


def cursor_is_hidden(output: str) -> bool:
    """Whether ``output`` leaves the cursor hidden: it was hidden, and not shown again."""
    changes = _CURSOR_VISIBILITY.findall(output)
    return bool(changes) and changes[-1] == "l"


def foreign_controls(output: str) -> list[str]:
    """The controls in ``output`` that a live display does not write itself.

    Each with what follows it. Text that arrives from outside can hold an escape sequence
    that clears the screen or retitles the window, and a live display draws it again with
    every frame. What a display draws for itself is taken out first, so that what is left
    is only what somebody else put there.
    """
    rest = _OWN.sub("", output)
    return [rest[found.start() : found.start() + 12] for found in _CONTROL.finditer(rest)]
