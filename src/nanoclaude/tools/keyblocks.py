# src/nanoclaude/tools/keyblocks.py
"""Which lines of a file are inside a private key block, and what a window of them shows.

A private key is recognised as a block: from its ``-----BEGIN ... PRIVATE KEY-----`` line to
its ``-----END`` line, by the recogniser in ``permissions/redact.py``, which is the only one
there is and which this module only drives. The body of one is base64 and means nothing to
anything that looks at a line of it alone, so a
window of a file (what Read shows with an ``offset`` and a ``limit``, what Grep shows of a
match and the lines around it) that begins or ends inside a block, and is scrubbed on its
own, shows the key. The extent of a block is therefore worked out on the whole file, and the
window is cut afterwards: a line that is inside a block is shown masked.

Two entry points, for the two ways a file is looked at. :func:`mask_window` is Read's: the
whole text is in hand, and a window of it is shown with the part of the block in it masked.
:func:`masked_line_numbers` is Grep's: the file is not in hand, only the numbers of the lines
that are to be shown, so it is read as a stream, one line at a time, and what is kept is the
lines of a block that is open and the numbers of the lines that were inside one.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from nanoclaude.permissions.redact import (
    PRIVATE_KEY_BEGIN,
    PRIVATE_KEY_END,
    PRIVATE_KEY_MARKER,
    Redactor,
)

__all__ = ["PRIVATE_KEY_MARKER", "mask_window", "masked_line_numbers"]


def mask_window(
    redactor: Redactor, content: str, lines: Sequence[str], offset: int, limit: int
) -> list[tuple[int, str]]:
    """The lines ``offset`` to ``offset + limit - 1`` (1-based) of ``content``, as (number, text).

    ``lines`` is ``content.splitlines()``. A line that is inside a private key block, wherever
    the block began and ends in the file, is shown with the part of it that is inside the block
    replaced by :data:`PRIVATE_KEY_MARKER`: text before a ``BEGIN`` on its line and after an
    ``END`` on its line is kept. A run of lines that are wholly inside a block is one masked
    line, numbered as its first line, and the numbers go on after it, as they do when a block is
    wholly in the window. A window that starts inside a block shows its first line masked.
    """
    last = min(len(lines), offset - 1 + limit)
    spans = redactor.private_key_spans(content)
    if not spans:
        return [(number, lines[number - 1]) for number in range(offset, last + 1)]

    shown: list[tuple[int, str]] = []
    starts = _line_starts(content, last)
    carried = False  # the mask above is still going on at the end of the line before this one
    for number in range(offset, last + 1):
        line = lines[number - 1]
        start = starts[number - 1]
        end = start + len(line)
        covered = [
            (max(s, start) - start, min(e, end) - start, e > end)
            for s, e in spans
            if s <= end and e > start
        ]
        if not covered:
            shown.append((number, line))
            carried = False
            continue
        pieces: list[str] = []
        position = 0
        for index, (low, high, _) in enumerate(covered):
            pieces.append(line[position:low])
            if not (index == 0 and low == 0 and carried):
                pieces.append(PRIVATE_KEY_MARKER)
            position = high
        pieces.append(line[position:])
        text = "".join(pieces)
        if not (carried and covered[0][0] == 0 and text == ""):
            shown.append((number, text))
        carried = covered[-1][1] == len(line) and covered[-1][2]
    return shown


def _line_starts(content: str, count: int) -> list[int]:
    """Where each of the first ``count`` lines of ``content`` begins, as ``splitlines`` has them."""
    starts: list[int] = []
    position = 0
    for piece in content.splitlines(keepends=True)[:count]:
        starts.append(position)
        position += len(piece)
    return starts


#: Characters that end a line for ``str.splitlines`` and not for ripgrep, which numbers by ``\n``.
_OTHER_BREAKS = re.compile("[\r\x0b\x0c\x1c-\x1e\x85" + chr(0x2028) + chr(0x2029) + "]")


def masked_line_numbers(redactor: Redactor, path: str, *, upto: int) -> frozenset[int]:
    """The numbers of the lines of the file at ``path``, up to line ``upto``, that are inside a
    private key block, however far past ``upto`` the block runs.

    The file is read as a stream and is not held. Lines are numbered both ways the search
    backends number them (ripgrep by ``\\n``, the pure-Python fallback by ``str.splitlines``),
    and the numbers of both are returned: they are the same for any file that has no other
    line break in it, and for one that has, the lines of a block are in the set whichever
    backend found the matches (a line that is not in a block may be, as well; that only masks
    more). Nothing is found when ``redactor`` is off. A file that cannot be read now (it was
    searched a moment ago) has all of its lines up to ``upto`` reported, since what is not known
    is not shown.
    """
    if not redactor.enabled:
        return frozenset()
    try:
        return _numbers_in(redactor, path, upto)
    except OSError:
        return frozenset(range(1, upto + 1))


def _numbers_in(redactor: Redactor, path: str, upto: int) -> frozenset[int]:
    exotic = False

    def physical() -> Iterator[str]:
        nonlocal exotic
        for line in _physical_lines(path):
            if _OTHER_BREAKS.search(line.removesuffix("\r")):
                exotic = True
            yield line

    found = _scan(physical(), redactor, upto)
    if exotic:
        found |= _scan(_split_lines(path), redactor, upto)
    return frozenset(found)


def _physical_lines(path: str) -> Iterator[str]:
    """The lines of the file as ripgrep counts them: ended by ``\\n`` and by nothing else."""
    with Path(path).open("rb") as stream:
        for raw in stream:
            yield raw.decode("utf-8", errors="replace").removesuffix("\n")


def _split_lines(path: str) -> Iterator[str]:
    """The lines of the file as ``str.splitlines`` counts them, which is what the fallback does."""
    with Path(path).open("rb") as stream:
        for raw in stream:
            yield from raw.decode("utf-8", errors="replace").splitlines()


def _opens_a_block(line: str) -> bool:
    """Whether the last ``BEGIN`` on ``line`` has no ``END`` after it on the same line."""
    begins = list(PRIVATE_KEY_BEGIN.finditer(line))
    return bool(begins) and PRIVATE_KEY_END.search(line, begins[-1].end()) is None


def _scan(lines: Iterable[str], redactor: Redactor, upto: int) -> set[int]:
    """The numbers (1-based) of the lines in a private key block, reading no further than
    ``upto`` unless a block is open there, in which case to its end.

    A block is gathered from its ``BEGIN`` line until its ``END`` line, the next ``BEGIN`` line
    or the end of the file, and the recogniser is run over what was gathered: the same regular
    expression the scrubber masks with, so the extent is the one it would find in the whole
    file. What is held is the one block, and never the file.
    """
    masked: set[int] = set()
    pending: list[str] = []
    first = 0

    def settle() -> None:
        text = "\n".join(pending)
        starts: list[int] = []
        position = 0
        for line in pending:
            starts.append(position)
            position += len(line) + 1
        for begin, end in redactor.private_key_spans(text):
            low = bisect_right(starts, begin) - 1
            high = bisect_right(starts, max(end - 1, begin)) - 1
            masked.update(range(first + low, first + high + 1))
        pending.clear()

    for number, line in enumerate(lines, 1):
        if pending:
            if PRIVATE_KEY_END.search(line):
                # The block ends on this line, which is part of it. The same line may open
                # the next one (an END and then a BEGIN), and then it is the first line of that.
                pending.append(line)
                settle()
                if _opens_a_block(line):
                    pending.append(line)
                    first = number
                continue
            if not PRIVATE_KEY_BEGIN.search(line):
                pending.append(line)
                continue
            # A BEGIN with no END on its line: the open block cannot go on past it, and this
            # line is where the next one starts.
            settle()
            if number > upto:
                break
        elif number > upto:
            break
        if PRIVATE_KEY_BEGIN.search(line):
            pending.append(line)
            first = number
            if not _opens_a_block(line):
                settle()
    if pending:
        settle()
    return masked
