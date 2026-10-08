"""What ncc keeps under its home is private, and private from the moment it exists.

The files there hold your code (the session store), what you typed (the history) and, for
``ncc init``, a key variable's name and the config around it. On a machine with other users
a file made with the default mode is readable by them until something narrows it, and a
directory made with the default mode lets them in to look. So nothing here is made and then
chmod-ed: every file is created by ``os.open`` with mode 0600 in its arguments and every
directory by ``mkdir`` with mode 0700, and the chmod that follows each is only for a umask
that took bits away from its owner (0277 would make a file read-only to the person it
belongs to), which it can only put back, never widen past.

What was already there is left as it is: a directory or a file that exists keeps its mode,
because it is the person's, and a person who shared it on purpose has not asked for it to
be taken back. A file that is *replaced* is a new file, and is private.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path


class TargetExistsError(Exception):
    """The file is there, and the write was not told that replacing it was agreed to."""


def make_private_directories(directory: Path) -> None:
    """Make ``directory`` and every part of it that is missing, each with mode 0700.

    Only what is made here is touched: a directory that was already there keeps its mode.
    ``mkdir`` filters its mode by the umask, so that with a umask of 0177 or tighter (any bit
    of 0300) a new directory would lose its owner's write or search bit and nothing could be
    made in it; the mode is set again after, and the directory that holds a private file
    is private too.
    """
    missing = []
    for candidate in (directory, *directory.parents):
        if candidate.exists():
            break
        missing.append(candidate)
    for candidate in reversed(missing):
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        candidate.chmod(0o700)


def write_private(path: Path, text: str, *, replace: bool) -> None:
    """Write ``text`` to ``path``, which exists with mode 0600 from the moment it does.

    The file is made by ``os.open`` with the mode in its arguments, so that no other user
    can read it between its being made and its being made private; that mode is filtered by
    the umask (0277 makes it 0400, 0777 makes it 0000), so it is set on the descriptor too.
    It is then moved into place: a file that was already there is replaced by one that is
    private, whatever its own mode was, and one that was interrupted leaves no half of a
    file. ``O_EXCL`` refuses a temporary name that is already there, a link included, so
    that nothing is written through one.

    ``replace`` says whether replacing a file is agreed to. When it is not, the file is
    put in place by a hard link, which fails if anything is there, a dangling link included,
    and :class:`TargetExistsError` is raised and nothing is replaced: a file that appeared
    after the last look is somebody's work.
    """
    make_private_directories(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(text)
        if replace:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise TargetExistsError from exc
            with contextlib.suppress(OSError):
                temporary.unlink()
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def create_private_file(path: Path) -> None:
    """Make an empty file at ``path``, mode 0600 from the moment it exists; leave one that is there.

    For what something else is going to open and write to, and so cannot be given a mode
    by the one writing: SQLite makes its write-ahead log and index with the mode of the
    database, so the database has to be private before it opens it, and a prompt history
    is appended to by a library that opens it with the default mode. ``O_EXCL`` says "not
    if it is there", and a link at the name counts as there, so nothing is made through one.
    """
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
