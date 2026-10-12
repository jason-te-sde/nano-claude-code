"""What ncc keeps under its home is private, and private from the moment it exists.

The files there hold your code (the session store), what you typed (the history) and, for
``ncc init``, a key variable's name and the config around it. On a machine with other users
a file made with the default mode is readable by them until something narrows it, and a
directory made with the default mode lets them in to look. So nothing here is made and then
chmod-ed: every file is created by ``os.open`` with mode 0600 in its arguments and every
directory by ``mkdir`` with mode 0700, and the chmod that follows each is only for a umask
that took bits away from its owner (0277 would make a file read-only to the person it
belongs to), which it can only put back, never widen past.

What an earlier version left open is narrowed when ncc opens it: the home directory, the
session store and the files SQLite keeps beside it, the prompt history and the capability
cache are ncc's own, were made with the default mode by versions that did not know better,
and are made 0700 (directories) and 0600 (files) by :func:`narrow_directory` and
:func:`narrow_file` the next time ncc opens them. What ncc does not own is not narrowed: the
person's ``config.toml`` keeps the mode they gave it, a link is not followed to change what
it leads to, and a file that belongs to another user is not touched (a ``chmod`` that is
refused, which is what that is, does not stop ncc). A file that is *replaced* is a new file,
and is private.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from collections.abc import Callable
from pathlib import Path

#: How many names :func:`write_private` tries for its temporary file before it gives up.
_TEMPORARY_ATTEMPTS = 8


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
    that nothing is written through one, and the name is random, so that one left behind by
    an earlier process is not in the way.

    ``replace`` says whether replacing a file is agreed to. When it is not, the file is
    put in place by a hard link, which fails if anything is there, a dangling link included,
    and :class:`TargetExistsError` is raised and nothing is replaced: a file that appeared
    after the last look is somebody's work.
    """
    make_private_directories(path.parent)
    # A name nothing can already be using: random, and made with O_EXCL, so that a leftover
    # from a process that died (or a link planted at the name) is never written through and
    # never makes this write fail. A process id was not enough: ids are recycled, and a
    # stale file with a recycled one was in the way for good.
    for _ in range(_TEMPORARY_ATTEMPTS):
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"no free name for the temporary file of {path} in {path.parent}")
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


def _narrow(path: Path, wanted: int, kind: Callable[[int], bool]) -> None:
    """Set ``path`` to ``wanted`` if it is ncc's to set: there, of the right kind (not a link),
    the running user's, and not already that mode. Best effort: a ``chmod`` that is refused
    is not an error."""
    try:
        status = os.lstat(path)
    except OSError:
        return
    if not kind(status.st_mode) or status.st_uid != os.geteuid():
        return
    if stat.S_IMODE(status.st_mode) == wanted:
        return
    with contextlib.suppress(OSError):
        path.chmod(wanted)


def narrow_file(path: Path) -> None:
    """Make a file ncc keeps private (0600) if an earlier version left it open to others.

    A path that is not there, is a link or is not a regular file is left alone, and so is a
    file that belongs to another user; a ``chmod`` that fails does not raise.
    """
    _narrow(path, 0o600, stat.S_ISREG)


def narrow_directory(path: Path) -> None:
    """Make a directory ncc keeps its files in private (0700) if an earlier version left it
    open to others. Otherwise as :func:`narrow_file`."""
    _narrow(path, 0o700, stat.S_ISDIR)
