# src/nanoclaude/tools/fs.py
"""Filesystem access with three guarantees the agent depends on.

**Atomic writes.** Content goes to a temporary file in the same directory, is
flushed, and is moved into place with ``os.replace``. A crash can lose the new
content, which is recoverable; it cannot leave a half-written source file,
which is not.

**Refusal to read what it cannot handle.** Files above the size limit, and
files that are not valid UTF-8, are rejected with an error the model can act on
rather than decoded into replacement characters that burn context and say
nothing. So is anything that is not a regular file, before it is opened: opening
a FIFO waits for someone to write to it, and a call that waits for that holds up
the whole session.

**A stamp on everything read.** Read-before-edit and stale-file detection both
key off the hash.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: Larger than almost any hand-written source file; small enough that reading
#: one by accident does not consume the context window.
MAX_READ_BYTES = 1_000_000
BINARY_SNIFF_BYTES = 8192


class FileSystemError(OSError):
    """A refusal this module makes on purpose."""


class FileTooLargeError(FileSystemError):
    pass


class BinaryFileError(FileSystemError):
    pass


@dataclass(frozen=True, slots=True)
class FileStamp:
    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    content: str
    stamp: FileStamp


def _stamp(path: Path, data: bytes) -> FileStamp:
    return FileStamp(hashlib.sha256(data).hexdigest(), len(data), path.stat().st_mtime_ns)


def _size_of_what_can_be_read(path: str) -> int:
    """How large ``path`` is, once it is known to be a file that can be opened and read to its end.

    Anything but a regular file is refused here, before it is opened. A FIFO is not read
    until somebody writes to it, and a device or a socket has no end: neither is a file the
    agent could mean to read or replace. A directory is left to the read that follows, which
    refuses it in its own way. What the path names is looked at through links, as it is
    opened through them.
    """
    status = Path(path).stat()
    if not (stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)):
        raise FileSystemError(f"{path} is not a regular file")
    return status.st_size


def stamp_of(path: str) -> FileStamp | None:
    target = Path(path)
    try:
        _size_of_what_can_be_read(path)
        return _stamp(target, target.read_bytes())
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return None


def read_text(path: str) -> FileSnapshot:
    target = Path(path)
    size = _size_of_what_can_be_read(path)
    if size > MAX_READ_BYTES:
        raise FileTooLargeError(
            f"{path} is {size} bytes, over the {MAX_READ_BYTES}-byte read limit. "
            "Use Grep to search it, or Bash to slice out the part you need."
        )
    data = target.read_bytes()
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        raise BinaryFileError(f"{path} looks binary (NUL byte near the start)")
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BinaryFileError(f"{path} is not valid UTF-8: {exc}") from exc
    return FileSnapshot(content, _stamp(target, data))


def write_atomic(path: str, content: str) -> FileStamp:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    mode = target.stat().st_mode & 0o777 if target.exists() else None

    handle, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            tmp_path.chmod(mode)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    # The directory entry is metadata too: without this a crash can lose the rename.
    dir_fd = os.open(str(target.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return _stamp(target, data)
