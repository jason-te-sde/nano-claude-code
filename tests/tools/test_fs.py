import os
from collections.abc import Hashable

import pytest

from nanoclaude.tools.fs import (
    BinaryFileError,
    FileSnapshot,
    FileStamp,
    FileTooLargeError,
    read_text,
    stamp_of,
    write_atomic,
)


def test_reading_returns_content_and_a_stamp(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello\n")
    snapshot = read_text(str(target))
    assert snapshot.content == "hello\n"
    assert snapshot.stamp.size == 6
    assert snapshot.stamp == stamp_of(str(target))


def test_a_file_over_the_limit_is_refused_with_a_usable_message(tmp_path):
    target = tmp_path / "big.txt"
    target.write_bytes(b"x" * 1_000_001)
    with pytest.raises(FileTooLargeError, match="Grep"):
        read_text(str(target))


def test_a_binary_file_is_refused_rather_than_mangled(tmp_path):
    target = tmp_path / "a.bin"
    target.write_bytes(b"\x89PNG\x00\x00binary")
    with pytest.raises(BinaryFileError):
        read_text(str(target))


def test_invalid_utf8_is_refused(tmp_path):
    target = tmp_path / "a.txt"
    target.write_bytes(b"caf\xe9 latte")
    with pytest.raises(BinaryFileError, match="UTF-8"):
        read_text(str(target))


def test_writing_is_atomic_and_leaves_no_temporary_files(tmp_path):
    target = tmp_path / "sub" / "a.txt"
    write_atomic(str(target), "one\n")
    write_atomic(str(target), "two\n")
    assert target.read_text() == "two\n"
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["a.txt"]


def test_writing_preserves_the_existing_file_mode(tmp_path):
    target = tmp_path / "run.sh"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    write_atomic(str(target), "#!/bin/sh\necho hi\n")
    assert oct(target.stat().st_mode & 0o777) == "0o755"


def test_a_failed_write_leaves_the_previous_content_intact(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("original\n")

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(str(target), "replacement\n")
    assert target.read_text() == "original\n"
    assert list(tmp_path.iterdir()) == [target]  # temp file cleaned up


def test_stamp_of_a_missing_file_is_none(tmp_path):
    assert stamp_of(str(tmp_path / "nope.txt")) is None


def test_a_nul_byte_is_refused_even_though_the_rest_decodes_as_utf8(tmp_path):
    """Distinct from test_a_binary_file_is_refused_rather_than_mangled above: that
    file's leading bytes (\\x89PNG) are *also* invalid UTF-8, so it would still
    raise BinaryFileError -- with a different message -- if the NUL sniff were
    deleted. Every byte here, including the NUL, is valid UTF-8 on its own, so
    deleting the sniff would make this read succeed instead of raising.
    """
    target = tmp_path / "a.txt"
    target.write_bytes(b"before\x00after")
    with pytest.raises(BinaryFileError, match="NUL byte"):
        read_text(str(target))


def test_file_stamp_is_hashable():
    """Every field is a primitive (fs.py), so this is safely hashable."""
    stamp = FileStamp("deadbeef", 4, 123)
    assert isinstance(stamp, Hashable)
    hash(stamp)  # must not raise


def test_file_snapshot_is_hashable():
    """content is a str and stamp is the (hashable) FileStamp above."""
    snapshot = FileSnapshot("hi\n", FileStamp("deadbeef", 4, 123))
    assert isinstance(snapshot, Hashable)
    hash(snapshot)  # must not raise


def test_stamp_of_a_directory_is_none(tmp_path):
    assert stamp_of(str(tmp_path)) is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file")
def test_stamp_of_an_unreadable_file_is_none(tmp_path):
    target = tmp_path / "locked.txt"
    target.write_text("x")
    target.chmod(0)
    try:
        assert stamp_of(str(target)) is None
    finally:
        target.chmod(0o600)
