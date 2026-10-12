"""Tests for nanoclaude.private: what ncc keeps under its home is private from the start."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.private import (
    TargetExistsError,
    create_private_file,
    make_private_directories,
    narrow_directory,
    narrow_file,
    write_private,
)
from tests.conftest import mode_of


def test_every_directory_that_is_made_is_private_and_one_that_was_there_keeps_its_mode(
    tmp_path, private_from_the_start
):
    (tmp_path / "there").mkdir(mode=0o755)
    make_private_directories(tmp_path / "there" / "a" / "b")
    assert mode_of(tmp_path / "there" / "a") == 0o700
    assert mode_of(tmp_path / "there" / "a" / "b") == 0o700
    assert mode_of(tmp_path / "there") == 0o755


def test_a_directory_is_private_even_under_a_umask_that_would_lock_its_owner_out(tmp_path):
    previous = os.umask(0o277)
    try:
        make_private_directories(tmp_path / "state" / "inner")
    finally:
        os.umask(previous)
    assert mode_of(tmp_path / "state") == 0o700
    assert mode_of(tmp_path / "state" / "inner") == 0o700
    (tmp_path / "state" / "inner" / "x").write_text("can be written")


def test_a_file_is_private_from_the_call_that_creates_it(tmp_path, private_from_the_start):
    create_private_file(tmp_path / "f")
    assert mode_of(tmp_path / "f") == 0o600
    assert (tmp_path / "f").read_bytes() == b""


def test_a_file_is_usable_under_a_umask_that_would_lock_its_owner_out(tmp_path):
    previous = os.umask(0o277)
    try:
        create_private_file(tmp_path / "f")
    finally:
        os.umask(previous)
    assert mode_of(tmp_path / "f") == 0o600
    (tmp_path / "f").write_text("can be written")


def test_a_file_that_is_already_there_is_left_exactly_as_it_is(tmp_path):
    existing = tmp_path / "f"
    existing.write_text("mine")
    existing.chmod(0o644)
    create_private_file(existing)
    assert (existing.read_text(), mode_of(existing)) == ("mine", 0o644)


def test_a_link_at_the_name_is_not_written_through(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("precious")
    (tmp_path / "f").symlink_to(victim)
    create_private_file(tmp_path / "f")
    assert victim.read_text() == "precious"
    assert (tmp_path / "f").is_symlink()


def test_a_file_is_written_private_and_replaces_what_was_there(tmp_path, private_from_the_start):
    target = tmp_path / "state" / "f.json"
    write_private(target, "one", replace=True)
    assert (target.read_text(), mode_of(target), mode_of(target.parent)) == ("one", 0o600, 0o700)
    write_private(target, "two", replace=True)
    assert (target.read_text(), mode_of(target)) == ("two", 0o600)


def test_a_written_file_is_private_and_not_locked_to_its_owner_under_a_restrictive_umask(tmp_path):
    previous = os.umask(0o277)
    try:
        write_private(tmp_path / "state" / "f.json", "one", replace=True)
    finally:
        os.umask(previous)
    target = tmp_path / "state" / "f.json"
    assert (mode_of(target), mode_of(target.parent)) == (0o600, 0o700)
    target.write_text("can be written again")


def test_a_file_that_was_open_to_others_is_private_once_it_is_replaced(tmp_path):
    target = tmp_path / "f.json"
    target.write_text("old")
    target.chmod(0o644)
    write_private(target, "new", replace=True)
    assert (target.read_text(), mode_of(target)) == ("new", 0o600)


def test_a_file_that_appeared_is_not_replaced_unless_that_was_agreed(tmp_path):
    target = tmp_path / "f.json"
    target.write_text("somebody's")
    with pytest.raises(TargetExistsError):
        write_private(target, "new", replace=False)
    assert target.read_text() == "somebody's"
    assert list(tmp_path.iterdir()) == [target]


def test_nothing_is_left_behind_when_the_write_is_interrupted(tmp_path, monkeypatch):
    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_private(tmp_path / "f.json", "new", replace=True)
    assert list(tmp_path.iterdir()) == []


# ---- what an earlier version made open to others is narrowed when ncc opens it


def test_a_file_an_earlier_version_made_open_to_others_is_made_private(tmp_path):
    target = tmp_path / "f"
    target.write_text("mine")
    target.chmod(0o644)
    narrow_file(target)
    assert (target.read_text(), mode_of(target)) == ("mine", 0o600)


def test_a_directory_an_earlier_version_made_open_to_others_is_made_private(tmp_path):
    target = tmp_path / "d"
    target.mkdir()
    target.chmod(0o755)
    (target / "inside").write_text("kept")
    narrow_directory(target)
    assert (mode_of(target), (target / "inside").read_text()) == (0o700, "kept")


@pytest.mark.parametrize("before", [0o400, 0o640, 0o660, 0o666, 0o604, 0o000])
def test_whatever_a_files_mode_was_it_ends_at_0600(tmp_path, before):
    target = tmp_path / "f"
    target.write_text("x")
    target.chmod(before)
    narrow_file(target)
    assert mode_of(target) == 0o600


@pytest.mark.parametrize("before", [0o500, 0o750, 0o770, 0o777, 0o705])
def test_whatever_a_directorys_mode_was_it_ends_at_0700(tmp_path, before):
    target = tmp_path / "d"
    target.mkdir()
    target.chmod(before)
    narrow_directory(target)
    assert mode_of(target) == 0o700
    target.chmod(0o700)


def test_what_is_not_there_is_not_made(tmp_path):
    narrow_file(tmp_path / "missing")
    narrow_directory(tmp_path / "missing-dir")
    assert list(tmp_path.iterdir()) == []


def test_a_file_where_a_directory_was_expected_and_the_other_way_round_is_left_alone(tmp_path):
    (tmp_path / "f").write_text("x")
    (tmp_path / "f").chmod(0o644)
    (tmp_path / "d").mkdir()
    (tmp_path / "d").chmod(0o755)
    narrow_directory(tmp_path / "f")
    narrow_file(tmp_path / "d")
    assert (mode_of(tmp_path / "f"), mode_of(tmp_path / "d")) == (0o644, 0o755)


def test_a_link_is_not_followed_to_change_what_it_leads_to(tmp_path):
    """A link at the name is somebody's arrangement, and its target is not ncc's."""
    shared = tmp_path / "shared"
    shared.write_text("theirs")
    shared.chmod(0o644)
    (tmp_path / "sessions.db").symlink_to(shared)
    narrow_file(tmp_path / "sessions.db")
    assert mode_of(shared) == 0o644
    directory = tmp_path / "dir"
    directory.mkdir()
    directory.chmod(0o755)
    (tmp_path / "link").symlink_to(directory, target_is_directory=True)
    narrow_directory(tmp_path / "link")
    assert mode_of(directory) == 0o755


def test_a_chmod_that_is_refused_does_not_stop_anything(tmp_path, monkeypatch):
    """The file belongs to another user and chmod says EPERM: ncc goes on with what it has."""
    target = tmp_path / "f"
    target.write_text("x")
    target.chmod(0o644)

    def refused(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "chmod", refused)
    narrow_file(target)
    narrow_directory(tmp_path)
    assert mode_of(target) == 0o644


def test_what_another_user_owns_is_not_touched_even_when_one_could(tmp_path, monkeypatch):
    """As root a chmod of somebody else's file would succeed: it is still not ncc's."""
    target = tmp_path / "f"
    target.write_text("x")
    target.chmod(0o644)
    owner = target.stat().st_uid
    monkeypatch.setattr(os, "geteuid", lambda: owner + 1)
    narrow_file(target)
    assert mode_of(target) == 0o644


def test_a_file_that_is_already_private_is_not_chmoded_again(tmp_path, monkeypatch):
    target = tmp_path / "f"
    target.write_text("x")
    target.chmod(0o600)
    calls: list[object] = []
    monkeypatch.setattr(os, "chmod", lambda *args, **_kw: calls.append(args))
    narrow_file(target)
    assert calls == []


# ---- the temporary file of a write has a name nothing can already be using


def test_a_leftover_temporary_file_with_the_old_name_does_not_make_a_write_fail(tmp_path):
    """The name used to be the file's and the process id: a recycled id found its own old
    leftover in the way."""
    target = tmp_path / "capabilities.json"
    stale = tmp_path / f".capabilities.json.{os.getpid()}.tmp"
    stale.write_text("left by a process that died")
    write_private(target, "new", replace=True)
    assert target.read_text() == "new"
    assert stale.read_text() == "left by a process that died"


def test_the_temporary_name_is_random_and_made_with_o_excl(tmp_path, monkeypatch):
    seen: list[tuple[str, int]] = []
    real = os.open

    def recording(path: str | Path, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
        seen.append((str(path), flags))
        return real(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", recording)
    for _ in range(3):
        write_private(tmp_path / "f.json", "x", replace=True)
    names = [Path(path).name for path, _ in seen if ".tmp" in path]
    assert len(names) == len(set(names)) == 3, names
    for name in names:
        stem = name.removeprefix(".f.json.").removesuffix(".tmp")
        assert len(stem) == 16 and set(stem) <= set("0123456789abcdef"), name
    assert all(flags & os.O_EXCL for path, flags in seen if ".tmp" in path)


def test_a_name_that_is_taken_is_not_written_through_and_another_is_tried(tmp_path, monkeypatch):
    names = iter(["a" * 16, "b" * 16])
    monkeypatch.setattr("nanoclaude.private.secrets.token_hex", lambda _n: next(names))
    victim = tmp_path / "victim"
    victim.write_text("precious")
    (tmp_path / f".f.json.{'a' * 16}.tmp").symlink_to(victim)  # even a link
    write_private(tmp_path / "f.json", "new", replace=True)
    assert (tmp_path / "f.json").read_text() == "new"
    assert victim.read_text() == "precious"


def test_a_write_gives_up_when_every_name_it_tries_is_taken(tmp_path, monkeypatch):
    monkeypatch.setattr("nanoclaude.private.secrets.token_hex", lambda _n: "c" * 16)
    (tmp_path / f".f.json.{'c' * 16}.tmp").write_text("in the way")
    with pytest.raises(FileExistsError):
        write_private(tmp_path / "f.json", "new", replace=True)
    assert not (tmp_path / "f.json").exists()
