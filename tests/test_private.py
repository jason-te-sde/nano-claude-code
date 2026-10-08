"""Tests for nanoclaude.private: what ncc keeps under its home is private from the start."""

from __future__ import annotations

import os

import pytest

from nanoclaude.private import (
    TargetExistsError,
    create_private_file,
    make_private_directories,
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
