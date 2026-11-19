"""Fingerprints of tool definitions, and the file they are kept in.

A pin is what a tool looked like when the person first saw it. The store is a file under
ncc's home, so it is private from the moment it exists and is never half written.
"""

import fcntl
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from nanoclaude.mcp import pins as pins_module
from nanoclaude.mcp.pins import (
    MAX_PINS_PER_SERVER,
    PINS_FILENAME,
    PinStore,
    PinStoreError,
    fingerprint,
    pins_path,
)
from nanoclaude.private import write_private
from tests.conftest import mode_of

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}


# ---- the fingerprint


def test_a_fingerprint_is_a_sha256_in_hex():
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint("does a thing", SCHEMA))


def test_the_same_definition_has_the_same_fingerprint_whatever_order_its_keys_came_in():
    reordered = {"properties": {"text": {"type": "string"}}, "type": "object"}
    assert fingerprint("d", SCHEMA) == fingerprint("d", reordered)


def test_a_change_to_the_description_changes_the_fingerprint():
    assert fingerprint("does a thing", SCHEMA) != fingerprint("does a thing.", SCHEMA)


def test_a_change_to_the_schema_changes_the_fingerprint_however_small():
    wider = {**SCHEMA, "properties": {"text": {"type": "string"}, "x": {"type": "string"}}}
    retyped = {"type": "object", "properties": {"text": {"type": "integer"}}}
    described = {"type": "object", "properties": {"text": {"type": "string", "description": "t"}}}
    prints = {fingerprint("d", s) for s in (SCHEMA, wider, retyped, described)}
    assert len(prints) == 4


def test_a_description_and_a_schema_cannot_trade_text_to_keep_one_fingerprint():
    # Joined naively, a description ending where a schema begins would be the same bytes.
    assert fingerprint('x{"type":"object"}', {}) != fingerprint("x", {"type": "object"})


def test_a_schema_with_a_lone_surrogate_or_non_ascii_text_still_has_a_fingerprint():
    assert fingerprint("d", {"type": "object", "title": "café \ud800"})


# ---- the store


def test_a_file_that_is_not_there_is_an_empty_store(tmp_path):
    store = PinStore.load(tmp_path / PINS_FILENAME)
    assert store.get("github", "create_issue") is None


def test_the_path_is_under_ncc_home_in_a_file_of_its_own(tmp_path):
    assert pins_path(tmp_path) == tmp_path / ".nanoclaude" / "mcp-pins.json"
    assert PINS_FILENAME == "mcp-pins.json"


def test_a_pin_that_was_recorded_and_saved_is_there_in_the_next_process(tmp_path):
    path = tmp_path / ".nanoclaude" / PINS_FILENAME
    store = PinStore.load(path)
    store.record("github", "create_issue", "a" * 64)
    store.save()
    again = PinStore.load(path)
    assert again.get("github", "create_issue") == "a" * 64
    assert again.get("github", "other") is None
    assert again.get("other", "create_issue") is None


def test_the_file_holds_server_then_tool_then_fingerprint(tmp_path):
    path = tmp_path / PINS_FILENAME
    store = PinStore.load(path)
    store.record("github", "create_issue", "a" * 64)
    store.record("github", "list", "b" * 64)
    store.record("fs", "read", "c" * 64)
    store.save()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "github": {"create_issue": "a" * 64, "list": "b" * 64},
        "fs": {"read": "c" * 64},
    }


def test_the_file_is_private_from_the_call_that_makes_it_and_so_is_its_directory(
    tmp_path, private_from_the_start
):
    path = tmp_path / ".nanoclaude" / PINS_FILENAME
    store = PinStore.load(path)
    store.record("s", "t", "a" * 64)
    store.save()
    assert mode_of(path) == 0o600
    assert mode_of(path.parent) == 0o700


def test_the_file_is_usable_under_a_umask_that_would_lock_its_owner_out(tmp_path):
    path = tmp_path / ".nanoclaude" / PINS_FILENAME
    store = PinStore.load(path)
    store.record("s", "t", "a" * 64)
    previous = os.umask(0o277)
    try:
        store.save()
    finally:
        os.umask(previous)
    assert mode_of(path) == 0o600
    assert PinStore.load(path).get("s", "t") == "a" * 64


def test_a_file_that_was_open_to_others_is_private_once_it_is_written_again(tmp_path):
    path = tmp_path / PINS_FILENAME
    path.write_text("{}")
    path.chmod(0o644)
    store = PinStore.load(path)
    store.record("s", "t", "a" * 64)
    store.save()
    assert mode_of(path) == 0o600


def test_a_write_that_fails_leaves_the_old_file_whole_and_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / PINS_FILENAME
    first = PinStore.load(path)
    first.record("s", "t", "a" * 64)
    first.save()
    before = path.read_bytes()

    def refuse(self: Path, target: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", refuse)
    second = PinStore.load(path)
    second.record("s", "u", "b" * 64)
    with pytest.raises(OSError, match="disk full"):
        second.save()
    assert path.read_bytes() == before
    # The pins and the lock beside them, and no temporary file.
    assert sorted(p.name for p in tmp_path.iterdir()) == [PINS_FILENAME, PINS_FILENAME + ".lock"]


def test_saving_with_nothing_new_writes_nothing(tmp_path):
    path = tmp_path / PINS_FILENAME
    PinStore.load(path).save()
    assert not path.exists()


def test_a_pin_another_process_saved_in_the_meantime_is_not_lost(tmp_path):
    # Two sessions open at once, each learning a different server's tools.
    path = tmp_path / PINS_FILENAME
    mine = PinStore.load(path)
    theirs = PinStore.load(path)
    theirs.record("theirs", "t", "b" * 64)
    theirs.save()
    mine.record("mine", "t", "a" * 64)
    mine.save()
    after = PinStore.load(path)
    assert after.get("mine", "t") == "a" * 64
    assert after.get("theirs", "t") == "b" * 64


def test_a_pin_that_is_renewed_replaces_the_old_one(tmp_path):
    path = tmp_path / PINS_FILENAME
    store = PinStore.load(path)
    store.record("s", "t", "a" * 64)
    store.save()
    store.record("s", "t", "b" * 64)
    store.save()
    assert PinStore.load(path).get("s", "t") == "b" * 64


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{not json",
        "[]",
        '"x"',
        '{"s": []}',
        '{"s": {"t": 1}}',
        '{"s": {"t": null}}',
        '{"s": "t"}',
    ],
    ids=["empty", "syntax", "list", "string", "server list", "number", "null", "server string"],
)
def test_a_file_that_is_not_the_shape_of_a_store_is_an_error_that_says_which_file(tmp_path, text):
    path = tmp_path / PINS_FILENAME
    path.write_text(text)
    with pytest.raises(PinStoreError) as raised:
        PinStore.load(path)
    assert str(path) in str(raised.value)
    assert " — " in str(raised.value)


def test_a_file_that_is_not_text_is_an_error_too(tmp_path):
    path = tmp_path / PINS_FILENAME
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(PinStoreError, match="UTF-8"):
        PinStore.load(path)


def test_a_file_that_became_unreadable_between_loading_and_saving_is_not_overwritten(tmp_path):
    path = tmp_path / PINS_FILENAME
    store = PinStore.load(path)
    path.write_text("{broken")
    store.record("s", "t", "a" * 64)
    with pytest.raises(PinStoreError):
        store.save()
    assert path.read_text() == "{broken"


def test_a_path_that_cannot_be_read_as_a_file_is_an_error_that_says_so(tmp_path):
    (tmp_path / PINS_FILENAME).mkdir()
    with pytest.raises(PinStoreError, match="could not read"):
        PinStore.load(tmp_path / PINS_FILENAME)


def test_the_store_says_whether_it_holds_pins_that_were_not_saved(tmp_path):
    store = PinStore.load(tmp_path / PINS_FILENAME)
    fresh = store.unsaved
    store.record("s", "t", "a" * 64)
    recorded = store.unsaved
    store.save()
    saved = store.unsaved
    assert (fresh, recorded, saved) == (False, True, False)
    assert store.path == tmp_path / PINS_FILENAME


# ---- two writers at once


def test_two_saves_that_overlap_cannot_lose_each_others_pins(tmp_path, monkeypatch):
    """Both writers have read the file before either writes: without a lock, whichever writes
    last puts back what it read and the other's pin is gone."""
    path = tmp_path / PINS_FILENAME
    first, second = PinStore.load(path), PinStore.load(path)
    first.record("one", "t", "1" * 64)
    second.record("two", "t", "2" * 64)
    first_is_writing, second_has_begun = threading.Event(), threading.Event()
    real_read, real_flock = pins_module._read, fcntl.flock

    def named(who: str) -> bool:
        return threading.current_thread().name == who

    def read(target: Path) -> dict[str, dict[str, str]]:
        found = real_read(target)
        if named("second"):
            second_has_begun.set()  # it has read what is there, and is about to write on it
        return found

    def flock(descriptor: int, operation: int) -> None:
        if named("second"):
            second_has_begun.set()  # it is asking for the lock, which the first holds
        return real_flock(descriptor, operation)

    def write(target: Path, text: str, *, replace: bool) -> None:
        if named("first"):
            first_is_writing.set()
            assert second_has_begun.wait(10), "the second writer never began"
        return write_private(target, text, replace=replace)

    monkeypatch.setattr(pins_module, "_read", read)
    monkeypatch.setattr(fcntl, "flock", flock)
    monkeypatch.setattr(pins_module, "write_private", write)
    one = threading.Thread(target=first.save, name="first")
    two = threading.Thread(target=second.save, name="second")
    one.start()
    assert first_is_writing.wait(10), "the first writer never got to its write"
    two.start()
    one.join(30)
    two.join(30)
    assert not one.is_alive() and not two.is_alive()
    after = PinStore.load(path)
    assert (after.get("one", "t"), after.get("two", "t")) == ("1" * 64, "2" * 64)


WRITER = """
import sys
from pathlib import Path
from nanoclaude.mcp.pins import PinStore

path, who = Path(sys.argv[1]), sys.argv[2]
for number in range(25):
    store = PinStore.load(path)
    store.record(who, f"tool_{number}", "a" * 64)
    store.save()
"""


def test_processes_that_save_at_once_keep_every_pin(tmp_path):
    path = tmp_path / ".nanoclaude" / PINS_FILENAME
    path.parent.mkdir()
    writers = [
        subprocess.Popen(  # noqa: S603 - this interpreter, a script that is in this file
            [sys.executable, "-c", WRITER, str(path), f"w{n}"], stderr=subprocess.PIPE
        )
        for n in range(6)
    ]
    failures = [(w.communicate(timeout=120)[1], w.returncode) for w in writers]
    assert all(code == 0 for _, code in failures), failures
    after = PinStore.load(path)
    missing = [
        (f"w{n}", f"tool_{i}")
        for n in range(6)
        for i in range(25)
        if after.get(f"w{n}", f"tool_{i}") is None
    ]
    assert missing == []


def test_the_lock_is_a_private_file_beside_the_pins_that_stays(tmp_path, private_from_the_start):
    path = tmp_path / ".nanoclaude" / PINS_FILENAME
    store = PinStore.load(path)
    store.record("s", "t", "a" * 64)
    store.save()
    lock = path.with_name(PINS_FILENAME + ".lock")
    assert lock.exists() and mode_of(lock) == 0o600
    store.record("s", "u", "b" * 64)
    store.save()
    assert lock.exists()


def test_a_lock_that_stays_held_ends_the_save_with_an_error_and_writes_nothing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pins_module, "LOCK_WAIT_SECONDS", 0.2)
    path = tmp_path / PINS_FILENAME
    first = PinStore.load(path)
    first.record("s", "t", "a" * 64)
    first.save()
    lock = path.with_name(PINS_FILENAME + ".lock")
    held = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        stuck = PinStore.load(path)
        stuck.record("s", "u", "b" * 64)
        with pytest.raises(PinStoreError, match="lock"):
            stuck.save()
    finally:
        os.close(held)
    assert PinStore.load(path).get("s", "u") is None
    stuck.save()  # once the lock is let go the same store can be saved
    assert PinStore.load(path).get("s", "u") == "b" * 64


# ---- the most pins a server may have


def full(path: Path, server: str = "s") -> PinStore:
    return PinStore(path, {server: {f"t{n}": "a" * 64 for n in range(MAX_PINS_PER_SERVER)}})


def test_a_server_may_have_4096_pins_and_no_more():
    assert MAX_PINS_PER_SERVER == 4096


def test_a_new_pin_past_the_limit_is_not_recorded_and_says_so(tmp_path):
    store = full(tmp_path / PINS_FILENAME)
    assert store.record("s", "one-more", "b" * 64) is False
    assert store.get("s", "one-more") is None
    assert store.unsaved is False


def test_a_pin_that_is_already_there_can_be_renewed_when_the_server_is_at_the_limit(tmp_path):
    store = full(tmp_path / PINS_FILENAME)
    assert store.record("s", "t7", "b" * 64) is True
    assert store.get("s", "t7") == "b" * 64


def test_the_limit_is_for_each_server_and_not_for_the_file(tmp_path):
    store = full(tmp_path / PINS_FILENAME)
    assert store.record("other", "t0", "b" * 64) is True


def test_a_save_does_not_take_the_file_past_the_limit_when_another_process_filled_it(tmp_path):
    path = tmp_path / PINS_FILENAME
    mine = PinStore.load(path)  # the file was not there, so there was room
    mine.record("s", "mine", "b" * 64)
    path.write_text(json.dumps({"s": {f"t{n}": "a" * 64 for n in range(MAX_PINS_PER_SERVER)}}))
    mine.save()
    kept = json.loads(path.read_text())["s"]
    assert len(kept) == MAX_PINS_PER_SERVER and "mine" not in kept
