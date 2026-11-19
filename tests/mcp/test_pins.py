"""Fingerprints of tool definitions, and the file they are kept in.

A pin is what a tool looked like when the person first saw it. The store is a file under
ncc's home, so it is private from the moment it exists and is never half written.
"""

import json
import os
import re
from pathlib import Path

import pytest

from nanoclaude.mcp.pins import PINS_FILENAME, PinStore, PinStoreError, fingerprint, pins_path
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
    assert [p.name for p in tmp_path.iterdir()] == [PINS_FILENAME]


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
