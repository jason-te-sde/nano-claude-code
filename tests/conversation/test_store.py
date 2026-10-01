"""Tests for nanoclaude.conversation.store.

Every database lives under pytest's ``tmp_path``: this module must never write
into the repository itself.
"""

import json
from collections.abc import Hashable
from types import SimpleNamespace

import pytest

from nanoclaude.conversation.store import SessionRow, Store, decode_blocks, encode_blocks
from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    user_text,
    validate,
)
from nanoclaude.providers.base import Usage


# One case per block type, plus the two variations the type alone does not
# capture: a ThinkingBlock with no signature (the dataclass default) and a
# ToolUseBlock whose arguments nest a list of dicts, the shape Edit and Grep
# actually send and the one a shallow copy would get wrong.
@pytest.mark.parametrize(
    "block",
    [
        pytest.param(TextBlock("hello"), id="text"),
        pytest.param(ThinkingBlock("pondering", signature="sig"), id="thinking-with-signature"),
        pytest.param(ThinkingBlock("pondering"), id="thinking-without-signature"),
        pytest.param(ToolUseBlock("t1", "Read", {"path": "a.py", "limit": 10}), id="tool-use"),
        pytest.param(
            ToolUseBlock("t1", "Edit", {"edits": [{"old_string": "a", "new_string": "b"}]}),
            id="tool-use-nested-arguments",
        ),
        pytest.param(ToolResultBlock("t1", "1\tx = 1", is_error=False), id="tool-result"),
        pytest.param(ToolResultBlock("t1", "boom", is_error=True), id="tool-result-error"),
    ],
)
def test_every_block_type_survives_a_round_trip(block):
    assert decode_blocks(encode_blocks((block,))) == (block,)


def test_a_tuple_of_mixed_blocks_round_trips_together():
    # The per-type cases above each encode a single block; this pins that
    # encode_blocks/decode_blocks also agree on ordering and count when
    # several different block types travel through the same call, the shape
    # every real message actually uses.
    blocks = (
        TextBlock("hello"),
        ThinkingBlock("pondering", signature="sig"),
        ToolUseBlock("t1", "Read", {"path": "a.py", "limit": 10}),
        ToolResultBlock("t1", "1\tx = 1", is_error=False),
    )
    assert decode_blocks(encode_blocks(blocks)) == blocks


def test_decoding_an_unknown_block_type_is_rejected():
    # A session database outliving the code that wrote it is the one place
    # this can happen: a future block type rolled back by an older build.
    with pytest.raises(ValueError, match="unknown block type 'bogus'"):
        decode_blocks(json.dumps([{"type": "bogus"}]))


def test_a_transcript_reloads_identically(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={"main": "sonnet"})
    messages = [
        user_text("hi"),
        Message("assistant", (ToolUseBlock("t1", "Read", {"path": "a.py"}),)),
        Message("user", (ToolResultBlock("t1", "1\tx = 1"),)),
    ]
    for seq, message in enumerate(messages):
        store.append_message("s1", seq, message)

    reloaded = store.load_transcript("s1")
    assert reloaded.messages == tuple(messages)
    validate(reloaded)


def _fixed_clock(monkeypatch: pytest.MonkeyPatch, *instants: float) -> None:
    # store.py reads time.time(); two back-to-back calls can return the same
    # value, so tests that depend on order pin the clock instead of racing it.
    # Only the store module's own reference is replaced, not the global clock.
    ticks = iter(instants)
    monkeypatch.setattr(
        "nanoclaude.conversation.store.time", SimpleNamespace(time=lambda: next(ticks))
    )


def test_latest_session_is_the_most_recently_started(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch, 1.0, 2.0)
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("old", cwd="/p", roles={})
    store.create_session("new", cwd="/p", roles={})
    assert store.latest_session_id() == "new"


def test_sessions_started_in_the_same_instant_resolve_to_the_later_insert(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch, 5.0, 5.0)
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("first", cwd="/p", roles={})
    store.create_session("second", cwd="/p", roles={})
    assert store.latest_session_id() == "second"
    assert [row.id for row in store.recent_sessions(10)] == ["second", "first"]


def test_recent_sessions_are_newest_first_and_respect_the_limit(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch, 1.0, 3.0, 2.0)
    store = Store(tmp_path / "s.db")
    store.open()
    for session_id in ("a", "b", "c"):
        store.create_session(session_id, cwd="/p", roles={})
    assert [row.id for row in store.recent_sessions(10)] == ["b", "c", "a"]
    assert [row.id for row in store.recent_sessions(2)] == ["b", "c"]


def test_latest_session_id_is_none_for_an_empty_database(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    assert store.latest_session_id() is None


def test_finishing_a_session_records_usage_and_cost(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.finish_session("s1", Usage(input_tokens=100, output_tokens=20), cost_usd=0.0031)
    row = store.recent_sessions(1)[0]
    assert (row.total_input_tokens, row.total_cost_usd) == (100, 0.0031)


def test_opening_an_existing_database_does_not_lose_data(tmp_path):
    path = tmp_path / "s.db"
    first = Store(path)
    first.open()
    first.create_session("s1", cwd="/p", roles={})
    first.close()

    second = Store(path)
    second.open()
    assert second.latest_session_id() == "s1"


def test_using_the_store_before_open_is_rejected(tmp_path):
    store = Store(tmp_path / "s.db")
    with pytest.raises(RuntimeError, match="store is not open"):
        _ = store.db


def test_session_row_is_hashable():
    # Every field is a primitive (str, float, int, or None), so frozen=True's
    # generated __hash__ is sound here -- unlike ToolUseBlock
    # (conversation/transcript.py), which holds a Mapping and must declare
    # __hash__ = None instead. Pinned so a later field (a dict, a list) does
    # not silently make recent_sessions() return unhashable rows.
    row = SessionRow(
        id="s1",
        started_at=1.0,
        ended_at=None,
        cwd="/p",
        total_cost_usd=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
    )
    assert isinstance(row, Hashable)
    assert hash(row) == hash(row)
