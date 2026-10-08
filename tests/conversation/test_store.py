"""Tests for nanoclaude.conversation.store.

Every database lives under pytest's ``tmp_path``: this module must never write
into the repository itself.
"""

import json
import os
import sqlite3
from collections.abc import Hashable
from types import SimpleNamespace

import pytest

from nanoclaude.conversation.store import (
    MessageConflictError,
    SessionRow,
    Store,
    decode_blocks,
    encode_blocks,
    same_directory,
)
from nanoclaude.conversation.transcript import (
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
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


def test_one_stored_session_can_be_read_back_by_its_id(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch, 4.0, 9.0)
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/proj", roles={"main": "sonnet"})
    store.finish_session("s1", Usage(input_tokens=100, output_tokens=20), cost_usd=0.0031)
    assert store.session_row("s1") == SessionRow(
        id="s1",
        started_at=4.0,
        ended_at=9.0,
        cwd="/proj",
        total_cost_usd=0.0031,
        total_input_tokens=100,
        total_output_tokens=20,
    )


def test_a_session_that_never_finished_reads_back_with_an_unknown_cost(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    row = store.session_row("s1")
    assert row is not None
    assert (row.ended_at, row.total_cost_usd, row.total_input_tokens) == (None, None, 0)


def test_a_session_that_is_not_stored_reads_back_as_none(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    assert store.session_row("s2") is None


def test_the_row_read_by_id_is_the_one_the_listing_shows(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    assert store.recent_sessions(1) == [store.session_row("s1")]


def _audit(store: Store, session_id: str, tool_use_id: str, turn: int) -> None:
    store.db.execute(
        "INSERT INTO tool_calls "
        "(session_id, turn, tool_use_id, ts, tool, args_json, decision, rule) "
        "VALUES (?, ?, ?, 0, 'Read', '{}', 'allow', 'rule.allow')",
        (session_id, turn, tool_use_id),
    )
    store.db.commit()


def test_the_next_turn_to_audit_is_one_past_the_largest_already_audited(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    assert store.next_tool_turn("s1") == 0  # nothing audited yet
    for tool_use_id, turn in (("a", 0), ("b", 3), ("c", 1)):
        _audit(store, "s1", tool_use_id, turn)
    assert store.next_tool_turn("s1") == 4


def test_the_next_turn_to_audit_counts_only_the_session_asked_about(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.create_session("s2", cwd="/p", roles={})
    _audit(store, "s2", "a", 7)
    assert store.next_tool_turn("s1") == 0
    assert store.next_tool_turn("s2") == 8


def test_latest_session_id_is_none_for_an_empty_database(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    assert store.latest_session_id() is None


def _talked(store: Store, session_id: str, cwd: str) -> None:
    """A session started in ``cwd`` in which something was said."""
    store.create_session(session_id, cwd=cwd, roles={})
    store.append_message(session_id, 0, user_text("hello"))


def _ticking(monkeypatch: pytest.MonkeyPatch) -> None:
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 100)))


def test_latest_session_in_a_directory_ignores_sessions_started_elsewhere(tmp_path, monkeypatch):
    _ticking(monkeypatch)
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "here-old", "/p/here")
    _talked(store, "here-new", "/p/here")
    _talked(store, "elsewhere", "/p/elsewhere")  # the latest in the store
    assert store.latest_session_id() == "elsewhere"
    assert store.latest_session_id(cwd="/p/here") == "here-new"
    assert store.latest_session_id(cwd="/p/elsewhere") == "elsewhere"


def test_latest_session_in_a_directory_is_none_when_nothing_was_started_there(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "s1", "/p/here")
    assert store.latest_session_id(cwd="/p/nowhere") is None


def test_a_directory_is_matched_whole_and_not_as_a_prefix_or_a_pattern(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "deep", "/p/here/deeper")
    _talked(store, "sibling", "/p/here-too")
    _talked(store, "pattern", "/p/h_re")
    assert store.latest_session_id(cwd="/p/here") is None
    assert store.latest_session_id(cwd="/p/h%") is None
    assert store.latest_session_id(cwd="/p/h_re") == "pattern"


def test_a_session_nothing_was_said_in_is_not_the_one_to_continue(tmp_path, monkeypatch):
    # A REPL left at once leaves a row with no conversation, and --continue landing on it
    # would take the person back to nothing while their last real conversation sits behind it.
    _ticking(monkeypatch)
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "real", "/p/here")
    store.create_session("abandoned", cwd="/p/here", roles={})
    assert store.latest_session_id() == "abandoned"  # the whole store is as it was
    assert store.latest_session_id(cwd="/p/here") == "real"


def test_with_nothing_said_in_any_session_there_is_none_to_continue(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("abandoned", cwd="/p/here", roles={})
    assert store.latest_session_id(cwd="/p/here") is None


def test_a_conversation_that_was_cleared_has_nothing_left_to_continue(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "s1", "/p/here")
    store.replace_transcript("s1", Transcript())
    assert store.latest_session_id(cwd="/p/here") is None


def test_a_directory_is_the_same_place_however_it_was_spelled(tmp_path, monkeypatch):
    _ticking(monkeypatch)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "by-link", str(link))
    _talked(store, "by-real", str(real))
    _talked(store, "elsewhere", str(tmp_path))
    assert store.latest_session_id(cwd=str(real)) == "by-real"
    assert store.latest_session_id(cwd=str(link)) == "by-real"  # the same place, the newest
    # And the older spelling is found from the newer, as the only one there is.
    only = Store(tmp_path / "t.db")
    only.open()
    _talked(only, "by-link", str(link))
    assert only.latest_session_id(cwd=str(real)) == "by-link"


def test_a_directory_with_another_capitalisation_is_the_same_place_where_the_disk_says_so(
    tmp_path,
):
    (tmp_path / "Project").mkdir()
    if not (tmp_path / "project").exists():
        pytest.skip("this file system tells Project from project")
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "s1", str(tmp_path / "Project"))
    assert store.latest_session_id(cwd=str(tmp_path / "project")) == "s1"


def test_a_directory_that_has_gone_is_still_compared_by_where_it_was(tmp_path):
    gone = tmp_path / "gone"
    store = Store(tmp_path / "s.db")
    store.open()
    _talked(store, "s1", str(gone))
    assert store.latest_session_id(cwd=str(gone)) == "s1"
    assert store.latest_session_id(cwd=str(tmp_path / "other")) is None


def test_same_directory_is_the_comparison_resume_uses_too(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert same_directory(str(link), str(real))
    assert not same_directory(str(real), str(tmp_path))
    assert same_directory("/does/not/exist", "/does/not/exist")
    # Gone, and reached through a link that is not: the place the link leads to.
    assert same_directory(str(link / "gone"), str(real / "gone"))
    assert not same_directory(str(link / "gone"), str(tmp_path / "gone"))
    # Gone, and spelled differently: where each was, once the dots are resolved.
    assert same_directory("/does/not/../not/exist", "/does/not/exist")
    assert not same_directory("/does/not/exist", "/does/not/exist-too")


def test_finishing_a_session_records_usage_and_cost(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.finish_session("s1", Usage(input_tokens=100, output_tokens=20), cost_usd=0.0031)
    row = store.recent_sessions(1)[0]
    assert (row.total_input_tokens, row.total_cost_usd) == (100, 0.0031)


def _stored_cost(store: Store, session_id: str) -> object:
    """The raw column, so a NULL is seen as NULL and not as whatever the row type makes of it."""
    return store.db.execute(
        "SELECT total_cost_usd FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()[0]


def test_a_session_whose_cost_is_unknown_is_stored_as_null_not_zero(tmp_path):
    # Router.total_cost() is None when any model used has no price. Stored as 0
    # it would read as "free", which is the one answer that is known to be wrong.
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.finish_session("s1", Usage(input_tokens=100, output_tokens=20), cost_usd=None)
    row = store.recent_sessions(1)[0]
    assert row.total_cost_usd is None
    assert _stored_cost(store, "s1") is None
    # Everything else about the session is still recorded.
    assert (row.total_input_tokens, row.total_output_tokens) == (100, 20)
    assert row.ended_at is not None


def test_a_known_zero_cost_is_stored_as_zero_and_stays_distinct_from_unknown(tmp_path):
    # A local model is priced at exactly zero. That is a known cost and must not
    # collapse into the unknown one, whichever way the fix for it is written.
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.finish_session("s1", Usage(), cost_usd=0.0)
    assert _stored_cost(store, "s1") == 0.0
    assert store.recent_sessions(1)[0].total_cost_usd == 0.0


def test_a_session_that_has_not_finished_has_no_cost_yet(tmp_path):
    # The column has no default: a row created and never finished says "not
    # known", where the old DEFAULT 0 said "free".
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    assert _stored_cost(store, "s1") is None
    assert store.recent_sessions(1)[0].total_cost_usd is None


def test_a_cost_that_becomes_unknown_replaces_the_one_that_was_known(tmp_path):
    # A session can start on a priced model and switch to an unpriced one. The
    # later total is unknown, and a stale partial figure must not outlive it.
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.finish_session("s1", Usage(input_tokens=10), cost_usd=0.5)
    store.finish_session("s1", Usage(input_tokens=20), cost_usd=None)
    assert _stored_cost(store, "s1") is None


def _transcript(*texts: str) -> Transcript:
    """Alternating user and assistant text messages, starting with the user."""
    roles: tuple[Role, ...] = ("user", "assistant")
    return Transcript(tuple(Message(roles[i % 2], (TextBlock(t),)) for i, t in enumerate(texts)))


def test_replacing_a_transcript_with_a_shorter_one_leaves_none_of_the_old_tail(tmp_path):
    # Compaction swaps a long history for a short one. Writing the short one over
    # the first rows would leave the old tail after it, and the session would
    # reload as the summary followed by messages it was meant to replace.
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    for seq, message in enumerate(_transcript("a", "b", "c", "d", "e", "f").messages):
        store.append_message("s1", seq, message)

    shorter = _transcript("summary", "kept")
    store.replace_transcript("s1", shorter)

    assert store.load_transcript("s1") == shorter


def test_replacing_a_transcript_with_an_empty_one_clears_it(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.replace_transcript("s1", _transcript("a", "b"))
    store.replace_transcript("s1", Transcript())
    assert store.load_transcript("s1") == Transcript()


def test_replacing_one_sessions_transcript_leaves_every_other_session_alone(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    store.create_session("s2", cwd="/p", roles={})
    keep = _transcript("x", "y", "z")
    store.replace_transcript("s2", keep)
    store.replace_transcript("s1", _transcript("only"))
    assert store.load_transcript("s2") == keep
    assert store.load_transcript("s1") == _transcript("only")


def test_a_replacement_that_cannot_be_stored_changes_nothing(tmp_path):
    # The delete and the inserts are one transaction: a failure part-way must
    # leave the old transcript, not an empty one and not half of each.
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    original = _transcript("a", "b", "c")
    store.replace_transcript("s1", original)

    # The first message encodes fine and the second cannot. Nothing is changed:
    # the next test makes the failure come after the first changes were made.
    unstorable = Transcript(
        (original.messages[0], Message("assistant", (TextBlock(object()),)))  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="not JSON serializable"):
        store.replace_transcript("s1", unstorable)

    assert store.load_transcript("s1") == original


# -- the archive: what a replacement removes is kept ---------------------------


def _archived(store: Store, session_id: str) -> list[tuple[int, Message]]:
    """The archived rows of a session in the order they were archived, as (seq, message)."""
    rows = store.db.execute(
        "SELECT seq, role, blocks_json FROM messages_archive WHERE session_id = ? ORDER BY rowid",
        (session_id,),
    ).fetchall()
    return [(row["seq"], Message(row["role"], decode_blocks(row["blocks_json"]))) for row in rows]


def _live_times(store: Store, session_id: str) -> dict[int, float]:
    rows = store.db.execute(
        "SELECT seq, created_at FROM messages WHERE session_id = ? ORDER BY seq", (session_id,)
    ).fetchall()
    return {row["seq"]: row["created_at"] for row in rows}


def _stored_six(store: Store) -> Transcript:
    store.create_session("s1", cwd="/p", roles={})
    stored = _transcript("a", "b", "c", "d", "e", "f")
    for seq, message in enumerate(stored.messages):
        store.append_message("s1", seq, message)
    return stored


def test_what_a_replacement_removes_is_moved_to_the_archive_and_not_lost(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    old = _stored_six(store)

    shorter = _transcript("summary", "kept")
    store.replace_transcript("s1", shorter)

    assert store.load_transcript("s1") == shorter  # the live rows are the working transcript
    assert _archived(store, "s1") == list(enumerate(old.messages))


def test_the_archive_and_the_live_rows_hold_every_message_ever_stored(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    ever = list(_stored_six(store).messages)
    for replacement in (
        _transcript("summary", "kept", "next"),
        _transcript("second summary", "last"),
        Transcript(),
    ):
        store.replace_transcript("s1", replacement)
        ever.extend(replacement.messages)
    live = store.load_transcript("s1").messages
    archived = [message for _, message in _archived(store, "s1")]
    assert all(message in (*live, *archived) for message in ever)


def test_rows_a_replacement_leaves_as_they_were_keep_their_time_and_are_not_archived(
    tmp_path, monkeypatch
):
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    for seq, message in enumerate(_transcript("a", "b", "c", "d").messages):
        store.append_message("s1", seq, message)  # ticks 2 to 5

    changed = _transcript("a", "b", "X", "Y")
    store.replace_transcript("s1", changed)  # tick 6

    assert store.load_transcript("s1") == changed
    assert _live_times(store, "s1") == {0: 2.0, 1: 3.0, 2: 6.0, 3: 6.0}
    assert [seq for seq, _ in _archived(store, "s1")] == [2, 3]


def test_a_message_that_moves_to_another_position_keeps_the_time_it_was_first_stored(
    tmp_path, monkeypatch
):
    # Compaction renumbers the messages it keeps. They are the same messages.
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    for seq, message in enumerate(_transcript("a", "b", "c", "d", "e", "f").messages):
        store.append_message("s1", seq, message)  # ticks 2 to 7: d, e, f at 5, 6, 7

    store.replace_transcript("s1", _transcript("summary", "d", "e", "f"))  # tick 8

    assert _live_times(store, "s1") == {0: 8.0, 1: 5.0, 2: 6.0, 3: 7.0}


def test_the_last_copy_of_a_message_that_comes_back_is_the_one_a_replacement_keeps(
    tmp_path, monkeypatch
):
    # What a replacement keeps it keeps from the end: the compaction that wrote it kept
    # the latest messages, so a message with several copies is the latest of them.
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    for seq, message in enumerate(_transcript("x", "ok", "y", "ok", "z", "ok").messages):
        store.append_message("s1", seq, message)  # ticks 2 to 7: "ok" at 3, 5 and 7

    longer = _transcript("s", "p", "q", "r", "t", "u", "v", "ok")  # "ok" moves to position 7
    store.replace_transcript("s1", longer)  # tick 8

    assert store.load_transcript("s1") == longer
    assert _live_times(store, "s1")[7] == 7.0


def test_a_repeated_message_that_compaction_keeps_has_its_own_time_and_not_an_earlier_copys(
    tmp_path, monkeypatch
):
    # "continue" typed twice, at ticks 2 and 6, and compaction keeps the later one.
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    old = _transcript("continue", "r1", "q", "r2", "continue", "r3")
    for seq, message in enumerate(old.messages):
        store.append_message("s1", seq, message)  # ticks 2 to 7

    kept = _transcript("summary", "r2", "continue", "r3")
    store.replace_transcript("s1", kept)  # tick 8

    assert store.load_transcript("s1") == kept
    assert _live_times(store, "s1") == {0: 8.0, 1: 5.0, 2: 6.0, 3: 7.0}
    # Every message left the position it was at, the kept ones for their new ones.
    assert _archived(store, "s1") == list(enumerate(old.messages))


def test_a_message_repeated_at_every_turn_keeps_its_own_time_through_compaction(
    tmp_path, monkeypatch
):
    # Every user message is "continue", so the one compaction keeps is also what already
    # sat at the position it moves to. Matching by what a message says would leave it
    # there, with the time of an earlier copy, and the times would run out of order.
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    old = _transcript("continue", "r1", "continue", "r2", "continue", "r3", "continue", "r4")
    for seq, message in enumerate(old.messages):
        store.append_message("s1", seq, message)  # ticks 2 to 9

    store.replace_transcript("s1", _transcript("summary", "r3", "continue", "r4"))  # tick 10

    assert _live_times(store, "s1") == {0: 10.0, 1: 7.0, 2: 8.0, 3: 9.0}


def test_a_shorter_replacement_leaves_the_rows_it_starts_with_alone(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    old = _transcript("a", "b", "c", "d")
    for seq, message in enumerate(old.messages):
        store.append_message("s1", seq, message)  # ticks 2 to 5

    store.replace_transcript("s1", _transcript("a", "b", "X"))  # tick 6

    assert _live_times(store, "s1") == {0: 2.0, 1: 3.0, 2: 6.0}
    assert _archived(store, "s1") == [(2, old.messages[2]), (3, old.messages[3])]


def test_a_row_whose_role_changed_is_replaced_even_when_its_text_did_not(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    was = Message("user", (TextBlock("same words"),))
    now = Message("assistant", (TextBlock("same words"),))
    store.append_message("s1", 0, was)
    store.replace_transcript("s1", Transcript((now,)))
    assert store.load_transcript("s1").messages == (now,)
    assert _archived(store, "s1") == [(0, was)]


def test_a_replacement_stamps_a_row_that_changed_at_the_time_of_the_replacement(
    tmp_path, monkeypatch
):
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    store.append_message("s1", 0, _transcript("a").messages[0])  # tick 2
    store.replace_transcript("s1", _transcript("b"))  # tick 3
    assert _live_times(store, "s1") == {0: 3.0}
    archived_at = store.db.execute(
        "SELECT created_at, archived_at FROM messages_archive"
    ).fetchall()
    assert [(row["created_at"], row["archived_at"]) for row in archived_at] == [(2.0, 3.0)]


def test_a_replacement_that_cannot_be_stored_part_way_leaves_both_tables_as_they_were(tmp_path):
    # The first row is kept and the second is archived and deleted before the third
    # is refused: a role the table will not take. All of it is one transaction.
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    original = _transcript("a", "b", "c")
    store.replace_transcript("s1", original)
    before = store.db.execute("SELECT * FROM messages ORDER BY seq").fetchall()

    refused = Transcript(
        (
            original.messages[0],
            Message("assistant", (TextBlock("changed"),)),
            Message(None, (TextBlock("no role"),)),  # type: ignore[arg-type]
        )
    )
    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
        store.replace_transcript("s1", refused)

    assert store.load_transcript("s1") == original
    assert [tuple(row) for row in store.db.execute("SELECT * FROM messages ORDER BY seq")] == [
        tuple(row) for row in before
    ]
    assert _archived(store, "s1") == []


def test_replacing_one_sessions_transcript_archives_nothing_of_another(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s2", cwd="/p", roles={})
    store.replace_transcript("s2", _transcript("x", "y", "z"))
    old = _stored_six(store)
    store.replace_transcript("s1", _transcript("only"))
    assert store.load_transcript("s2") == _transcript("x", "y", "z")
    assert _archived(store, "s2") == []
    assert _archived(store, "s1") == list(enumerate(old.messages))  # none of s2's among them


@pytest.mark.parametrize(
    "other",
    [
        Message("assistant", (TextBlock("same words"),)),
        Message("user", (TextBlock("other words"),)),
    ],
    ids=["another-role", "other-words"],
)
def test_writing_a_different_message_where_one_is_stored_is_refused_and_changes_nothing(
    tmp_path, monkeypatch, other
):
    # Another process has written to the session: the rows are not the ones this writer
    # believes they are, and the audit trail is not for overwriting.
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    stored = Message("user", (TextBlock("same words"),))
    store.append_message("s1", 0, stored)  # tick 2
    with pytest.raises(MessageConflictError) as caught:
        store.append_message("s1", 0, other)  # tick 3
    assert (caught.value.session_id, caught.value.seq) == ("s1", 0)
    assert store.load_transcript("s1").messages == (stored,)
    assert _live_times(store, "s1") == {0: 2.0}
    assert _archived(store, "s1") == []


def test_a_row_the_table_refuses_for_another_reason_is_not_reported_as_a_conflict(tmp_path):
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
        store.append_message("s1", 0, Message(None, (TextBlock("no role"),)))  # type: ignore[arg-type]


def test_writing_the_message_that_is_already_stored_changes_nothing(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch, *(float(n) for n in range(1, 40)))
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})  # tick 1
    message = _transcript("a").messages[0]
    store.append_message("s1", 0, message)  # tick 2
    store.append_message("s1", 0, message)  # tick 3
    assert _live_times(store, "s1") == {0: 2.0}
    assert _archived(store, "s1") == []


def test_a_database_made_before_the_archive_existed_gains_it_and_keeps_its_data(tmp_path):
    path = tmp_path / "s.db"
    earlier = Store(path)
    earlier.open()
    earlier.create_session("s1", cwd="/p", roles={})
    kept = _transcript("a", "b")
    for seq, message in enumerate(kept.messages):
        earlier.append_message("s1", seq, message)
    earlier.db.execute("DROP TABLE messages_archive")  # what a build without it left behind
    earlier.db.commit()
    earlier.close()

    later = Store(path)
    later.open()
    assert later.load_transcript("s1") == kept
    later.replace_transcript("s1", _transcript("summary"))
    assert [seq for seq, _ in _archived(later, "s1")] == [0, 1]


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


# ---- what ncc keeps under its home is private from the moment it exists


def test_a_new_store_makes_its_directory_and_files_private_from_the_start(
    tmp_path, private_from_the_start
):
    from tests.conftest import mode_of

    store = Store(tmp_path / ".nanoclaude" / "sessions.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    assert mode_of(tmp_path / ".nanoclaude") == 0o700
    assert mode_of(store.path) == 0o600
    # The write-ahead log and its index are made by SQLite, with the database's own mode.
    sidecars = [p for p in store.path.parent.iterdir() if p.name.startswith("sessions.db-")]
    assert {p.name for p in sidecars} == {"sessions.db-wal", "sessions.db-shm"}
    assert all(mode_of(p) == 0o600 for p in sidecars), [(p.name, oct(mode_of(p))) for p in sidecars]


def test_a_new_store_is_private_and_works_under_a_restrictive_umask(tmp_path):
    from tests.conftest import mode_of

    previous = os.umask(0o277)
    try:
        store = Store(tmp_path / ".nanoclaude" / "sessions.db")
        store.open()
        store.create_session("s1", cwd="/p", roles={})
    finally:
        os.umask(previous)
    assert (mode_of(tmp_path / ".nanoclaude"), mode_of(store.path)) == (0o700, 0o600)
    assert store.session_row("s1") is not None


def test_a_store_that_is_already_there_keeps_the_mode_it_has(tmp_path):
    from tests.conftest import mode_of

    home = tmp_path / ".nanoclaude"
    home.mkdir(mode=0o755)
    home.chmod(0o755)
    first = Store(home / "sessions.db")
    first.open()
    first.close()
    first.path.chmod(0o644)
    again = Store(home / "sessions.db")
    again.open()
    assert (mode_of(home), mode_of(again.path)) == (0o755, 0o644)
