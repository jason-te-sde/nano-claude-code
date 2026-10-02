import pytest

from nanoclaude.conversation.budget import HeuristicCounter, transcript_text
from nanoclaude.conversation.compaction import SUMMARY_TEMPLATE, full_compact, micro_compact
from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    assistant_text,
    user_text,
    validate,
)


def conversation(rounds: int) -> Transcript:
    messages = [user_text("start the task")]
    for index in range(rounds):
        messages.append(
            Message("assistant", (ToolUseBlock(f"t{index}", "Read", {"path": f"f{index}.py"}),))
        )
        messages.append(Message("user", (ToolResultBlock(f"t{index}", "x" * 4000),)))
    messages.append(Message("assistant", (TextBlock("done"),)))
    return Transcript(tuple(messages))


def test_micro_compaction_shrinks_old_tool_results_and_keeps_recent_ones():
    before = conversation(6)
    after = micro_compact(before, keep_recent=3, counter=HeuristicCounter())
    assert len(transcript_text(after)) < len(transcript_text(before))
    assert "x" * 4000 in transcript_text(after)  # the newest survives intact


def test_micro_compaction_keeps_the_transcript_valid():
    after = micro_compact(conversation(6), keep_recent=3, counter=HeuristicCounter())
    validate(after)


def test_micro_compaction_never_drops_a_tool_result():
    before = conversation(5)
    after = micro_compact(before, keep_recent=2, counter=HeuristicCounter())
    assert len(after.messages) == len(before.messages)
    for original, compacted in zip(before.messages, after.messages, strict=True):
        assert len(original.blocks) == len(compacted.blocks)


def test_micro_compaction_says_what_it_replaced():
    after = micro_compact(conversation(6), keep_recent=1, counter=HeuristicCounter())
    assert "[compacted:" in transcript_text(after)


async def test_full_compaction_replaces_history_with_one_summary():
    async def summarise(text: str) -> str:
        return "GOAL: ship it\nCHANGED: a.py\nOPEN: none"

    before = conversation(8)
    after = await full_compact(before, keep_recent=2, summarise=summarise)
    validate(after)
    assert len(after.messages) < len(before.messages)
    assert "GOAL: ship it" in transcript_text(after)


async def test_full_compaction_keeps_the_last_user_message():
    async def summarise(text: str) -> str:
        return "summary"

    before = Transcript(
        (
            *conversation(4).messages,
            user_text("and now do this other thing"),
        )
    )
    after = await full_compact(before, keep_recent=1, summarise=summarise)
    assert "and now do this other thing" in transcript_text(after)


async def test_full_compaction_preserves_the_set_of_files_touched():
    """Losing which files were changed is the failure that costs real work."""

    async def summarise(text: str) -> str:
        return "summary"

    before = conversation(5)
    after = await full_compact(before, keep_recent=1, summarise=summarise)
    for index in range(5):
        assert f"f{index}.py" in transcript_text(after)


async def test_full_compaction_refuses_to_compact_an_unanswered_turn():
    async def summarise(text: str) -> str:
        return "summary"

    pending = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolUseBlock("t1", "Read", {}),)),
        )
    )
    with pytest.raises(ValueError, match="unanswered"):
        await full_compact(pending, keep_recent=1, summarise=summarise)


# The tests below extend the brief's own cases: byte-for-byte proof that
# keep_recent is untouched (not just "the big string survives somewhere"), the
# short-tool-result and non-tool-result paths micro_compact's threshold check
# leaves alone, every early-return guard in full_compact, the two-part "did we
# find any files" branch, and SUMMARY_TEMPLATE -- the one Produces name
# neither test file otherwise names directly.


def test_micro_compaction_keeps_recent_turns_byte_for_byte():
    before = conversation(6)
    keep_recent = 3
    after = micro_compact(before, keep_recent=keep_recent, counter=HeuristicCounter())
    # Equality on the dataclasses compares every field, recursively -- this is
    # stronger than "the big string is still in there somewhere".
    assert after.messages[-keep_recent * 2 :] == before.messages[-keep_recent * 2 :]


def test_micro_compaction_leaves_short_old_tool_results_and_stray_blocks_untouched():
    before = Transcript(
        (
            user_text("start"),
            Message("assistant", (ToolUseBlock("t0", "Read", {"path": "f0.py"}),)),
            Message("user", (ToolResultBlock("t0", "short"), TextBlock("a stray note"))),
            Message("assistant", (ToolUseBlock("t1", "Read", {"path": "f1.py"}),)),
            Message("user", (ToolResultBlock("t1", "y" * 4000),)),
            Message("assistant", (ToolUseBlock("t2", "Read", {"path": "f2.py"}),)),
            Message("user", (ToolResultBlock("t2", "z" * 4000),)),
            Message("assistant", (TextBlock("done"),)),
        )
    )
    after = micro_compact(before, keep_recent=1, counter=HeuristicCounter())
    validate(after)
    # Old and short (<=50 tokens): the stray TextBlock beside it is likewise
    # under no threshold at all, so both survive byte for byte.
    assert after.messages[2] == before.messages[2]
    # Old but large: the threshold's other half, proving the previous
    # assertion is the threshold at work and not the loop being a no-op.
    assert after.messages[4] != before.messages[4]
    assert "y" * 4000 not in transcript_text(after)
    # Newest tool result, inside keep_recent: untouched even though large.
    assert after.messages[6] == before.messages[6]
    assert "z" * 4000 in transcript_text(after)


async def test_full_compaction_does_not_split_a_tool_use_from_its_result_at_the_cut():
    """A naive slice of the last keep_recent*2 messages can start on the user
    message holding the tool_result for a tool_use left behind in head --
    splitting a call from its answer. conversation(3) with keep_recent=1 is
    the minimal case where the naive cut falls exactly on that boundary.
    """

    async def summarise(text: str) -> str:
        return "summary"

    after = await full_compact(conversation(3), keep_recent=1, summarise=summarise)
    validate(after)


async def test_full_compaction_omits_files_seen_when_nothing_was_touched():
    async def summarise(text: str) -> str:
        return "summary"

    before = Transcript(
        (
            user_text("hello"),
            assistant_text("hi there"),
            user_text("how are you"),
            assistant_text("fine"),
            user_text("tell me a joke"),
            assistant_text("knock knock"),
        )
    )
    after = await full_compact(before, keep_recent=1, summarise=summarise)
    validate(after)
    assert "FILES SEEN" not in transcript_text(after)


async def test_full_compaction_ignores_non_string_tool_arguments_when_collecting_files():
    async def summarise(text: str) -> str:
        return "summary"

    before = Transcript(
        (
            user_text("start"),
            Message("assistant", (ToolUseBlock("t0", "Bash", {"timeout": 30, "path": "f0.py"}),)),
            Message("user", (ToolResultBlock("t0", "ok"),)),
            Message("assistant", (ToolUseBlock("t1", "Read", {"path": "f1.py"}),)),
            Message("user", (ToolResultBlock("t1", "ok"),)),
            Message("assistant", (TextBlock("done"),)),
        )
    )
    after = await full_compact(before, keep_recent=1, summarise=summarise)
    validate(after)
    assert "f0.py" in transcript_text(after)
    assert "30" not in transcript_text(after)


async def test_full_compaction_returns_unchanged_when_already_small_enough():
    async def summarise(text: str) -> str:
        raise AssertionError("summarise must not run when nothing needs compacting")

    before = conversation(1)
    after = await full_compact(before, keep_recent=5, summarise=summarise)
    assert after is before


async def test_full_compaction_returns_unchanged_when_no_safe_cut_point_exists():
    """full_compact only checks the *last* message for a pending tool_use, not
    that the whole transcript alternates -- so a malformed transcript (three
    consecutive user messages at the end) can empty the tail entirely as the
    loop hunts for an assistant message to start it on. There is then no safe
    split, and the transcript must come back unchanged rather than silently
    emitting a two-message, assistant-less result.
    """

    async def summarise(text: str) -> str:
        raise AssertionError("summarise must not run when there is no safe cut")

    malformed = Transcript(
        (
            user_text("a"),
            assistant_text("b"),
            user_text("c"),
            user_text("d"),
            user_text("e"),
        )
    )
    after = await full_compact(malformed, keep_recent=1, summarise=summarise)
    assert after is malformed


async def test_full_compaction_propagates_a_summarise_failure_without_mutating_the_input():
    async def summarise(text: str) -> str:
        raise RuntimeError("model unavailable")

    before = conversation(6)
    snapshot = before.messages
    with pytest.raises(RuntimeError, match="model unavailable"):
        await full_compact(before, keep_recent=2, summarise=summarise)
    # Transcript is frozen, so there is no way for it to come back
    # half-replaced; pinned anyway, so a future rewrite that builds the result
    # by mutation would be caught here.
    assert before.messages == snapshot


def test_summary_template_has_every_required_heading_in_order():
    headings = ["GOAL:", "DECISIONS:", "CHANGED:", "OPEN:", "FAILED:"]
    positions = [SUMMARY_TEMPLATE.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert SUMMARY_TEMPLATE.startswith("Summarise this coding session")
    assert SUMMARY_TEMPLATE.endswith("Session:\n")


@pytest.mark.parametrize("keep_recent", [0, -1])
async def test_full_compaction_refuses_to_keep_fewer_than_one_recent_turn(keep_recent):
    async def summarise(_text: str) -> str:
        return "summary"

    with pytest.raises(ValueError, match="keep_recent must be at least 1"):
        await full_compact(conversation(6), keep_recent=keep_recent, summarise=summarise)
