from collections.abc import Hashable

import pytest

from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    TranscriptError,
    assistant_text,
    user_text,
    validate,
)


def test_valid_conversation_passes():
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolUseBlock("t1", "Read", {"path": "a.py"}),)),
            Message("user", (ToolResultBlock("t1", "1\tx = 1"),)),
            Message("assistant", (TextBlock("it sets x"),)),
        )
    )
    validate(t)


def test_roles_must_alternate_starting_with_user():
    t = Transcript((Message("assistant", (TextBlock("hi"),)),))
    with pytest.raises(TranscriptError, match="must start with a user message"):
        validate(t)


def test_unanswered_tool_use_is_only_legal_at_the_end():
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolUseBlock("t1", "Read", {}),)),
            Message("user", (TextBlock("never mind"),)),
        )
    )
    # Pinned to the exact clause, not just the substring "tool_result": four
    # of validate()'s ten error sites contain that substring, so a weaker
    # match would keep passing under a regression that broke this specific
    # ordered-pairing check (a loosened length comparison, an off-by-one)
    # as long as whatever fired instead happened to mention tool_result too.
    with pytest.raises(TranscriptError, match="needs exactly one tool_result, in order"):
        validate(t)


def test_pending_tool_uses_reports_the_trailing_group():
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolUseBlock("t1", "Read", {}), ToolUseBlock("t2", "Glob", {}))),
        )
    )
    validate(t)
    assert [b.id for b in t.pending_tool_uses()] == ["t1", "t2"]


def test_duplicate_tool_use_ids_are_rejected():
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolUseBlock("t1", "Read", {}),)),
            Message("user", (ToolResultBlock("t1", "ok"),)),
            Message("assistant", (ToolUseBlock("t1", "Read", {}),)),
        )
    )
    with pytest.raises(TranscriptError, match="duplicate"):
        validate(t)


def test_tool_arguments_cannot_be_mutated_after_construction():
    args = {"path": "a.py"}
    block = ToolUseBlock("t1", "Read", args)
    args["path"] = "b.py"
    assert block.arguments["path"] == "a.py"
    with pytest.raises(TypeError):
        block.arguments["path"] = "c.py"  # type: ignore[index]


def test_tool_arguments_freeze_reaches_nested_lists_and_dicts():
    # A shallow copy would still share a list nested inside arguments with
    # whatever the caller holds, so appending to it after construction would
    # silently change what this block reports -- the exact shape Edit and
    # Grep use (a list of edit objects, a list of patterns).
    args = {"edits": [{"old_string": "a", "new_string": "b"}]}
    block = ToolUseBlock("t1", "Edit", args)
    args["edits"].append({"old_string": "STOLEN", "new_string": "x"})
    assert block.arguments["edits"] == [{"old_string": "a", "new_string": "b"}]


def test_empty_messages_are_rejected():
    t = Transcript((Message("user", ()),))
    with pytest.raises(TranscriptError, match="no blocks"):
        validate(t)


# The tests below close two gaps found after the initial implementation: every
# branch of validate() gets a test that pins the specific error message, not
# just the exception type, and the public helpers left untested by the tests
# above are exercised directly.


def test_empty_transcript_is_valid():
    # No messages at all is not the same violation as a transcript that starts
    # with the wrong role; validate() returns rather than raising.
    validate(Transcript(()))


def test_role_violation_after_the_first_message_names_the_index():
    t = Transcript(
        (
            user_text("hi"),
            user_text("again"),  # index 1 must be "assistant", not "user"
        )
    )
    with pytest.raises(TranscriptError, match="message 1 has role 'user', expected 'assistant'"):
        validate(t)


def test_tool_use_in_a_user_message_is_rejected():
    t = Transcript((Message("user", (ToolUseBlock("t1", "Read", {}),)),))
    with pytest.raises(TranscriptError, match="tool_use in user message 0"):
        validate(t)


def test_tool_result_in_an_assistant_message_is_rejected():
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolResultBlock("t1", "ok"),)),
        )
    )
    with pytest.raises(TranscriptError, match="tool_result in assistant message 1"):
        validate(t)


def test_thinking_block_in_a_user_message_is_rejected():
    # Thinking is provider-side reasoning that only ever comes from an
    # assistant turn; Anthropic itself rejects it in a user message.
    t = Transcript((Message("user", (ThinkingBlock("reasoning", "sig"),)),))
    with pytest.raises(TranscriptError, match="thinking in user message 0"):
        validate(t)


def test_thinking_block_in_an_assistant_message_is_valid():
    # validate() not raising is the whole signal here -- the positive half of
    # the rule the test above pins the negative half of. Without this, a
    # regression that rejected ThinkingBlock unconditionally (dropping the
    # role check instead of getting it wrong) would still leave that test
    # green while breaking every legitimate use of the class.
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ThinkingBlock("reasoning", "sig"),)),
        )
    )
    validate(t)


def test_tool_result_in_message_zero_is_rejected():
    t = Transcript((Message("user", (ToolResultBlock("t1", "ok"),)),))
    with pytest.raises(TranscriptError, match="message 0 cannot contain tool_result blocks"):
        validate(t)


def test_tool_result_answering_an_unrequested_id_is_rejected():
    # message 1 has no tool_use at all, so the ordered-pairing check that
    # would otherwise fire first never runs -- this reaches the separate
    # check that a tool_result's id was actually offered.
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (TextBlock("nothing to do"),)),
            Message("user", (ToolResultBlock("t1", "ok"),)),
        )
    )
    with pytest.raises(TranscriptError, match="answers a tool_use that was never requested"):
        validate(t)


def test_message_text_joins_only_text_blocks():
    m = Message(
        "assistant",
        (TextBlock("first"), ToolUseBlock("t1", "Read", {}), TextBlock("second")),
    )
    assert m.text() == "first\nsecond"


def test_transcript_append_adds_a_message_without_changing_the_original():
    t = Transcript((user_text("hi"),))
    appended = t.append(assistant_text("hello"))
    assert appended.messages == (user_text("hi"), assistant_text("hello"))
    assert t.messages == (user_text("hi"),)


def test_transcript_len_counts_messages():
    t = Transcript((user_text("hi"), assistant_text("hello")))
    assert len(t) == 2


def test_assistant_text_builds_an_assistant_message_with_one_text_block():
    assert assistant_text("hello") == Message("assistant", (TextBlock("hello"),))


def test_pending_tool_uses_is_empty_for_an_empty_transcript():
    assert Transcript(()).pending_tool_uses() == ()


def test_pending_tool_uses_is_empty_when_the_last_message_is_from_the_user():
    t = Transcript((user_text("hi"),))
    assert t.pending_tool_uses() == ()


def test_tool_use_block_is_not_hashable():
    # Locks in the decision at transcript.py's ToolUseBlock: arguments is
    # decoded JSON and can hold lists or nested objects, so no version of this
    # class can be reliably hashable. Asserted two ways -- the type itself
    # says so, and using it as a hash actually fails -- so a later change that
    # makes one true without the other is caught.
    block = ToolUseBlock("t1", "Read", {"path": "a.py"})
    # mypy proves ToolUseBlock and Hashable can never intersect (that is
    # exactly the point being tested), and flags the isinstance check itself
    # rather than helping assert it; same shape as the arguments[...]
    # assignment above, where the static type already forbids what the test
    # exists to confirm still fails correctly at runtime.
    assert not isinstance(block, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError):
        hash(block)
