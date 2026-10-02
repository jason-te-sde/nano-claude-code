from nanoclaude.conversation.transcript import TextBlock, ToolUseBlock
from nanoclaude.providers.base import ModelReply, StopKind, ToolSpec, Usage
from nanoclaude.providers.texttools import (
    MAX_PARSE_RETRIES,
    TOOL_PROTOCOL_PROMPT,
    parse_text_tools,
    render_tools,
    wrap_reply,
)

SPECS = (
    ToolSpec(
        "Read",
        "Read a file.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
)

TWO_SPECS = (
    ToolSpec("Read", "Read a file.", {"type": "object"}),
    ToolSpec("Write", "Write a file.", {"type": "object"}),
)


def ids():
    counter = iter(range(1, 99))
    return lambda: f"tt_{next(counter)}"


def test_a_single_call_is_parsed():
    text = 'Let me look.\n<tool name="Read">\n{"path": "a.py"}\n</tool>'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert problems == []
    assert isinstance(blocks[0], TextBlock) and blocks[0].text.strip() == "Let me look."
    assert isinstance(blocks[1], ToolUseBlock)
    assert blocks[1].name == "Read" and blocks[1].arguments == {"path": "a.py"}


def test_several_calls_are_parsed_in_order():
    text = '<tool name="Read">{"path": "a"}</tool>\n<tool name="Read">{"path": "b"}</tool>'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert problems == []
    assert [b.arguments["path"] for b in blocks if isinstance(b, ToolUseBlock)] == ["a", "b"]


def test_prose_around_and_between_calls_is_kept():
    text = 'first\n<tool name="Read">{"path":"a"}</tool>\nthen this'
    blocks, _ = parse_text_tools(text, next_id=ids())
    assert [type(b).__name__ for b in blocks] == ["TextBlock", "ToolUseBlock", "TextBlock"]


def test_malformed_json_is_reported_rather_than_guessed():
    text = '<tool name="Read">{"path": </tool>'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert not any(isinstance(b, ToolUseBlock) for b in blocks)
    assert problems and "Read" in problems[0]


def test_a_missing_closing_tag_is_reported():
    text = '<tool name="Read">{"path": "a"}'
    _, problems = parse_text_tools(text, next_id=ids())
    assert problems and "closing" in problems[0].lower()


def test_fenced_code_around_the_block_is_tolerated():
    """Small models wrap everything in triple backticks. That is not an error."""
    text = '```\n<tool name="Read">{"path": "a"}</tool>\n```'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert problems == []
    assert any(isinstance(b, ToolUseBlock) for b in blocks)


def test_plain_prose_with_no_calls_parses_to_one_text_block():
    blocks, problems = parse_text_tools("just talking", next_id=ids())
    assert problems == [] and len(blocks) == 1


def test_wrap_reply_upgrades_the_stop_reason_when_calls_were_found():
    reply = ModelReply(
        (TextBlock('<tool name="Read">{"path":"a"}</tool>'),),
        StopKind.END_TURN,
        Usage(),
        "local",
    )
    wrapped = wrap_reply(reply, next_id=ids())
    assert wrapped.stop is StopKind.TOOL_USE
    assert any(isinstance(b, ToolUseBlock) for b in wrapped.blocks)


def test_wrap_reply_leaves_a_plain_reply_alone():
    reply = ModelReply((TextBlock("hello"),), StopKind.END_TURN, Usage(), "local")
    assert wrap_reply(reply, next_id=ids()).stop is StopKind.END_TURN


def test_rendered_tools_include_the_name_description_and_schema():
    rendered = render_tools(SPECS)
    assert "Read" in rendered and "Read a file." in rendered and '"path"' in rendered


# --- Beyond the brief's own tests: the extra guarantees the task calls out. ---


def test_several_calls_with_no_prose_produce_no_stray_text_blocks():
    """The falsy half of "if prose:" / "if tail:": whitespace-only gaps between
    two calls must not turn into an empty TextBlock."""
    text = '<tool name="Read">{"path": "a"}</tool>\n<tool name="Read">{"path": "b"}</tool>'
    blocks, _ = parse_text_tools(text, next_id=ids())
    assert [type(b).__name__ for b in blocks] == ["ToolUseBlock", "ToolUseBlock"]


def test_prose_between_two_tool_calls_is_kept_in_order():
    text = (
        '<tool name="Read">{"path": "a"}</tool>\n'
        "middle text\n"
        '<tool name="Read">{"path": "b"}</tool>'
    )
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert problems == []
    assert [type(b).__name__ for b in blocks] == ["ToolUseBlock", "TextBlock", "ToolUseBlock"]
    middle = next(b for b in blocks if isinstance(b, TextBlock))
    assert middle.text.strip() == "middle text"


def test_ids_from_next_id_are_assigned_in_order_and_are_unique():
    text = (
        '<tool name="Read">{"path": "a"}</tool>'
        '<tool name="Read">{"path": "b"}</tool>'
        '<tool name="Read">{"path": "c"}</tool>'
    )
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert problems == []
    calls = [b for b in blocks if isinstance(b, ToolUseBlock)]
    assert [c.id for c in calls] == ["tt_1", "tt_2", "tt_3"]
    assert len(calls) == len({c.id for c in calls})


def test_arguments_that_parse_to_a_list_are_a_complaint_not_a_call():
    text = '<tool name="Read">[1, 2, 3]</tool>'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert not any(isinstance(b, ToolUseBlock) for b in blocks)
    assert problems and "list" in problems[0] and "Read" in problems[0]
    # Nothing else parsed from the text either, so the fallback that guarantees
    # at least one block surfaces the whole original attempt instead of
    # dropping it silently -- the parser never invents arguments, but it also
    # never erases a call it refused to guess at.
    assert blocks == [TextBlock(text)]


def test_arguments_that_parse_to_a_string_are_a_complaint_not_a_call():
    text = '<tool name="Read">"just a string"</tool>'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert not any(isinstance(b, ToolUseBlock) for b in blocks)
    assert problems and "str" in problems[0]


def test_arguments_that_parse_to_a_number_are_a_complaint_not_a_call():
    text = '<tool name="Read">42</tool>'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert not any(isinstance(b, ToolUseBlock) for b in blocks)
    assert problems and "int" in problems[0]


def test_a_language_tagged_fence_has_its_backticks_stripped():
    """``` fences small models annotate with a language (` ```python`) are not
    one of the two exact tokens treated as pure noise, so the generic
    backtick-stripping path runs instead -- it still does not break the call."""
    text = '```python\n<tool name="Read">{"path": "a"}</tool>\n```'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert problems == []
    assert [type(b).__name__ for b in blocks] == ["TextBlock", "ToolUseBlock"]
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "python"


def test_text_after_a_missing_closing_tag_is_still_captured_as_prose():
    text = '<tool name="Read">{"path": "a"} and more text with no closing'
    blocks, problems = parse_text_tools(text, next_id=ids())
    assert problems and "closing" in problems[0].lower()
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == '{"path": "a"} and more text with no closing'


def test_wrap_reply_preserves_usage_and_model_when_nothing_parses_as_a_call():
    """ "<tool" can appear in prose without ever forming a real open tag (no
    name attribute here). No calls and no problems come out of parsing it, so
    this exercises the "no calls" half of the stop-upgrade decision by a path
    other than the short-circuit for text with no "<tool" substring at all."""
    reply = ModelReply(
        (TextBlock("I mentioned <tool once, nothing more"),),
        StopKind.END_TURN,
        Usage(5, 5, 1, 1),
        "local",
    )
    wrapped = wrap_reply(reply, next_id=ids())
    assert wrapped == reply


def test_wrap_reply_reports_problems_without_upgrading_stop_when_no_call_survives():
    reply = ModelReply(
        (TextBlock('<tool name="Read">{"path": </tool>'),),
        StopKind.END_TURN,
        Usage(1, 2, 3, 4),
        "local",
    )
    wrapped = wrap_reply(reply, next_id=ids())
    assert wrapped.stop is StopKind.END_TURN
    assert wrapped.usage == Usage(1, 2, 3, 4)
    assert wrapped.model == "local"
    assert not any(isinstance(b, ToolUseBlock) for b in wrapped.blocks)
    assert isinstance(wrapped.blocks[0], TextBlock)
    assert "[tool protocol]" in wrapped.blocks[0].text
    assert "not valid JSON" in wrapped.blocks[0].text


def test_max_parse_retries_is_two():
    assert MAX_PARSE_RETRIES == 2


def test_tool_protocol_prompt_is_byte_identical_to_the_brief():
    assert (
        TOOL_PROTOCOL_PROMPT
        == """\
You do not have a function-calling interface. To use a tool, write a block in \
exactly this form and nothing else on those lines:

<tool name="ToolName">
{"argument": "value"}
</tool>

Rules:
- The content between the tags must be a single valid JSON object.
- One block per tool call. You may write several blocks in one reply.
- Write the block exactly as shown. Do not rename the tags or add attributes.
- You may write ordinary prose before or after a block; it is shown to the user.
- After a block, stop and wait. The result comes back in the next message.

Available tools:
"""
    )


def test_render_tools_lists_every_spec_exactly_once():
    rendered = render_tools(TWO_SPECS)
    assert rendered.startswith(TOOL_PROTOCOL_PROMPT)
    assert rendered.count("### Read") == 1
    assert rendered.count("### Write") == 1
