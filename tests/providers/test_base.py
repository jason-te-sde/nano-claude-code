from collections.abc import Hashable

import pytest

from nanoclaude.conversation.transcript import ToolUseBlock, Transcript, user_text
from nanoclaude.providers.base import ModelReply, ModelRequest, StopKind, ToolSpec, Usage


def test_usage_add_accumulates_all_four_fields():
    # Four distinct values on each side so a transposed or dropped field (e.g. the
    # accumulator crediting cache_write_tokens to cache_read_tokens) shows up as a
    # wrong number instead of accidentally cancelling out.
    total = Usage(100, 20, 5, 3) + Usage(50, 10, 2, 1)
    assert total == Usage(150, 30, 7, 4)


def test_usage_defaults_are_all_zero():
    assert Usage() == Usage(0, 0, 0, 0)


def test_tool_spec_is_not_hashable():
    # schema is a JSON Schema object decoded the same way ToolUseBlock's
    # arguments are (see transcript.py) -- no version of this class has
    # reliably hashable schemas, so it is declared unhashable outright.
    spec = ToolSpec("Read", "Reads a file", {"type": "object"})
    assert not isinstance(spec, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'ToolSpec'"):
        hash(spec)


def test_model_reply_is_not_hashable_when_it_holds_a_tool_use_block():
    # A ModelReply built by says() (TextBlock only) would hash fine -- that is
    # exactly the trap this declaration closes: hashability that depends on
    # the reply's contents would pass a test written with simple values and
    # only fail later, on a real tool-call turn, naming "ToolUseBlock" rather
    # than this class.
    reply = ModelReply(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), StopKind.TOOL_USE, Usage(), "scripted"
    )
    assert not isinstance(reply, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'ModelReply'"):
        hash(reply)


def test_model_request_is_not_hashable():
    # Unconditional, unlike ModelReply above and Message/Transcript in
    # transcript.py: transcript is a required field, and Transcript is itself
    # declared unhashable unconditionally, so every ModelRequest fails
    # regardless of what it carries -- even this one, with no tools and a
    # transcript holding only plain text, which is the shape a prior survey of
    # this codebase took to be "hashable and honest". Left undeclared,
    # dataclass's generated __hash__ would still raise here -- by calling
    # hash() on the embedded Transcript -- but would name "Transcript" instead
    # of this class.
    request = ModelRequest("sys", Transcript((user_text("hi"),)), (), 1024)
    assert not isinstance(request, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'ModelRequest'"):
        hash(request)
