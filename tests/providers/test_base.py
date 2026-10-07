import os
import re
import subprocess
import sys
from collections.abc import Hashable
from pathlib import Path

import pytest

from nanoclaude.conversation.transcript import TextBlock, ToolUseBlock, Transcript, user_text
from nanoclaude.providers.base import (
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    ToolSpec,
    Usage,
    new_call_id,
    partial_reply,
)


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


def test_a_minted_call_id_is_the_prefix_and_twelve_hex_characters():
    assert re.fullmatch(r"call_[0-9a-f]{12}", new_call_id("call"))
    assert re.fullmatch(r"tt_[0-9a-f]{12}", new_call_id("tt"))


def test_minted_call_ids_do_not_repeat_within_a_process():
    # Two calls in one conversation must never share an id; chance gets no say in it.
    ids = {new_call_id("call") for _ in range(2_000)}
    assert len(ids) == 2_000


def test_a_new_process_does_not_mint_the_ids_an_earlier_one_did():
    # What a resumed conversation depends on: it already holds the ids the process
    # that wrote it made, and a counter would start again at the same number.
    code = "from nanoclaude.providers.base import new_call_id; print(new_call_id('call'))"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    minted = [
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    ]
    assert all(re.fullmatch(r"call_[0-9a-f]{12}", one) for one in minted)
    assert minted[0] != minted[1]


def test_an_error_carries_no_partial_reply_unless_it_is_given_one():
    assert ModelError("the provider is down").partial is None


def test_an_error_holds_what_had_arrived_of_a_reply_that_broke_off():
    arrived = ModelReply((TextBlock("half an ans"),), StopKind.CUT_OFF, Usage(10, 2), "m")
    error = ModelError("the connection was lost", partial=arrived)
    assert error.partial is arrived
    assert error.retryable is False


def test_a_reply_that_was_partly_received_cannot_be_marked_retryable():
    # Asking again would show the person the same words twice, and run a call the
    # first attempt may already have made (spec 7.4). An adapter that built such an
    # error has a bug, and finds out when it is built rather than when it is retried.
    arrived = ModelReply((TextBlock("half"),), StopKind.CUT_OFF, Usage(), "m")
    with pytest.raises(ValueError, match="never retried"):
        ModelError("the connection was lost", retryable=True, partial=arrived)


def test_a_partial_reply_is_the_text_that_arrived_and_nothing_else():
    arrived = partial_reply(["Hello", "", ", there"], Usage(5, 1), "m")
    assert arrived == ModelReply(
        (TextBlock("Hello"), TextBlock(", there")), StopKind.CUT_OFF, Usage(5, 1), "m"
    )


def test_a_partial_reply_with_no_text_is_a_reply_with_no_blocks():
    # What a stream that broke inside its first tool call leaves: nothing to keep, and
    # still a reply, so that what the provider had billed by then is not lost.
    arrived = partial_reply([], Usage(7, 0), "m")
    assert arrived.blocks == ()
    assert arrived.usage == Usage(7, 0)
