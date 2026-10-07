import pytest

from nanoclaude.conversation.transcript import TextBlock, ToolUseBlock, Transcript, user_text
from nanoclaude.providers.base import ModelClient, ModelError, ModelRequest, StopKind, Usage
from nanoclaude.testing.scripted import ScriptedModel, calls, calls_many, cut_off, says

# A static check, not a runtime one: mypy --strict verifies ScriptedModel's shape
# (the model_id property, the signature of complete() including async and the
# return type) against the ModelClient Protocol right here. ModelClient itself is
# not otherwise imported or checked anywhere until Task 17's real adapters arrive,
# so this is what would catch the two shapes drifting apart in the meantime.
_conforms: ModelClient = ScriptedModel([])


def request() -> ModelRequest:
    return ModelRequest("sys", Transcript((user_text("hi"),)), (), 1024)


async def test_returns_replies_in_order_and_records_requests():
    model = ScriptedModel([says("one"), says("two")])
    first = (await model.complete(request())).blocks[0]
    second = (await model.complete(request())).blocks[0]
    # blocks is a tuple of the transcript's Block union, not Any, so reading .text
    # back out needs the same isinstance narrowing any real caller would have to do.
    assert isinstance(first, TextBlock)
    assert first.text == "one"
    assert isinstance(second, TextBlock)
    assert second.text == "two"
    assert len(model.requests) == 2


async def test_running_out_of_script_is_a_loud_error():
    model = ScriptedModel([])
    with pytest.raises(ModelError, match="ran out of replies"):
        await model.complete(request())


def test_model_id_defaults_to_scripted_and_can_be_overridden():
    assert ScriptedModel([]).model_id == "scripted"
    assert ScriptedModel([], model_id="local-qwen").model_id == "local-qwen"


async def test_exhausted_tracks_whether_replies_remain():
    model = ScriptedModel([says("one")])
    assert not model.exhausted
    await model.complete(request())
    assert model.exhausted


async def test_calls_builds_a_tool_use_reply():
    reply = calls("Read", {"path": "a.py"}, call_id="t1", preamble="reading")
    assert reply.stop is StopKind.TOOL_USE
    preamble, tool_use = reply.blocks
    assert isinstance(preamble, TextBlock)
    assert preamble.text == "reading"
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.id == "t1"


async def test_calls_many_preserves_order():
    reply = calls_many(("Read", {"path": "a"}, "t1"), ("Glob", {"pattern": "*"}, "t2"))
    ids = [block.id for block in reply.blocks if isinstance(block, ToolUseBlock)]
    assert ids == ["t1", "t2"]
    assert len(ids) == len(reply.blocks)  # every block really was a ToolUseBlock


async def test_calls_defaults_usage_to_zero():
    assert calls("Read", {"path": "a.py"}).usage == Usage()


async def test_calls_accepts_explicit_token_usage():
    # calls() used to hardcode Usage() while says() already took input_tokens/
    # output_tokens -- an asymmetry that forced test_usage_accumulates_across_turns
    # in test_loop.py to hand-build a ModelReply just to get non-zero usage on a
    # tool-call turn. Mirrors says()'s own kwargs exactly.
    reply = calls("Read", {"path": "a.py"}, input_tokens=10, output_tokens=3)
    assert reply.usage == Usage(10, 3)


async def test_calls_many_defaults_usage_to_zero():
    assert calls_many(("Read", {}, "t1")).usage == Usage()


async def test_calls_many_accepts_explicit_token_usage():
    reply = calls_many(("Read", {}, "t1"), input_tokens=10, output_tokens=3)
    assert reply.usage == Usage(10, 3)


async def test_a_reply_is_streamed_to_on_text_in_a_few_pieces_that_join_to_its_text():
    # In one piece, a double would hide what streaming exists to expose: a front end that
    # copes with the first piece and nothing after it.
    model = ScriptedModel([says("hello there")])
    pieces: list[str] = []
    reply = await model.complete(request(), on_text=pieces.append)
    assert len(pieces) > 1
    assert all(pieces)  # an empty piece is no text, and a front end may take it for the first
    assert "".join(pieces) == "hello there"
    assert reply == says("hello there")  # the reply itself is as scripted


async def test_every_text_block_is_streamed_in_order_and_a_tool_call_is_not_text():
    model = ScriptedModel([calls("Read", {"path": "src/app.py"}, preamble="reading it")])
    pieces: list[str] = []
    await model.complete(request(), on_text=pieces.append)
    assert "".join(pieces) == "reading it"


async def test_a_reply_with_no_text_streams_nothing():
    model = ScriptedModel([calls("Read", {"path": "a.py"})])
    pieces: list[str] = []
    await model.complete(request(), on_text=pieces.append)
    assert pieces == []


async def test_a_text_block_with_nothing_in_it_streams_nothing():
    model = ScriptedModel([says("")])
    pieces: list[str] = []
    await model.complete(request(), on_text=pieces.append)
    assert pieces == []


async def test_a_model_that_is_not_asked_to_stream_hands_over_the_reply_all_the_same():
    model = ScriptedModel([says("hello there")])
    assert await model.complete(request()) == says("hello there")


def test_cut_off_is_an_error_that_is_not_retryable_and_holds_the_text_that_arrived():
    error = cut_off("half an ans", input_tokens=40, output_tokens=3)
    assert isinstance(error, ModelError)
    assert error.retryable is False
    assert error.partial is not None
    assert error.partial.blocks == (TextBlock("half an ans"),)
    assert error.partial.usage == Usage(40, 3)
    assert error.partial.stop is StopKind.CUT_OFF
