import pytest

from nanoclaude.conversation.transcript import TextBlock, ToolUseBlock, Transcript, user_text
from nanoclaude.providers.base import ModelClient, ModelError, ModelRequest, StopKind, Usage
from nanoclaude.testing.scripted import ScriptedModel, calls, calls_many, says

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
