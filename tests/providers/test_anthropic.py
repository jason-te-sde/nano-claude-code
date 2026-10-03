import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    user_text,
)
from nanoclaude.providers.anthropic import (
    DEFAULT_MODEL,
    AnthropicClient,
    StreamAccumulator,
    encode_messages,
    encode_tools,
    iter_sse,
)
from nanoclaude.providers.base import ModelClient, ModelError, ModelRequest, StopKind, ToolSpec

CASSETTES = Path(__file__).resolve().parents[1] / "cassettes"


def replay(name: str) -> StreamAccumulator:
    accumulator = StreamAccumulator()
    for line in (CASSETTES / name).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        accumulator.handle(event["event"], event["data"])
    return accumulator


def test_a_text_reply_decodes_to_one_text_block():
    reply = replay("anthropic_text.jsonl").result()
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.stop is StopKind.END_TURN
    assert reply.usage.input_tokens > 0


def test_a_tool_call_decodes_with_its_arguments_reassembled():
    reply = replay("anthropic_tool_use.jsonl").result()
    call = next(b for b in reply.blocks if isinstance(b, ToolUseBlock))
    assert call.name == "Read"
    assert call.arguments["path"].endswith(".py")
    assert reply.stop is StopKind.TOOL_USE


def test_parallel_tool_calls_keep_their_order():
    reply = replay("anthropic_parallel_tools.jsonl").result()
    calls = [b for b in reply.blocks if isinstance(b, ToolUseBlock)]
    assert len(calls) >= 2
    assert len({c.id for c in calls}) == len(calls)
    # Not just "two calls somewhere" -- the order the cassette opened their
    # content blocks in (README.md first) must survive to the decoded reply.
    assert calls[0].arguments["path"] == "README.md"
    assert calls[1].arguments["path"] == "pyproject.toml"


def test_a_stream_that_ends_mid_tool_call_is_a_usable_error():
    """Review Focus #1. Retrying is wrong: earlier tool calls in the turn may
    already have run. The caller must be told, not silently handed {}.
    """
    with pytest.raises(ModelError, match="incomplete") as excinfo:
        replay("anthropic_truncated.jsonl").result()
    assert excinfo.value.retryable is False


def test_an_error_event_becomes_a_model_error():
    accumulator = StreamAccumulator()
    with pytest.raises(ModelError, match="overloaded"):
        accumulator.handle(
            "error", {"error": {"type": "overloaded_error", "message": "overloaded"}}
        )


def test_a_non_overloaded_error_event_is_not_retryable():
    # The retryable flag on an "error" event depends on the provider's own
    # error type, not just on whether we raised at all -- the other half of
    # that membership test, untouched by the overloaded case above.
    accumulator = StreamAccumulator()
    with pytest.raises(ModelError) as excinfo:
        accumulator.handle(
            "error", {"error": {"type": "invalid_request_error", "message": "bad request"}}
        )
    assert not excinfo.value.retryable


def test_a_stream_with_no_content_blocks_at_all_is_a_usable_error():
    # A complete stream (message_start through message_stop) that never opens
    # a single content block -- result()'s other refusal branch, distinct
    # from the mid-tool-call one above: nothing was cut off here, there was
    # simply nothing to return.
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 5}}})
    accumulator.handle("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {}})
    accumulator.handle("message_stop", {})
    with pytest.raises(ModelError, match="no content blocks"):
        accumulator.result()


def test_a_thinking_block_decodes_its_text():
    # None of the four cassettes carry an extended-thinking block -- no Task 17
    # scenario calls for one -- so this drives the state machine directly, the
    # same way test_an_error_event_becomes_a_model_error does, to reach
    # content_block_start/content_block_delta's "thinking" branch and
    # result()'s matching one.
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 10}}})
    accumulator.handle(
        "content_block_start",
        {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me check. "}},
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "thinking_delta", "thinking": "Yes."}},
    )
    accumulator.handle("content_block_stop", {"index": 0})
    accumulator.handle("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {}})
    accumulator.handle("message_stop", {})
    reply = accumulator.result()
    assert isinstance(reply.blocks[0], ThinkingBlock)
    assert reply.blocks[0].text == "Let me check. Yes."


def test_a_tool_call_with_no_arguments_decodes_to_an_empty_mapping():
    # A parameterless tool call can arrive with zero input_json_delta events
    # at all -- _decode's empty-raw-string branch. Every tool call in the
    # cassettes carries a path argument, so none of them reach this.
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 5}}})
    accumulator.handle(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "Noop"}},
    )
    accumulator.handle("content_block_stop", {"index": 0})
    accumulator.handle("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {}})
    accumulator.handle("message_stop", {})
    call = accumulator.result().blocks[0]
    assert isinstance(call, ToolUseBlock)
    assert call.arguments == {}


def test_malformed_tool_arguments_raise_a_usable_error():
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 5}}})
    accumulator.handle(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "Read"}},
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{not valid"}},
    )
    accumulator.handle("content_block_stop", {"index": 0})
    accumulator.handle("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {}})
    accumulator.handle("message_stop", {})
    with pytest.raises(ModelError, match="not valid JSON"):
        accumulator.result()


def test_tool_arguments_that_decode_to_a_non_mapping_raise():
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 5}}})
    accumulator.handle(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "Read"}},
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "[1, 2, 3]"}},
    )
    accumulator.handle("content_block_stop", {"index": 0})
    accumulator.handle("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {}})
    accumulator.handle("message_stop", {})
    with pytest.raises(ModelError, match="decoded to list"):
        accumulator.result()


def test_encoding_round_trips_every_block_type():
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (TextBlock("looking"), ToolUseBlock("t1", "Read", {"path": "a"}))),
            Message("user", (ToolResultBlock("t1", "1\tx", is_error=False),)),
        )
    )
    encoded = encode_messages(transcript)
    assert encoded[0]["role"] == "user"
    assert encoded[1]["content"][1]["type"] == "tool_use"
    assert encoded[2]["content"][0]["type"] == "tool_result"


def test_encoding_a_thinking_block_keeps_its_signature():
    # encode_messages's ThinkingBlock branch -- the round-trip test above
    # covers Text/ToolUse/ToolResult but not this fourth block type.
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ThinkingBlock("reasoning...", signature="sig-1"),)),
        )
    )
    encoded = encode_messages(transcript)
    assert encoded[1]["content"][0] == {
        "type": "thinking",
        "thinking": "reasoning...",
        "signature": "sig-1",
    }


def test_encode_tools_maps_each_spec_to_its_wire_shape():
    specs = (ToolSpec("Read", "Reads a file", {"type": "object", "properties": {}}),)
    assert encode_tools(specs) == [
        {
            "name": "Read",
            "description": "Reads a file",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_cache_breakpoints_are_placed_on_the_stable_prefix():
    client = AnthropicClient("k", cache=True)
    payload = client.payload(
        ModelRequest(
            "system text",
            Transcript((user_text("hi"),)),
            (ToolSpec("Read", "d", {"type": "object"}),),
            1024,
        )
    )
    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_disabling_cache_omits_cache_control():
    # The other half of "if self._cache:", untouched by the test above (which
    # only ever constructs clients with the cache=True default).
    client = AnthropicClient("k", cache=False)
    payload = client.payload(
        ModelRequest(
            "system text",
            Transcript((user_text("hi"),)),
            (ToolSpec("Read", "d", {"type": "object"}),),
            1024,
        )
    )
    assert "cache_control" not in payload["system"][-1]
    assert "cache_control" not in payload["tools"][-1]


def test_no_temperature_is_sent_unless_one_is_asked_for():
    # Current Claude models answer a non-default temperature with a 400, so an
    # ordinary request must leave the provider's own default in place.
    client = AnthropicClient("k")
    plain = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert "temperature" not in plain
    # 0.0 included: it is the one set value that is falsy.
    for value in (0.0, 0.3):
        asked = client.payload(
            ModelRequest("sys", Transcript((user_text("hi"),)), (), 64, temperature=value)
        )
        assert asked["temperature"] == value


def test_the_request_prefix_is_byte_stable_across_turns():
    """Automatic and explicit caching both need this; a timestamp would break it."""
    client = AnthropicClient("k")
    first = client.payload(ModelRequest("sys", Transcript((user_text("a"),)), (), 10))
    second = client.payload(ModelRequest("sys", Transcript((user_text("b"),)), (), 10))
    assert json.dumps(first["system"]) == json.dumps(second["system"])


def test_a_missing_api_key_names_the_adapter_and_the_fix():
    """Spec section 17.9, verbatim -- the exact wording and the em-dash both
    matter here, so this checks equality rather than a substring."""
    with pytest.raises(ModelError) as excinfo:
        AnthropicClient("")
    assert str(excinfo.value) == (
        'no API key for adapter "anthropic" — set ANTHROPIC_API_KEY or run: ncc init'
    )
    assert not excinfo.value.retryable


def test_the_client_satisfies_the_model_client_protocol():
    client = AnthropicClient("test-key")
    assert isinstance(client, ModelClient)
    assert client.model_id == DEFAULT_MODEL


async def test_aclose_closes_a_client_it_created_itself():
    client = AnthropicClient("test-key")
    await client.aclose()
    assert client._client.is_closed


async def test_aclose_does_not_close_a_client_it_was_given():
    # The other half of "if self._owns_client:" -- a caller-supplied client
    # may still be in use elsewhere, and aclose() must not pull it out from
    # under them.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = AnthropicClient("test-key", client=http)
        await client.aclose()
        assert not http.is_closed


async def test_an_http_error_is_classified_not_swallowed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = AnthropicClient("bad", client=http)
        with pytest.raises(ModelError, match="ncc init"):
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))


async def test_a_timeout_is_classified_as_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = AnthropicClient("test-key", client=http)
        with pytest.raises(ModelError, match="timed out") as excinfo:
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert excinfo.value.retryable


async def test_a_connection_failure_is_classified_as_retryable():
    # httpx.ConnectError is a TransportError but not a TimeoutException --
    # the except clause after the timeout-specific one, which the test above
    # never reaches.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("could not connect", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = AnthropicClient("test-key", client=http)
        with pytest.raises(ModelError, match="could not reach the provider") as excinfo:
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert excinfo.value.retryable


async def _sse_lines(*raw: str) -> AsyncIterator[str]:
    for line in raw:
        yield line


async def test_iter_sse_parses_events_ignoring_comments_and_unknown_fields():
    # Anthropic's own streams send periodic ": ping" comment lines to keep a
    # long-running connection alive; a generic "id:" field is valid SSE too.
    # Neither reaches StreamAccumulator through replay(), which reads already
    # -parsed (event, data) pairs straight out of the cassette files, so
    # iter_sse itself needs its own test.
    stream = _sse_lines(
        ": ping",
        "id: 1",
        "event: message_start",
        'data: {"type": "message_start"}',
        "",
        "event: message_stop",
        'data: {"type": "message_stop"}',
        "",
    )
    events = [event async for event in iter_sse(stream)]
    assert events == [
        ("message_start", {"type": "message_start"}),
        ("message_stop", {"type": "message_stop"}),
    ]


async def test_complete_streams_a_successful_reply_over_http():
    # The only test that drives AnthropicClient.complete()'s happy path all
    # the way through real SSE wire text -- every other passing test replays
    # already-decoded events straight into StreamAccumulator, bypassing both
    # iter_sse and complete() itself.
    body = (
        "event: message_start\n"
        'data: {"type": "message_start", "message": {"model": "claude-sonnet-5", '
        '"usage": {"input_tokens": 9, "output_tokens": 1, '
        '"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}\n'
        "\n"
        "event: content_block_start\n"
        'data: {"type": "content_block_start", "index": 0, '
        '"content_block": {"type": "text", "text": ""}}\n'
        "\n"
        "event: content_block_delta\n"
        'data: {"type": "content_block_delta", "index": 0, '
        '"delta": {"type": "text_delta", "text": "hi"}}\n'
        "\n"
        "event: content_block_stop\n"
        'data: {"type": "content_block_stop", "index": 0}\n'
        "\n"
        "event: message_delta\n"
        'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", '
        '"stop_sequence": null}, "usage": {"output_tokens": 3}}\n'
        "\n"
        "event: message_stop\n"
        'data: {"type": "message_stop"}\n'
        "\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = AnthropicClient("test-key", client=http)
        reply = await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.blocks[0].text == "hi"
    assert reply.stop is StopKind.END_TURN
    assert reply.usage.input_tokens == 9


def _thinking_stream(accumulator: StreamAccumulator) -> None:
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 10}}})
    accumulator.handle(
        "content_block_start",
        {"index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "thinking_delta", "thinking": "weighing it"}},
    )
    for part in ("sig-part-one-", "sig-part-two"):
        accumulator.handle(
            "content_block_delta",
            {"index": 0, "delta": {"type": "signature_delta", "signature": part}},
        )
    accumulator.handle("content_block_stop", {"index": 0})
    accumulator.handle("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {}})
    accumulator.handle("message_stop", {})


def test_a_thinking_block_keeps_its_streamed_signature():
    accumulator = StreamAccumulator()
    _thinking_stream(accumulator)
    assert accumulator.result().blocks == (
        ThinkingBlock("weighing it", "sig-part-one-sig-part-two"),
    )


def test_a_thinking_signature_survives_the_round_trip_back_to_the_api():
    accumulator = StreamAccumulator()
    _thinking_stream(accumulator)
    reply = accumulator.result()
    encoded = encode_messages(
        Transcript((user_text("q"), Message("assistant", reply.blocks), user_text("next")))
    )
    thinking = next(b for m in encoded for b in m["content"] if b.get("type") == "thinking")
    assert thinking["signature"] == "sig-part-one-sig-part-two"


def test_the_tool_list_is_byte_stable_across_turns():
    tools = (
        ToolSpec(
            "Read", "read a file", {"type": "object", "properties": {"path": {"type": "string"}}}
        ),
        ToolSpec(
            "Grep", "search", {"type": "object", "properties": {"pattern": {"type": "string"}}}
        ),
    )
    client = AnthropicClient("k")
    first = client.payload(ModelRequest("sys", Transcript((user_text("a"),)), tools, 10))
    second = client.payload(ModelRequest("sys", Transcript((user_text("b"),)), tools, 10))
    assert first["tools"]
    assert json.dumps(first["tools"]) == json.dumps(second["tools"])
