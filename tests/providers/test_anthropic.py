import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
from nanoclaude.providers.base import (
    CredentialsError,
    EmptyReplyError,
    ModelClient,
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    ToolSpec,
    Usage,
)

CASSETTES = Path(__file__).resolve().parents[1] / "cassettes"


def events_of(name: str) -> list[dict[str, Any]]:
    """The events a cassette holds, each ``{"event": ..., "data": ...}``, in order."""
    lines = (CASSETTES / name).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def replay(name: str, on_text: Callable[[str], None] | None = None) -> StreamAccumulator:
    accumulator = StreamAccumulator(on_text=on_text)
    for event in events_of(name):
        accumulator.handle(event["event"], event["data"])
    return accumulator


def wire(events: Sequence[Mapping[str, Any]]) -> bytes:
    """The Server-Sent Events text a server sends for ``events``."""
    return "".join(
        f"event: {e['event']}\ndata: {json.dumps(e['data'])}\n\n" for e in events
    ).encode()


def arriving(
    events: Sequence[Mapping[str, Any]],
    *,
    log: list[str] | None = None,
    then: Exception | None = None,
) -> httpx.Response:
    """A 200 response that sends ``events`` one at a time, and then fails with ``then``.

    ``then`` is what a dropped connection looks like to the client: raised from the
    body, after the response began. ``log`` hears each event as it is sent, so that a
    test can tell what the client did before the next one went out.
    """

    async def body() -> AsyncIterator[bytes]:
        for event in events:
            if log is not None:
                log.append(f"sent {event['event']}")
            yield wire([event])
        if then is not None:
            raise then

    return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})


def ask() -> ModelRequest:
    return ModelRequest("s", Transcript((user_text("hi"),)), (), 10)


async def complete_from(
    response: httpx.Response, on_text: Callable[[str], None] | None = None
) -> ModelReply:
    """What ``AnthropicClient.complete`` makes of a response it was sent."""
    transport = httpx.MockTransport(lambda _request: response)
    async with httpx.AsyncClient(transport=transport) as http:
        return await AnthropicClient("test-key", client=http).complete(ask(), on_text=on_text)


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
    with pytest.raises(EmptyReplyError, match="no content blocks"):
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
    with pytest.raises(CredentialsError) as excinfo:
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
        with pytest.raises(CredentialsError, match="ncc init") as excinfo:
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert excinfo.value.status == 401 and not excinfo.value.retryable


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


# -- streaming: on_text, and a stream that breaks midway (spec 7.4) --------------


def test_text_deltas_reach_on_text_in_order_and_join_to_the_reply_text():
    pieces: list[str] = []
    reply = replay("anthropic_text.jsonl", on_text=pieces.append).result()
    assert pieces == ["Hello! ", "The answer is 42."]
    assert reply.blocks == (TextBlock("Hello! The answer is 42."),)


def test_a_reply_with_a_tool_call_streams_its_text_and_none_of_its_arguments():
    pieces: list[str] = []
    reply = replay("anthropic_tool_use.jsonl", on_text=pieces.append).result()
    assert pieces == ["I'll read that file."]
    call = next(b for b in reply.blocks if isinstance(b, ToolUseBlock))
    assert call.arguments == {"path": "src/app.py"}  # the arguments were still collected


def test_thinking_text_never_reaches_on_text():
    pieces: list[str] = []
    accumulator = StreamAccumulator(on_text=pieces.append)
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 10}}})
    accumulator.handle(
        "content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "thinking_delta", "thinking": "weighing it"}},
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
    )
    accumulator.handle("content_block_stop", {"index": 0})
    accumulator.handle(
        "content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}}
    )
    accumulator.handle(
        "content_block_delta", {"index": 1, "delta": {"type": "text_delta", "text": "Done."}}
    )
    accumulator.handle("content_block_stop", {"index": 1})
    accumulator.handle("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {}})
    accumulator.handle("message_stop", {})
    assert pieces == ["Done."]
    assert accumulator.result().blocks[0] == ThinkingBlock("weighing it", "sig")


async def test_complete_hands_text_over_while_the_stream_is_still_arriving():
    # Not after it: a person reads a reply that takes half a minute as it is written.
    log: list[str] = []
    reply = await complete_from(
        arriving(events_of("anthropic_text.jsonl"), log=log),
        on_text=lambda piece: log.append(f"text {piece!r}"),
    )
    assert reply.blocks == (TextBlock("Hello! The answer is 42."),)
    assert log == [
        "sent message_start",
        "sent content_block_start",
        "sent content_block_delta",
        "text 'Hello! '",  # before the next delta was even sent
        "sent content_block_delta",
        "text 'The answer is 42.'",
        "sent content_block_stop",
        "sent message_delta",
        "sent message_stop",
    ]


@pytest.mark.parametrize(
    "dropped",
    [
        httpx.ReadError("connection reset by peer"),
        httpx.RemoteProtocolError("peer closed connection without a complete message body"),
        httpx.ReadTimeout("no bytes for 600 seconds"),
    ],
    ids=["read error", "protocol error", "timeout"],
)
async def test_a_connection_lost_after_text_is_not_retried_and_hands_over_what_arrived(dropped):
    pieces: list[str] = []
    seen = events_of("anthropic_text.jsonl")[:3]  # message_start, the block, "Hello! "
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=dropped), on_text=pieces.append)
    error = caught.value
    assert error.retryable is False
    assert error.partial == ModelReply(
        (TextBlock("Hello! "),), StopKind.CUT_OFF, Usage(15, 0), "claude-sonnet-5"
    )
    assert pieces == ["Hello! "]
    assert str(dropped) in str(error)


async def test_a_connection_lost_inside_a_tool_call_is_not_retried_either():
    # Nothing to keep but what the provider billed: the half-received call is dropped, and
    # asking again could run the call twice if an earlier one in the turn already ran.
    seen = events_of("anthropic_parallel_tools.jsonl")[:3]  # message_start, a call, its start
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    error = caught.value
    assert error.retryable is False
    assert error.partial == ModelReply((), StopKind.CUT_OFF, Usage(150, 0), "claude-sonnet-5")


@pytest.mark.parametrize(
    "dropped", [httpx.ReadError("connection reset"), httpx.ReadTimeout("timed out")]
)
async def test_a_connection_lost_before_anything_arrived_is_retryable_as_before(dropped):
    seen = events_of("anthropic_text.jsonl")[:1]  # message_start, and then nothing
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=dropped))
    assert caught.value.retryable is True
    assert caught.value.partial is None


async def test_a_thinking_only_stream_that_breaks_is_still_retryable():
    # Thinking is not shown and cannot have run anything, so asking again costs tokens
    # and nothing else.
    seen = [
        {"event": "message_start", "data": {"message": {"usage": {"input_tokens": 9}}}},
        {
            "event": "content_block_start",
            "data": {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        },
        {
            "event": "content_block_delta",
            "data": {"index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        },
    ]
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    assert caught.value.retryable is True


async def test_an_overloaded_error_after_text_is_not_retried_and_hands_over_what_arrived():
    # An error the provider sends inside the stream, not a dropped connection: the same
    # rule, because a retry would write the first half again.
    fault = {
        "event": "error",
        "data": {"error": {"type": "overloaded_error", "message": "Overloaded"}},
    }
    with pytest.raises(ModelError, match="Overloaded") as caught:
        await complete_from(arriving([*events_of("anthropic_text.jsonl")[:3], fault]))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("Hello! "),), StopKind.CUT_OFF, Usage(15, 0), "claude-sonnet-5"
    )


async def test_an_overloaded_error_before_anything_arrived_is_retryable_as_before():
    fault = {
        "event": "error",
        "data": {"error": {"type": "overloaded_error", "message": "Overloaded"}},
    }
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving([*events_of("anthropic_text.jsonl")[:1], fault]))
    assert caught.value.retryable is True
    assert caught.value.partial is None


def test_a_stream_that_ends_cleanly_mid_tool_call_still_hands_over_the_text_before_it():
    with pytest.raises(ModelError, match="incomplete") as caught:
        replay("anthropic_truncated.jsonl").result()
    assert caught.value.partial == ModelReply(
        (TextBlock("I'll read that file."),), StopKind.CUT_OFF, Usage(120, 0), "claude-sonnet-5"
    )


async def test_only_the_visible_text_is_kept_when_the_stream_breaks_after_thinking_and_text():
    seen = [
        {"event": "message_start", "data": {"message": {"usage": {"input_tokens": 9}}}},
        {
            "event": "content_block_start",
            "data": {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        },
        {
            "event": "content_block_delta",
            "data": {"index": 0, "delta": {"type": "thinking_delta", "thinking": "weighing it"}},
        },
        {"event": "content_block_stop", "data": {"index": 0}},
        {
            "event": "content_block_start",
            "data": {"index": 1, "content_block": {"type": "text", "text": ""}},
        },
        {
            "event": "content_block_delta",
            "data": {"index": 1, "delta": {"type": "text_delta", "text": "The answer"}},
        },
    ]
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    assert caught.value.partial == ModelReply(
        (TextBlock("The answer"),), StopKind.CUT_OFF, Usage(9, 0), "claude-sonnet-5"
    )


def test_a_partial_reply_keeps_each_text_block_in_order_and_leaves_the_calls_out():
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 3}}})
    for index, block, delta in (
        (0, {"type": "text", "text": ""}, {"type": "text_delta", "text": "First. "}),
        (
            1,
            {"type": "tool_use", "id": "t1", "name": "Read"},
            {"type": "input_json_delta", "partial_json": '{"pa'},
        ),
        (2, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Third."}),
    ):
        accumulator.handle("content_block_start", {"index": index, "content_block": block})
        accumulator.handle("content_block_delta", {"index": index, "delta": delta})
    assert accumulator.partial() == ModelReply(
        (TextBlock("First. "), TextBlock("Third.")),
        StopKind.CUT_OFF,
        Usage(3, 0),
        "claude-sonnet-5",
    )


# -- a stream that closes without its end marker was interrupted midway (spec 7.4) --


async def test_a_stream_that_closes_after_a_finished_text_block_without_message_stop_was_cut_off():
    # message_start, a text block and its deltas and its stop, and then the connection
    # closes cleanly: nothing raised, and the reply is not whole.
    seen = events_of("anthropic_text.jsonl")[:5]
    pieces: list[str] = []
    with pytest.raises(ModelError, match="incomplete") as caught:
        await complete_from(arriving(seen), on_text=pieces.append)
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("Hello! The answer is 42."),), StopKind.CUT_OFF, Usage(15, 0), "claude-sonnet-5"
    )
    assert "".join(pieces) == "Hello! The answer is 42."


async def test_a_stream_that_closes_after_a_finished_tool_call_without_message_stop_drops_it():
    # The call is closed and its arguments parse, so it looks runnable. It must not run: it
    # was not confirmed to be the whole of the reply, and an earlier call may already have.
    seen = events_of("anthropic_tool_use.jsonl")[:8]  # the text, and the call, stopped
    with pytest.raises(ModelError, match="incomplete") as caught:
        await complete_from(arriving(seen))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("I'll read that file."),), StopKind.CUT_OFF, Usage(120, 0), "claude-sonnet-5"
    )


async def test_a_stream_that_closes_before_anything_arrived_is_not_a_reply_stream():
    # message_start and nothing after it: the stream never said it was over, so it is not an
    # empty reply (which is what message_stop with no block in between is).
    with pytest.raises(ModelError, match="not a reply stream") as caught:
        await complete_from(arriving(events_of("anthropic_text.jsonl")[:1]))
    assert not isinstance(caught.value, EmptyReplyError)
    assert caught.value.partial is None and caught.value.retryable is False


def test_a_stream_that_closes_inside_a_thinking_block_is_an_error_with_nothing_to_keep():
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 4}}})
    accumulator.handle(
        "content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}
    )
    with pytest.raises(ModelError, match="incomplete") as caught:
        accumulator.result()
    assert caught.value.retryable is False
    assert caught.value.partial is None  # nothing a person could have read


def test_thinking_alone_with_no_message_stop_is_not_a_reply():
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 4}}})
    accumulator.handle(
        "content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}
    )
    accumulator.handle(
        "content_block_delta",
        {"index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
    )
    accumulator.handle("content_block_stop", {"index": 0})
    with pytest.raises(ModelError, match="incomplete"):
        accumulator.result()


# -- a stream that cannot be read is a stream that broke, not only one that dropped --


def raw_response(*parts: bytes) -> httpx.Response:
    return httpx.Response(
        200, content=b"".join(parts), headers={"content-type": "text/event-stream"}
    )


async def test_a_body_that_cannot_be_decoded_after_text_keeps_what_arrived():
    seen = events_of("anthropic_text.jsonl")[:3]  # message_start, the block, "Hello! "
    with pytest.raises(ModelError, match="bad gzip") as caught:
        await complete_from(arriving(seen, then=httpx.DecodingError("bad gzip data")))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("Hello! "),), StopKind.CUT_OFF, Usage(15, 0), "claude-sonnet-5"
    )


async def test_an_event_that_is_not_json_after_text_keeps_what_arrived():
    seen = wire(events_of("anthropic_text.jsonl")[:3])
    broken = b"event: content_block_delta\ndata: {not json\n\n"
    with pytest.raises(ModelError, match="could not be read") as caught:
        await complete_from(raw_response(seen, broken))
    assert caught.value.retryable is False
    assert caught.value.partial is not None
    assert caught.value.partial.blocks == (TextBlock("Hello! "),)


async def test_a_value_the_stream_cannot_be_read_with_after_text_keeps_what_arrived():
    seen = wire(events_of("anthropic_text.jsonl")[:3])
    odd = wire([{"event": "content_block_delta", "data": {"index": "two", "delta": {}}}])
    with pytest.raises(ModelError, match="could not be read") as caught:
        await complete_from(raw_response(seen, odd))
    assert caught.value.partial is not None
    assert caught.value.partial.blocks == (TextBlock("Hello! "),)


async def test_a_body_that_cannot_be_read_before_anything_arrived_raises_as_it_always_did():
    with pytest.raises(httpx.DecodingError):
        await complete_from(
            arriving(events_of("anthropic_text.jsonl")[:1], then=httpx.DecodingError("bad gzip"))
        )
    with pytest.raises(json.JSONDecodeError):
        await complete_from(
            raw_response(wire(events_of("anthropic_text.jsonl")[:1]), b"event: x\ndata: {oops\n\n")
        )


async def test_the_model_the_server_reported_is_the_model_of_the_partial_reply():
    # What answered is what the server says it was, which need not be the alias asked for.
    start = {
        "event": "message_start",
        "data": {"message": {"model": "claude-sonnet-5-20261001", "usage": {"input_tokens": 9}}},
    }
    seen = [start, *events_of("anthropic_text.jsonl")[1:3]]
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    assert caught.value.partial is not None
    assert caught.value.partial.model == "claude-sonnet-5-20261001"


# -- an answer that is not a reply stream is not a reply ----------------------------------
# A 200 proves only that something answered. "Empty reply" is for a stream that ran from
# message_start to message_stop with nothing in it; anything else is not a reply stream.

NOT_A_STREAM = {
    "a page": lambda: httpx.Response(
        200, content=b"<html>Sign in to the network</html>", headers={"content-type": "text/html"}
    ),
    "an empty body": lambda: httpx.Response(200, content=b""),
    "a whole reply that was not streamed": lambda: httpx.Response(
        200, json={"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}
    ),
}


@pytest.mark.parametrize("answer", NOT_A_STREAM)
async def test_a_200_that_is_not_a_reply_stream_is_an_error_and_not_an_empty_reply(answer):
    with pytest.raises(ModelError, match="not a reply stream") as caught:
        await complete_from(NOT_A_STREAM[answer]())
    assert not isinstance(caught.value, EmptyReplyError)
    assert caught.value.retryable is False and caught.value.partial is None


def test_a_message_stop_with_no_message_start_is_not_a_reply_stream():
    accumulator = StreamAccumulator()
    accumulator.handle("message_stop", {})
    with pytest.raises(ModelError, match="not a reply stream") as caught:
        accumulator.result()
    assert not isinstance(caught.value, EmptyReplyError)


def test_a_message_start_with_no_message_stop_is_not_an_empty_reply():
    accumulator = StreamAccumulator()
    accumulator.handle("message_start", {"message": {"usage": {"input_tokens": 5}}})
    with pytest.raises(ModelError, match="not a reply stream") as caught:
        accumulator.result()
    assert not isinstance(caught.value, EmptyReplyError)


@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_a_redirect_is_an_error_and_its_page_is_not_read_as_a_reply(status):
    response = httpx.Response(
        status, headers={"location": "https://x/login"}, content=b"<html>moved</html>"
    )
    with pytest.raises(ModelError, match=rf"redirect \(HTTP {status}\)") as caught:
        await complete_from(response)
    assert caught.value.status == status and caught.value.retryable is False
    assert not isinstance(caught.value, EmptyReplyError)
