import json

import httpx
import pytest

from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    user_text,
)
from nanoclaude.providers.base import ModelClient, ModelError, ModelRequest, StopKind, ToolSpec
from nanoclaude.providers.ollama import (
    DEFAULT_BASE_URL,
    OllamaClient,
    encode_messages,
    encode_tools,
    parse_arguments,
)


def stream_response(chunks: list[dict[str, object]]) -> httpx.Response:
    body = "".join(json.dumps(c) + "\n" for c in chunks)
    return httpx.Response(200, text=body)


# -- parse_arguments ---------------------------------------------------------
# Review focus: a model's tool_calls[].function.arguments arrives as Ollama's
# native object shape from most templates, but some emit a JSON string
# instead -- both must decode to the same mapping, and neither a malformed
# string nor an unrelated type may crash the adapter.


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"path": "a.py"}', {"path": "a.py"}),  # the documented shape
        ({"path": "a.py"}, {"path": "a.py"}),  # ollama's native object shape
        ("{}", {}),
        ("", {}),
        (None, {}),
    ],
)
def test_arguments_arrive_as_a_string_or_an_object_and_both_work(raw, expected):
    assert parse_arguments(raw) == expected


def test_arguments_that_are_not_valid_json_are_a_clear_error():
    with pytest.raises(ModelError, match="not valid JSON"):
        parse_arguments('{"path": "a.py"')


def test_arguments_that_are_neither_a_string_nor_a_mapping_are_a_clear_error():
    with pytest.raises(ModelError, match="tool arguments arrived as list"):
        parse_arguments([1, 2, 3])


def test_arguments_that_decode_to_a_non_mapping_are_a_clear_error():
    with pytest.raises(ModelError, match="decoded to list"):
        parse_arguments("[1, 2, 3]")


# -- encode_messages / encode_tools ------------------------------------------


def test_the_system_prompt_is_the_first_message():
    encoded = encode_messages("sys", Transcript((user_text("hi"),)))
    assert encoded[0] == {"role": "system", "content": "sys"}
    assert encoded[1] == {"role": "user", "content": "hi"}


def test_a_tool_result_becomes_a_tool_role_message():
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolUseBlock("t1", "Read", {"path": "a"}),)),
            Message("user", (ToolResultBlock("t1", "1\tx"),)),
        )
    )
    encoded = encode_messages("sys", transcript)
    assert encoded[2]["tool_calls"][0]["function"]["name"] == "Read"
    assert encoded[2]["tool_calls"][0]["function"]["arguments"] == {"path": "a"}
    assert encoded[3] == {"role": "tool", "content": "1\tx"}


def test_an_assistant_turn_with_text_and_a_call_keeps_both():
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (TextBlock("looking"), ToolUseBlock("t1", "Read", {}))),
        )
    )
    encoded = encode_messages("sys", transcript)
    assert encoded[2]["content"] == "looking"
    assert len(encoded[2]["tool_calls"]) == 1


def test_an_assistant_turn_with_no_tool_calls_omits_the_key():
    # The other half of "if calls:" -- a plain-text reply must not grow an
    # empty tool_calls list that the next request would send back as "[]".
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (TextBlock("just text"),)),
        )
    )
    encoded = encode_messages("sys", transcript)
    assert "tool_calls" not in encoded[2]
    assert encoded[2]["content"] == "just text"


def test_encode_tools_maps_each_spec_to_its_wire_shape():
    specs = (ToolSpec("Read", "Reads a file", {"type": "object", "properties": {}}),)
    assert encode_tools(specs) == [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Reads a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# -- OllamaClient construction and payload ------------------------------------


def test_the_default_base_url_is_localhost_ollama():
    client = OllamaClient(model="m")
    assert client._base_url == DEFAULT_BASE_URL


def test_the_client_satisfies_the_model_client_protocol():
    client = OllamaClient(model="m")
    assert isinstance(client, ModelClient)
    assert client.model_id == "m"


async def test_aclose_closes_a_client_it_created_itself():
    client = OllamaClient(model="m")
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
        client = OllamaClient(model="m", client=http)
        await client.aclose()
        assert not http.is_closed


def test_the_system_prompt_is_the_first_message_in_the_payload():
    client = OllamaClient(model="m")
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert payload["messages"][0] == {"role": "system", "content": "sys"}


def test_tools_are_included_in_the_payload_when_present():
    client = OllamaClient(model="m")
    tools = (ToolSpec("Read", "Reads a file", {"type": "object", "properties": {}}),)
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), tools, 64))
    assert payload["tools"] == encode_tools(tools)


def test_no_tools_key_when_there_are_none():
    # The other half of "if request.tools:" above.
    client = OllamaClient(model="m")
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert "tools" not in payload


def test_payload_carries_temperature_and_max_tokens_in_options():
    client = OllamaClient(model="m")
    payload = client.payload(
        ModelRequest("sys", Transcript((user_text("hi"),)), (), 64, temperature=0.3)
    )
    assert payload["options"] == {"temperature": 0.3, "num_predict": 64}


# -- complete(): happy paths, from the brief -----------------------------


async def test_a_text_reply_is_decoded():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            [
                {"message": {"content": "hel"}, "done": False},
                {
                    "message": {"content": "lo"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 12,
                    "eval_count": 3,
                },
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(model="qwen3-coder", client=http)
        reply = await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 64))
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.blocks[0].text == "hello"
    assert reply.stop is StopKind.END_TURN
    assert reply.usage.input_tokens == 12
    assert reply.usage.output_tokens == 3


async def test_a_native_tool_call_is_decoded():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "Read", "arguments": {"path": "a.py"}}}
                        ]
                    },
                    "done": True,
                    "done_reason": "stop",
                },
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reply = await OllamaClient(model="m", client=http).complete(
            ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
        )
    assert reply.stop is StopKind.TOOL_USE
    assert isinstance(reply.blocks[0], ToolUseBlock)
    assert reply.blocks[0].arguments == {"path": "a.py"}


async def test_a_tool_calls_arguments_as_a_json_string_are_decoded():
    # Review focus: not every template sends ollama's native object shape.
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "Read", "arguments": '{"path": "a.py"}'}}
                        ]
                    },
                    "done": True,
                },
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reply = await OllamaClient(model="m", client=http).complete(
            ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
        )
    call = reply.blocks[0]
    assert isinstance(call, ToolUseBlock)
    assert call.arguments == {"path": "a.py"}


async def test_a_missing_server_says_how_to_start_it():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        # Only ModelError may cross the adapter boundary -- asserted here by
        # catching that exact type rather than the brief's own looser
        # `Exception`, which also lets `.retryable` be checked below.
        with pytest.raises(ModelError, match="ollama serve") as excinfo:
            await OllamaClient(model="m", client=http).complete(
                ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
            )
    assert not excinfo.value.retryable


# -- complete(): the lessons from Tasks 17/18, applied here too -------------


async def test_an_unfinished_stream_with_a_pending_tool_call_is_a_non_retryable_error():
    # Review focus #1: the stream ends (cleanly, from the client's point of
    # view) right after announcing a call but before the "done": true line
    # that would have confirmed it was the whole story. Handing it to the
    # executor anyway risks running a tool on arguments the model never
    # finished sending.
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "Read", "arguments": {"path": "a.py"}}}
                        ]
                    }
                },
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelError, match="before its tool calls were complete") as excinfo:
            await OllamaClient(model="m", client=http).complete(
                ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
            )
    assert excinfo.value.retryable is False


async def test_a_text_only_reply_without_a_final_done_is_still_accepted():
    # The other half of the check above: a dropped connection after plain
    # prose is just a shorter reply, not a usable-error condition.
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response([{"message": {"content": "hi"}}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reply = await OllamaClient(model="m", client=http).complete(
            ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
        )
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.blocks[0].text == "hi"
    assert reply.stop is StopKind.END_TURN


async def test_a_non_object_stream_chunk_is_a_model_error_not_a_crash():
    # Review focus #3: valid JSON need not be an object. A bare string or list
    # on the wire must become a ModelError, not an AttributeError from
    # chunk.get(...) escaping the adapter.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='"oops"\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelError, match="unexpected stream chunk"):
            await OllamaClient(model="m", client=http).complete(
                ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
            )


async def test_a_blank_line_in_the_stream_is_skipped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="\n" + json.dumps({"message": {"content": "hi"}, "done": True}) + "\n"
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reply = await OllamaClient(model="m", client=http).complete(
            ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
        )
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.blocks[0].text == "hi"


async def test_a_stream_with_no_content_at_all_is_a_usable_error():
    # Nothing was cut off here -- a complete stream that never carries text or
    # a tool call. The other refusal branch, distinct from the truncation one
    # above.
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response([{"done": True, "done_reason": "stop"}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelError, match="no content"):
            await OllamaClient(model="m", client=http).complete(
                ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
            )


# -- complete(): HTTP and transport failures ---------------------------------


async def test_a_404_is_classified_as_not_knowing_the_model():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "m" not found, try pulling it first'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelError, match="does not know that model or endpoint") as excinfo:
            await OllamaClient(model="m", client=http).complete(
                ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
            )
    assert not excinfo.value.retryable


async def test_a_429_is_classified_as_retryable():
    # Pins that the HTTP-error branch defers to classify_status's real table
    # (spec 7.4): a naive "retryable = status >= 500" would mark a 429 as
    # non-retryable, contradicting the spec.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelError) as excinfo:
            await OllamaClient(model="m", client=http).complete(
                ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
            )
    assert excinfo.value.retryable


async def test_a_timeout_is_classified_as_retryable():
    # The except clause before the connection-refused one, which the test
    # above never reaches: a slow local model is not "ollama is not running".
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OllamaClient(model="m", client=http)
        with pytest.raises(ModelError, match="timed out") as excinfo:
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 64))
    assert excinfo.value.retryable


async def test_a_trailing_slash_on_base_url_does_not_double_up():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return stream_response([{"message": {"content": "hi"}, "done": True}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(model="m", base_url="http://x/", client=http)
        await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 64))
    assert str(seen[0].url) == "http://x/api/chat"
