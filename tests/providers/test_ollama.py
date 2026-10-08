import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import httpx
import pytest

from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    user_text,
    validate,
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
from nanoclaude.providers.ollama import (
    DEFAULT_BASE_URL,
    OllamaClient,
    encode_messages,
    encode_tools,
    parse_arguments,
)

CASSETTES = Path(__file__).resolve().parents[1] / "cassettes"


def stream_response(chunks: list[dict[str, object]]) -> httpx.Response:
    body = "".join(json.dumps(c) + "\n" for c in chunks)
    return httpx.Response(200, text=body)


def lines_of(name: str) -> list[str]:
    """The lines of a cassette: one chunk of the reply each, as the server sends them."""
    return [line for line in (CASSETTES / name).read_text().splitlines() if line.strip()]


def arriving(
    lines: Sequence[str], *, log: list[str] | None = None, then: Exception | None = None
) -> httpx.Response:
    """A 200 response that sends ``lines`` one at a time, and then fails with ``then``.

    ``then`` is what a dropped connection looks like to the client: raised from the body,
    after the response began. ``log`` hears each line as it is sent, so that a test can
    tell what the client did before the next one went out.
    """

    async def body() -> AsyncIterator[bytes]:
        for number, line in enumerate(lines):
            if log is not None:
                log.append(f"sent {number}")
            yield (line + "\n").encode()
        if then is not None:
            raise then

    return httpx.Response(200, content=body())


async def complete_from(
    response: httpx.Response, on_text: Callable[[str], None] | None = None
) -> ModelReply:
    """What ``OllamaClient.complete`` makes of a response it was sent."""
    transport = httpx.MockTransport(lambda _request: response)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OllamaClient(model="qwen3-coder:30b", client=http)
        return await client.complete(
            ModelRequest("s", Transcript((user_text("hi"),)), (), 64), on_text=on_text
        )


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


def test_no_temperature_option_is_sent_unless_one_is_asked_for():
    # Left out, the model's own Modelfile default applies.
    client = OllamaClient(model="m")
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert payload["options"] == {"num_predict": 64}
    zero = client.payload(
        ModelRequest("sys", Transcript((user_text("hi"),)), (), 64, temperature=0.0)
    )
    assert zero["options"] == {"temperature": 0.0, "num_predict": 64}


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


async def test_replies_whose_calls_come_without_ids_get_ids_of_their_own():
    # Ollama's tool calls carry no id. The adapter mints one, and it must not be the
    # same in every reply: the transcript refuses a tool_use id it already holds, so
    # the second tool round of a conversation would have ended the session.
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "Read", "arguments": {"path": "a.py"}}},
                            {"function": {"name": "Read", "arguments": {"path": "b.py"}}},
                        ]
                    },
                    "done": True,
                    "done_reason": "stop",
                }
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OllamaClient(model="m", client=http)
        ask = ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
        first = await client.complete(ask)
        second = await client.complete(ask)
    calls = [b for reply in (first, second) for b in reply.blocks if isinstance(b, ToolUseBlock)]
    assert len(calls) == 4
    assert len({call.id for call in calls}) == 4
    assert all(re.fullmatch(r"call_[0-9a-f]{12}", call.id) for call in calls)
    # As the loop would build the conversation from them.
    answered = tuple(
        ToolResultBlock(b.id, "ok") for b in first.blocks if isinstance(b, ToolUseBlock)
    )
    validate(
        Transcript(
            (
                user_text("hi"),
                Message("assistant", tuple(first.blocks)),
                Message("user", answered),
                Message("assistant", tuple(second.blocks)),
            )
        )
    )


async def test_an_id_the_server_does_send_is_kept():
    def handler(request: httpx.Request) -> httpx.Response:
        return stream_response(
            [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call_fromserver", "function": {"name": "Read", "arguments": {}}}
                        ]
                    },
                    "done": True,
                }
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reply = await OllamaClient(model="m", client=http).complete(
            ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
        )
    call = reply.blocks[0]
    assert isinstance(call, ToolUseBlock) and call.id == "call_fromserver"


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
        with pytest.raises(EmptyReplyError, match="no content"):
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


async def test_a_401_from_a_server_behind_a_proxy_is_a_credentials_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(CredentialsError, match="check the key") as excinfo:
            await OllamaClient(model="m", client=http).complete(
                ModelRequest("s", Transcript((user_text("hi"),)), (), 64)
            )
    assert excinfo.value.status == 401 and not excinfo.value.retryable


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


# -- streaming: on_text, and a stream that breaks midway (spec 7.4) --------------


async def test_text_chunks_reach_on_text_in_order_and_join_to_the_reply_text():
    pieces: list[str] = []
    reply = await complete_from(arriving(lines_of("ollama_text.jsonl")), on_text=pieces.append)
    # The closing chunk carries an empty content, which is no text to show.
    assert pieces == ["hel", "lo"]
    assert reply.blocks == (TextBlock("hello"),)


async def test_a_reply_with_a_tool_call_streams_its_text_and_none_of_its_arguments():
    pieces: list[str] = []
    reply = await complete_from(arriving(lines_of("ollama_tool_use.jsonl")), on_text=pieces.append)
    assert pieces == ["I'll read that file."]
    call = next(b for b in reply.blocks if isinstance(b, ToolUseBlock))
    assert call.arguments == {"path": "README.md"}  # the call was still collected


async def test_complete_hands_text_over_while_the_stream_is_still_arriving():
    # Not after it: a person reads a reply that takes half a minute as it is written.
    log: list[str] = []
    reply = await complete_from(
        arriving(lines_of("ollama_text.jsonl"), log=log),
        on_text=lambda piece: log.append(f"text {piece!r}"),
    )
    assert reply.blocks == (TextBlock("hello"),)
    assert log == ["sent 0", "text 'hel'", "sent 1", "text 'lo'", "sent 2"]


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
    seen = lines_of("ollama_text.jsonl")[:1]  # "hel"
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=dropped), on_text=pieces.append)
    error = caught.value
    assert error.retryable is False
    # The counts come with the closing chunk, which had not arrived.
    assert error.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(), "qwen3-coder:30b"
    )
    assert pieces == ["hel"]
    assert str(dropped) in str(error)
    # The server was up and answering: telling the person to start it would be wrong.
    assert "ollama serve" not in str(error)


async def test_a_connection_lost_after_a_tool_call_is_not_retried_either():
    # Nothing to keep, but not asking again: an earlier call in the turn may have run.
    seen = lines_of("ollama_tool_use.jsonl")[1:2]  # the call, and no text
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    error = caught.value
    assert error.retryable is False
    assert error.partial == ModelReply((), StopKind.CUT_OFF, Usage(), "qwen3-coder:30b")


async def test_a_timeout_before_anything_arrived_is_retryable_as_before():
    with pytest.raises(ModelError, match="timed out") as caught:
        await complete_from(arriving([], then=httpx.ReadTimeout("no bytes for 600 seconds")))
    assert caught.value.retryable is True
    assert caught.value.partial is None


async def test_a_connection_lost_before_anything_arrived_still_says_how_to_start_the_server():
    # As before: for ollama a connection that fails with nothing sent is a server that is
    # not running, and is not retried.
    with pytest.raises(ModelError, match="ollama serve") as caught:
        await complete_from(arriving([], then=httpx.ReadError("connection reset")))
    assert caught.value.retryable is False
    assert caught.value.partial is None


async def test_a_stream_that_ends_cleanly_after_a_tool_call_still_hands_over_the_text_before_it():
    seen = lines_of("ollama_tool_use.jsonl")[:2]  # the text, and a call, with no closing chunk
    with pytest.raises(ModelError, match="before its tool calls were complete") as caught:
        await complete_from(arriving(seen))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("I'll read that file."),), StopKind.CUT_OFF, Usage(), "qwen3-coder:30b"
    )


# -- a stream that was interrupted midway (spec 7.4) -------------------------------

#: What ollama sends in the middle of a stream when the runner behind it dies.
RUNNER_DIED = json.dumps({"error": "llama runner process has terminated"})


async def test_an_error_line_after_text_is_not_retried_and_hands_over_what_arrived():
    # The server reports the failure on a line of its own and closes the stream cleanly,
    # so no connection error is raised: what arrived must not be returned as a reply.
    seen = [*lines_of("ollama_text.jsonl")[:1], RUNNER_DIED]  # "hel", then the error
    pieces: list[str] = []
    with pytest.raises(ModelError, match="llama runner process has terminated") as caught:
        await complete_from(arriving(seen), on_text=pieces.append)
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(), "qwen3-coder:30b"
    )
    assert pieces == ["hel"]


async def test_an_error_line_before_any_text_carries_the_providers_message():
    # It used to come back as "ollama returned no content", and the reason was lost.
    with pytest.raises(ModelError, match="llama runner process has terminated") as caught:
        await complete_from(arriving([RUNNER_DIED]))
    assert caught.value.retryable is False
    assert caught.value.partial is None


async def test_an_error_line_after_a_tool_call_is_not_retried_either():
    seen = [lines_of("ollama_tool_use.jsonl")[1], RUNNER_DIED]  # a call, and no text
    with pytest.raises(ModelError, match="llama runner process has terminated") as caught:
        await complete_from(arriving(seen))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply((), StopKind.CUT_OFF, Usage(), "qwen3-coder:30b")


async def test_a_stream_that_ends_after_text_without_its_done_line_was_cut_off():
    # A clean close, as a proxy or a restarting server makes: the end marker never came.
    seen = lines_of("ollama_text.jsonl")[:1]
    with pytest.raises(ModelError, match="complete") as caught:
        await complete_from(arriving(seen))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(), "qwen3-coder:30b"
    )


async def test_a_stream_that_ends_before_anything_arrived_is_still_no_content():
    with pytest.raises(EmptyReplyError, match="no content") as caught:
        await complete_from(arriving([]))
    assert caught.value.partial is None


# -- a stream that cannot be read is a stream that broke, not only one that dropped --


async def test_a_body_that_cannot_be_decoded_after_text_keeps_what_arrived():
    seen = lines_of("ollama_text.jsonl")[:1]
    with pytest.raises(ModelError, match="bad gzip") as caught:
        await complete_from(arriving(seen, then=httpx.DecodingError("bad gzip data")))
    assert caught.value.retryable is False
    assert "ollama serve" not in str(caught.value)
    assert caught.value.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(), "qwen3-coder:30b"
    )


async def test_a_line_that_is_not_json_after_text_keeps_what_arrived():
    seen = lines_of("ollama_text.jsonl")[:1]
    with pytest.raises(ModelError, match="could not be read") as caught:
        await complete_from(arriving([*seen, "{not json"]))
    assert caught.value.retryable is False
    assert caught.value.partial is not None
    assert caught.value.partial.blocks == (TextBlock("hel"),)


async def test_a_body_that_cannot_be_read_before_anything_arrived_raises_as_it_always_did():
    with pytest.raises(httpx.DecodingError):
        await complete_from(arriving([], then=httpx.DecodingError("bad gzip")))
    with pytest.raises(json.JSONDecodeError):
        await complete_from(arriving(["{oops"]))


async def test_the_counts_the_server_had_reported_are_in_the_partial_reply():
    lines = lines_of("ollama_text.jsonl")
    seen = [lines[0], lines[2]]  # "hel", and the closing line with the counts
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    assert caught.value.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(11, 2), "qwen3-coder:30b"
    )
