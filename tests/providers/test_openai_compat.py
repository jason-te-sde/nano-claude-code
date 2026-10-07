import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
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
    validate,
)
from nanoclaude.providers.base import (
    ModelClient,
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    ToolSpec,
    Usage,
)
from nanoclaude.providers.openai_compat import (
    ChunkAccumulator,
    OpenAICompatClient,
    encode_messages,
    encode_tools,
    parse_arguments,
)

CASSETTES = Path(__file__).resolve().parents[1] / "cassettes"


def payloads_of(name: str) -> list[Any]:
    """What a cassette holds, one value per line: a chunk object, or the string ``"[DONE]"``."""
    lines = (CASSETTES / name).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def replay(
    name: str, *, model: str = "gpt-5", on_text: Callable[[str], None] | None = None
) -> ChunkAccumulator:
    """Decode one cassette straight into a ChunkAccumulator, line by line.

    Each line is either a chunk object (fed to ``.handle()``) or the JSON
    string ``"[DONE]"``, which carries no information for the accumulator and
    is skipped -- the same sentinel ``OpenAICompatClient.complete()`` strips
    before it ever reaches ``.handle()`` on the wire.
    """
    accumulator = ChunkAccumulator(model=model, on_text=on_text)
    for payload in payloads_of(name):
        if payload == "[DONE]":
            continue
        accumulator.handle(payload)
    return accumulator


def wire(payloads: Sequence[Any]) -> bytes:
    """The literal SSE wire text a server sends for ``payloads``.

    Every payload becomes ``data: ...\\n\\n``, the string ``"[DONE]"`` as the bare
    (unquoted) token real servers send.
    """
    return "".join(
        f"data: {'[DONE]' if payload == '[DONE]' else json.dumps(payload)}\n\n"
        for payload in payloads
    ).encode()


def wire_body(name: str) -> bytes:
    """Reconstruct the literal SSE wire text a server would have sent.

    This is what lets a cassette be checked against the *real* HTTP-streaming
    parser, not just against ChunkAccumulator directly: every stored line is
    turned back into ``data: ...\\n\\n``, including the bare (unquoted)
    ``[DONE]`` token real servers send, and fed through
    ``OpenAICompatClient.complete()`` over ``httpx.MockTransport``.
    """
    return wire(payloads_of(name))


def arriving(
    payloads: Sequence[Any],
    *,
    log: list[str] | None = None,
    then: Exception | None = None,
) -> httpx.Response:
    """A 200 response that sends ``payloads`` one at a time, and then fails with ``then``.

    ``then`` is what a dropped connection looks like to the client: raised from the body,
    after the response began. ``log`` hears each chunk as it is sent, so that a test can
    tell what the client did before the next one went out.
    """

    async def body() -> AsyncIterator[bytes]:
        for number, payload in enumerate(payloads):
            if log is not None:
                log.append(f"sent {number}")
            yield wire([payload])
        if then is not None:
            raise then

    return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})


async def complete_from(
    response: httpx.Response, on_text: Callable[[str], None] | None = None
) -> ModelReply:
    """What ``OpenAICompatClient.complete`` makes of a response it was sent."""
    transport = httpx.MockTransport(lambda _request: response)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1", client=http)
        return await client.complete(
            ModelRequest("s", Transcript((user_text("hi"),)), (), 10), on_text=on_text
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"path": "a.py"}', {"path": "a.py"}),  # the documented shape
        ({"path": "a.py"}, {"path": "a.py"}),  # already an object
        ("{}", {}),
        ("", {}),
        (None, {}),
    ],
)
def test_arguments_arrive_as_a_string_or_an_object_and_both_work(raw, expected):
    """Review Focus #2. The spec says string; several endpoints send an object."""
    assert parse_arguments(raw) == expected


def test_arguments_that_are_not_valid_json_are_a_clear_error():
    with pytest.raises(ModelError, match="not valid JSON"):
        parse_arguments('{"path": "a.py"')


def test_arguments_that_are_neither_a_string_nor_a_mapping_are_a_clear_error():
    # The branch between the two above: not None/"", not a dict, not a str
    # either -- an int or a list must not reach json.loads with the wrong type.
    with pytest.raises(ModelError, match="tool arguments arrived as list"):
        parse_arguments([1, 2, 3])


def test_arguments_that_decode_to_a_non_mapping_are_a_clear_error():
    # The far side of the JSON-decode guard: valid JSON that is not an object.
    with pytest.raises(ModelError, match="decoded to list"):
        parse_arguments("[1, 2, 3]")


def test_a_tool_result_becomes_a_tool_role_message_not_a_content_block():
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ToolUseBlock("t1", "Read", {"path": "a"}),)),
            Message("user", (ToolResultBlock("t1", "1\tx"),)),
        )
    )
    encoded = encode_messages(transcript)
    assert encoded[1]["tool_calls"][0]["function"]["name"] == "Read"
    # The other half of "content: text or None" -- a call with no text at all
    # must not become an empty-string content field.
    assert encoded[1]["content"] is None
    assert encoded[2]["role"] == "tool"
    assert encoded[2]["tool_call_id"] == "t1"


def test_an_assistant_turn_with_text_and_a_call_keeps_both():
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (TextBlock("looking"), ToolUseBlock("t1", "Read", {}))),
        )
    )
    encoded = encode_messages(transcript)
    assert encoded[1]["content"] == "looking"
    assert len(encoded[1]["tool_calls"]) == 1


def test_an_assistant_turn_with_no_tool_calls_omits_the_key():
    # The other half of "if calls:" -- a plain-text reply must not grow an
    # empty tool_calls list that the next request would send back as "[]".
    transcript = Transcript(
        (
            user_text("hi"),
            Message("assistant", (TextBlock("just text"),)),
        )
    )
    encoded = encode_messages(transcript)
    assert "tool_calls" not in encoded[1]
    assert encoded[1]["content"] == "just text"


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


def test_streamed_tool_call_fragments_are_reassembled_by_index():
    accumulator = ChunkAccumulator(model="gpt-5")
    accumulator.handle(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "Read", "arguments": '{"pa'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    accumulator.handle(
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}
            ]
        }
    )
    accumulator.handle({"choices": [{"finish_reason": "tool_calls", "delta": {}}]})
    reply = accumulator.result()
    call = reply.blocks[0]
    assert isinstance(call, ToolUseBlock) and call.arguments == {"path": "a.py"}
    assert reply.stop is StopKind.TOOL_USE


def test_tool_call_arguments_split_across_many_chunks_reassemble_correctly():
    # "Many", not two: five fragments after the declaring one, each far too
    # small to parse on its own -- one key or string fragment at a time.
    accumulator = ChunkAccumulator(model="gpt-5")
    accumulator.handle(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "function": {"name": "Write", "arguments": ""}}
                        ]
                    }
                }
            ]
        }
    )
    for fragment in ('{"pa', 'th": ', '"a/b/', "c.py", '", "content": "x"}'):
        accumulator.handle(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}}
                ]
            }
        )
    accumulator.handle({"choices": [{"finish_reason": "tool_calls", "delta": {}}]})
    call = accumulator.result().blocks[0]
    assert isinstance(call, ToolUseBlock)
    assert call.name == "Write"
    assert call.arguments == {"path": "a/b/c.py", "content": "x"}


def test_parallel_tool_calls_keep_their_order_by_index():
    # Fragments arrive for index 1 before index 0 -- order in the reply must
    # still follow the index, not arrival order.
    accumulator = ChunkAccumulator(model="gpt-5")
    accumulator.handle(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "c2",
                                "function": {"name": "Read", "arguments": '{"path": "b"}'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    accumulator.handle(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "Read", "arguments": '{"path": "a"}'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    accumulator.handle({"choices": [{"finish_reason": "tool_calls", "delta": {}}]})
    reply = accumulator.result()
    calls = [b for b in reply.blocks if isinstance(b, ToolUseBlock)]
    assert len(calls) == 2
    assert calls[0].id == "c1" and calls[0].arguments == {"path": "a"}
    assert calls[1].id == "c2" and calls[1].arguments == {"path": "b"}


def test_a_stream_truncated_mid_tool_call_raises_a_non_retryable_error():
    # The connection drops while function.arguments is still open: no closing
    # fragment, no finish_reason. Silently handing {} to the tool would be
    # worse than refusing, and retrying is also wrong -- an earlier call in
    # the same turn may already have run.
    accumulator = ChunkAccumulator(model="gpt-5")
    accumulator.handle(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "Read", "arguments": '{"path": "a'},
                            }
                        ]
                    }
                }
            ]
        }
    )
    with pytest.raises(ModelError) as excinfo:
        accumulator.result()
    assert not excinfo.value.retryable


def test_a_tool_call_that_never_receives_a_name_is_a_usable_error():
    # The call is opened (an index exists) and the stream reports it finished,
    # yet function.name never arrived -- result()'s own guard for a server that
    # sends a malformed call, distinct from the unfinished-stream check.
    accumulator = ChunkAccumulator(model="gpt-5")
    accumulator.handle({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1"}]}}]})
    accumulator.handle({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    with pytest.raises(ModelError, match="no function name") as excinfo:
        accumulator.result()
    assert not excinfo.value.retryable


def test_a_tool_call_with_no_id_falls_back_to_a_generated_one():
    # The other half of "slot['id'] or a minted id" -- every other tool-call test
    # supplies an id in the opening fragment; this one never does, so the call
    # still needs an id the executor can refer back to.
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "Read"}}]}}]}
    )
    accumulator.handle({"choices": [{"finish_reason": "tool_calls", "delta": {}}]})
    call = accumulator.result().blocks[0]
    assert isinstance(call, ToolUseBlock)
    assert re.fullmatch(r"call_[0-9a-f]{12}", call.id)


async def test_replies_whose_calls_come_without_ids_get_ids_of_their_own():
    # A server that sends no id (some compatible ones do not) must not get "call_0"
    # in every reply: the transcript refuses a tool_use id it already holds, so the
    # second tool round of a conversation would have ended the session.
    stream = (
        b'data: {"choices": [{"delta": {"tool_calls": ['
        b'{"index": 0, "function": {"name": "Read", "arguments": "{\\"path\\": \\"a.py\\"}"}},'
        b'{"index": 1, "function": {"name": "Read", "arguments": "{\\"path\\": \\"b.py\\"}"}}'
        b"]}}]}\n\n"
        b'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatClient("test-key", model="m", base_url="https://x/v1", client=http)
        ask = ModelRequest("s", Transcript((user_text("hi"),)), (), 10)
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


def test_an_error_chunk_raises_a_model_error():
    accumulator = ChunkAccumulator(model="gpt-5")
    with pytest.raises(ModelError, match="rate limited"):
        accumulator.handle({"error": {"message": "rate limited"}})


def test_a_stream_with_no_content_at_all_is_a_usable_error():
    # Nothing was cut off here -- a complete stream that never carries text,
    # reasoning or a tool call. result()'s other refusal branch, distinct from
    # the truncation ones above.
    accumulator = ChunkAccumulator(model="gpt-5")
    accumulator.handle({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    with pytest.raises(ModelError, match="no content"):
        accumulator.result()


def test_reasoning_content_is_preserved_as_a_thinking_block():
    accumulator = ChunkAccumulator(model="o-series")
    accumulator.handle({"choices": [{"delta": {"reasoning_content": "thinking..."}}]})
    accumulator.handle({"choices": [{"finish_reason": "stop", "delta": {"content": "done"}}]})
    kinds = {type(b).__name__ for b in accumulator.result().blocks}
    assert "ThinkingBlock" in kinds


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning", "thinking"])
def test_all_three_reasoning_key_spellings_are_recognized(key):
    # The module docstring's other documented divergence: "reasoning content
    # arrives under three different key names depending on who is serving."
    # The brief's own test above only drives "reasoning_content"; this pins
    # the other two as well, and that the recovered text is the right text.
    accumulator = ChunkAccumulator(model="o-series")
    accumulator.handle({"choices": [{"delta": {key: "because X"}}]})
    accumulator.handle({"choices": [{"finish_reason": "stop", "delta": {"content": "done"}}]})
    reasoning = next(b for b in accumulator.result().blocks if isinstance(b, ThinkingBlock))
    assert reasoning.text == "because X"


def test_finish_reason_length_maps_to_max_tokens():
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]})
    assert accumulator.result().stop is StopKind.MAX_TOKENS


def test_finish_reason_content_filter_maps_to_refusal():
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle(
        {"choices": [{"delta": {"content": "no"}, "finish_reason": "content_filter"}]}
    )
    assert accumulator.result().stop is StopKind.REFUSAL


def test_finish_reason_function_call_maps_to_tool_use():
    # The deprecated single-function-call shape some endpoints still send
    # instead of "tool_calls".
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "Read", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "function_call",
                }
            ]
        }
    )
    assert accumulator.result().stop is StopKind.TOOL_USE


def test_an_unrecognized_finish_reason_defaults_to_end_turn():
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle({"choices": [{"delta": {"content": "x"}, "finish_reason": "something_new"}]})
    assert accumulator.result().stop is StopKind.END_TURN


def test_usage_without_cached_token_detail_defaults_to_zero():
    # The other half of "(usage.get('prompt_tokens_details') or {}).get(...)"
    # -- a provider that sends prompt/completion tokens but no
    # prompt_tokens_details at all.
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle(
        {
            "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
    )
    reply = accumulator.result()
    assert reply.usage.input_tokens == 5
    assert reply.usage.cache_read_tokens == 0


def test_the_system_prompt_is_the_first_message_not_a_parameter():
    client = OpenAICompatClient("k", model="gpt-5", base_url="https://x/v1")
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert payload["messages"][0] == {"role": "system", "content": "sys"}


def test_tools_are_included_in_the_payload_when_present():
    client = OpenAICompatClient("k", model="gpt-5", base_url="https://x/v1")
    tools = (ToolSpec("Read", "Reads a file", {"type": "object", "properties": {}}),)
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), tools, 64))
    assert payload["tools"] == encode_tools(tools)


def test_no_tools_key_when_there_are_none():
    # The other half of "if request.tools:" above.
    client = OpenAICompatClient("k", model="gpt-5", base_url="https://x/v1")
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert "tools" not in payload


def test_no_temperature_is_sent_unless_one_is_asked_for():
    # Reasoning models refuse a non-default temperature; every server accepts
    # the parameter left out.
    client = OpenAICompatClient("k", model="gpt-5", base_url="https://x/v1")
    plain = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert "temperature" not in plain
    # 0.0 included: it is the one set value that is falsy.
    for value in (0.0, 0.3):
        asked = client.payload(
            ModelRequest("sys", Transcript((user_text("hi"),)), (), 64, temperature=value)
        )
        assert asked["temperature"] == value


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://API.OpenAI.com/v1/",
        "https://api.openai.com:443/v1",
        "http://api.openai.com/v1",
        "https://user@api.openai.com/v1",
        "https://api.openai.com./v1",
        "https://eu.api.openai.com/v1",
        "https://my-resource.openai.azure.com/openai/v1",
    ],
)
def test_openai_itself_is_sent_max_completion_tokens(base_url):
    # On OpenAI's own hosts and Azure, max_tokens is deprecated and refused by
    # reasoning models.
    client = OpenAICompatClient("k", model="gpt-5", base_url=base_url)
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert payload["max_completion_tokens"] == 64
    assert "max_tokens" not in payload


@pytest.mark.parametrize(
    "base_url",
    [
        "https://openrouter.ai/api/v1",
        "http://localhost:8000/v1",
        "https://api.openai.com.example/v1",
        "https://gateway.example/api.openai.com/v1",
        "https://notapi.openai.com/v1",
        "https://notopenai.azure.com/v1",
    ],
)
def test_other_compatible_servers_keep_max_tokens(base_url):
    # The parameter the other compatible servers understand.
    client = OpenAICompatClient("k", model="m", base_url=base_url)
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert payload["max_tokens"] == 64
    assert "max_completion_tokens" not in payload


def test_a_malformed_base_url_is_left_for_the_request_to_report():
    # Building the client must not raise: the first request reports the address.
    client = OpenAICompatClient("k", model="m", base_url="https://[abc/v1")
    payload = client.payload(ModelRequest("sys", Transcript((user_text("hi"),)), (), 64))
    assert payload["max_tokens"] == 64


def test_a_missing_api_key_names_the_adapter_and_the_fix():
    """Spec section 17.9, verbatim -- the exact wording and the em-dash both
    matter here, so this checks equality rather than a substring."""
    with pytest.raises(ModelError) as excinfo:
        OpenAICompatClient("", model="gpt-5", base_url="https://x/v1")
    assert str(excinfo.value) == (
        'no API key for adapter "openai_compat" — set OPENAI_API_KEY or run: ncc init'
    )
    assert not excinfo.value.retryable


def test_the_client_satisfies_the_model_client_protocol():
    client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1")
    assert isinstance(client, ModelClient)
    assert client.model_id == "gpt-5"


async def test_aclose_closes_a_client_it_created_itself():
    client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1")
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
        client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1", client=http)
        await client.aclose()
        assert not http.is_closed


async def test_an_http_error_is_classified_the_same_way_as_anthropic():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = OpenAICompatClient("k", model="m", base_url="https://x/v1", client=http)
        with pytest.raises(ModelError) as caught:
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
        assert caught.value.retryable


async def test_a_timeout_is_classified_as_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1", client=http)
        with pytest.raises(ModelError, match="timed out") as excinfo:
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert excinfo.value.retryable


async def test_a_connection_failure_is_classified_as_retryable():
    # httpx.ConnectError is a TransportError but not a TimeoutException -- the
    # except clause after the timeout-specific one, which the test above never
    # reaches.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("could not connect", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1", client=http)
        with pytest.raises(ModelError, match="could not reach the provider") as excinfo:
            await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert excinfo.value.retryable


def test_openai_text_cassette_decodes_to_one_text_block():
    reply = replay("openai_text.jsonl").result()
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.blocks[0].text == "hello"
    assert reply.stop is StopKind.END_TURN
    assert reply.usage.input_tokens == 11
    assert reply.usage.output_tokens == 2


def test_openai_tool_use_cassette_decodes_with_arguments_reassembled():
    reply = replay("openai_tool_use.jsonl").result()
    call = next(b for b in reply.blocks if isinstance(b, ToolUseBlock))
    assert call.name == "Read"
    assert call.id == "call_OpenAIReadFixture01"
    assert call.arguments == {"path": "README.md"}
    assert reply.stop is StopKind.TOOL_USE
    assert reply.usage.cache_read_tokens == 32


async def test_complete_streams_the_text_cassette_over_real_wire_text():
    # The one test that drives OpenAICompatClient.complete() all the way
    # through literal SSE wire text rebuilt from the committed cassette --
    # every test above either calls ChunkAccumulator.handle() directly or
    # replays decoded chunks, bypassing complete()'s own line-splitting,
    # "data:" stripping and "[DONE]" handling entirely. A trailing slash on
    # base_url is used here too, pinning that it does not produce "//".
    body = wire_body("openai_text.jsonl")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient(
            "test-key", model="gpt-5", base_url="https://x/v1/", client=http
        )
        reply = await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert str(seen[0].url) == "https://x/v1/chat/completions"
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.blocks[0].text == "hello"
    assert reply.stop is StopKind.END_TURN
    assert reply.usage.input_tokens == 11
    assert reply.usage.output_tokens == 2


async def test_a_blank_data_line_is_skipped_without_error():
    # The other half of "payload in ('', '[DONE]')" -- a keep-alive-style
    # "data:" line with nothing after it, distinct from the "[DONE]" sentinel
    # that every cassette-based test above already exercises.
    body = (
        b"data: \n\n"
        b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1", client=http)
        reply = await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    assert isinstance(reply.blocks[0], TextBlock)
    assert reply.blocks[0].text == "hi"


async def test_complete_streams_the_tool_use_cassette_over_real_wire_text():
    body = wire_body("openai_tool_use.jsonl")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        client = OpenAICompatClient("test-key", model="gpt-5", base_url="https://x/v1", client=http)
        reply = await client.complete(ModelRequest("s", Transcript((user_text("hi"),)), (), 10))
    call = next(b for b in reply.blocks if isinstance(b, ToolUseBlock))
    assert call.name == "Read"
    assert call.arguments == {"path": "README.md"}
    assert reply.stop is StopKind.TOOL_USE
    assert reply.usage.cache_read_tokens == 32


def test_the_missing_key_message_names_the_variable_the_config_says_to_use():
    with pytest.raises(ModelError) as excinfo:
        OpenAICompatClient(
            "",
            model="deepseek/deepseek-v3",
            base_url="https://x/v1",
            api_key_env="OPENROUTER_API_KEY",
        )
    assert str(excinfo.value) == (
        'no API key for adapter "openai_compat" \u2014 set OPENROUTER_API_KEY or run: ncc init'
    )


def _call_fragment(**function: object) -> dict[str, object]:
    return {
        "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": function}]}}]
    }


def test_a_stream_that_stops_after_the_tool_name_raises_instead_of_inventing_empty_arguments():
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle(_call_fragment(name="Write"))
    with pytest.raises(ModelError, match="before its tool calls were complete") as excinfo:
        accumulator.result()
    assert excinfo.value.retryable is False


def test_complete_looking_arguments_without_a_finish_reason_still_raise():
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle(_call_fragment(name="Read", arguments='{"path": "a.py"}'))
    with pytest.raises(ModelError, match="before its tool calls were complete"):
        accumulator.result()


def test_arguments_sent_as_an_object_are_used_as_is():
    accumulator = ChunkAccumulator(model="m")
    accumulator.handle(_call_fragment(name="Read", arguments={"path": "a.py"}))
    accumulator.handle({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    (block,) = accumulator.result().blocks
    assert isinstance(block, ToolUseBlock)
    assert dict(block.arguments) == {"path": "a.py"}


def test_an_error_chunk_whose_error_is_a_bare_string_is_a_model_error():
    with pytest.raises(ModelError, match="upstream exploded"):
        ChunkAccumulator(model="m").handle({"error": "upstream exploded"})


# -- streaming: on_text, and a stream that breaks midway (spec 7.4) --------------


def test_text_chunks_reach_on_text_in_order_and_join_to_the_reply_text():
    pieces: list[str] = []
    reply = replay("openai_text.jsonl", on_text=pieces.append).result()
    # The opening chunk carries an empty content, which is no text to show.
    assert pieces == ["hel", "lo"]
    assert reply.blocks == (TextBlock("hello"),)


def test_a_reply_with_a_tool_call_streams_its_text_and_none_of_its_arguments():
    pieces: list[str] = []
    reply = replay("openai_tool_use.jsonl", on_text=pieces.append).result()
    assert pieces == ["I'll read that file."]
    call = next(b for b in reply.blocks if isinstance(b, ToolUseBlock))
    assert call.arguments == {"path": "README.md"}  # the arguments were still collected


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning", "thinking"])
def test_reasoning_never_reaches_on_text(key):
    pieces: list[str] = []
    accumulator = ChunkAccumulator(model="o-series", on_text=pieces.append)
    accumulator.handle({"choices": [{"delta": {key: "because X"}}]})
    accumulator.handle({"choices": [{"finish_reason": "stop", "delta": {"content": "done"}}]})
    assert pieces == ["done"]
    assert accumulator.result().blocks == (ThinkingBlock("because X"), TextBlock("done"))


async def test_complete_hands_text_over_while_the_stream_is_still_arriving():
    # Not after it: a person reads a reply that takes half a minute as it is written.
    log: list[str] = []
    reply = await complete_from(
        arriving(payloads_of("openai_text.jsonl"), log=log),
        on_text=lambda piece: log.append(f"text {piece!r}"),
    )
    assert reply.blocks == (TextBlock("hello"),)
    assert log == [
        "sent 0",
        "sent 1",
        "text 'hel'",  # before the next chunk was even sent
        "sent 2",
        "text 'lo'",
        "sent 3",
        "sent 4",
        "sent 5",
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
    seen = payloads_of("openai_text.jsonl")[:2]  # the opening chunk, and "hel"
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=dropped), on_text=pieces.append)
    error = caught.value
    assert error.retryable is False
    # The usage chunk comes last, so none had arrived: there is nothing to record yet.
    assert error.partial == ModelReply((TextBlock("hel"),), StopKind.CUT_OFF, Usage(), "gpt-5")
    assert pieces == ["hel"]
    assert str(dropped) in str(error)


async def test_a_connection_lost_inside_a_tool_call_is_not_retried_either():
    # Nothing to keep: the half-received call is dropped, and asking again could run the
    # call twice if an earlier one in the turn already ran.
    chunks = payloads_of("openai_tool_use.jsonl")
    seen = [chunks[0], chunks[2], chunks[3]]  # no text; a call, and the start of its arguments
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    error = caught.value
    assert error.retryable is False
    assert error.partial == ModelReply((), StopKind.CUT_OFF, Usage(), "gpt-5")


@pytest.mark.parametrize(
    "dropped", [httpx.ReadError("connection reset"), httpx.ReadTimeout("timed out")]
)
async def test_a_connection_lost_before_anything_arrived_is_retryable_as_before(dropped):
    seen = payloads_of("openai_text.jsonl")[:1]  # the opening chunk, with no text in it
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=dropped))
    assert caught.value.retryable is True
    assert caught.value.partial is None


async def test_a_reasoning_only_stream_that_breaks_is_still_retryable():
    # Reasoning is not shown and cannot have run anything, so asking again costs tokens
    # and nothing else.
    seen = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]}]
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    assert caught.value.retryable is True


async def test_an_error_chunk_after_text_is_not_retried_and_hands_over_what_arrived():
    # An error the server sends inside the stream, not a dropped connection: the same rule.
    fault = {"error": {"message": "upstream exploded"}}
    seen = [*payloads_of("openai_text.jsonl")[:2], fault]
    with pytest.raises(ModelError, match="upstream exploded") as caught:
        await complete_from(arriving(seen))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(), "gpt-5"
    )


async def test_an_error_chunk_before_anything_arrived_carries_no_partial_reply():
    fault = {"error": {"message": "upstream exploded"}}
    with pytest.raises(ModelError, match="upstream exploded") as caught:
        await complete_from(arriving([fault]))
    assert caught.value.partial is None


def test_a_stream_that_ends_cleanly_mid_tool_call_still_hands_over_the_text_before_it():
    accumulator = ChunkAccumulator(model="gpt-5")
    # The text, a call, and part of its arguments: and then the stream ends.
    for chunk in payloads_of("openai_tool_use.jsonl")[:4]:
        accumulator.handle(chunk)
    with pytest.raises(ModelError, match="before its tool calls were complete") as caught:
        accumulator.result()
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("I'll read that file."),), StopKind.CUT_OFF, Usage(), "gpt-5"
    )


async def test_only_the_visible_text_is_kept_when_the_stream_breaks_after_reasoning_and_text():
    seen = [
        {"choices": [{"delta": {"reasoning_content": "weighing it"}}]},
        {"choices": [{"delta": {"content": "The answer"}}]},
    ]
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    assert caught.value.partial == ModelReply(
        (TextBlock("The answer"),), StopKind.CUT_OFF, Usage(), "gpt-5"
    )


# -- a stream that closes without its end marker was interrupted midway (spec 7.4) --


async def test_a_stream_that_closes_after_text_with_no_finish_reason_and_no_done_was_cut_off():
    seen = payloads_of("openai_text.jsonl")[:3]  # the opening chunk, "hel", "lo"
    pieces: list[str] = []
    with pytest.raises(ModelError, match="complete") as caught:
        await complete_from(arriving(seen), on_text=pieces.append)
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("hello"),), StopKind.CUT_OFF, Usage(), "gpt-5"
    )
    assert "".join(pieces) == "hello"


async def test_done_without_a_finish_reason_still_ends_the_reply():
    # Some servers close with [DONE] and never send a finish_reason.
    seen = [*payloads_of("openai_text.jsonl")[:3], "[DONE]"]
    reply = await complete_from(arriving(seen))
    assert reply.blocks == (TextBlock("hello"),)
    assert reply.stop is StopKind.END_TURN


async def test_a_stream_that_closes_after_a_call_with_only_done_keeps_the_call_whole():
    # The end marker is what makes a call safe to hand over, not its arguments looking whole.
    seen = [*payloads_of("openai_tool_use.jsonl")[:7], "[DONE]"]  # no finish_reason chunk
    reply = await complete_from(arriving(seen))
    call = next(b for b in reply.blocks if isinstance(b, ToolUseBlock))
    assert call.arguments == {"path": "README.md"}


async def test_reasoning_alone_with_no_end_marker_is_not_a_reply():
    seen = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]}]
    with pytest.raises(ModelError, match="complete") as caught:
        await complete_from(arriving(seen))
    assert caught.value.retryable is False
    assert caught.value.partial is None  # nothing a person could have read


# -- a stream that cannot be read is a stream that broke, not only one that dropped --


async def test_a_body_that_cannot_be_decoded_after_text_keeps_what_arrived():
    seen = payloads_of("openai_text.jsonl")[:2]  # the opening chunk, "hel"
    with pytest.raises(ModelError, match="bad gzip") as caught:
        await complete_from(arriving(seen, then=httpx.DecodingError("bad gzip data")))
    assert caught.value.retryable is False
    assert caught.value.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(), "gpt-5"
    )


async def test_a_line_that_is_not_json_after_text_keeps_what_arrived():
    seen = wire(payloads_of("openai_text.jsonl")[:2])
    response = httpx.Response(200, content=seen + b"data: {not json\n\n")
    with pytest.raises(ModelError, match="could not be read") as caught:
        await complete_from(response)
    assert caught.value.retryable is False
    assert caught.value.partial is not None
    assert caught.value.partial.blocks == (TextBlock("hel"),)


async def test_a_body_that_cannot_be_read_before_anything_arrived_raises_as_it_always_did():
    with pytest.raises(httpx.DecodingError):
        await complete_from(arriving([], then=httpx.DecodingError("bad gzip")))
    with pytest.raises(json.JSONDecodeError):
        await complete_from(httpx.Response(200, content=b"data: {oops\n\n"))


async def test_the_usage_the_server_had_reported_is_in_the_partial_reply():
    chunks = payloads_of("openai_text.jsonl")
    seen = [chunks[4], chunks[1]]  # the usage chunk, which some servers send first, and "hel"
    with pytest.raises(ModelError) as caught:
        await complete_from(arriving(seen, then=httpx.ReadError("connection reset")))
    assert caught.value.partial == ModelReply(
        (TextBlock("hel"),), StopKind.CUT_OFF, Usage(11, 2), "gpt-5"
    )
