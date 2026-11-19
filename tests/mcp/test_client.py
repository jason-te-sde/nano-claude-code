# tests/mcp/test_client.py
"""The client against the fake server in a real child process: which protocol era it finds,
what it asks in each, and what it does when the server misbehaves."""

import asyncio
import dataclasses
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from nanoclaude import __version__
from nanoclaude.mcp.client import (
    DISCOVER_TIMEOUT_SECONDS,
    LEGACY_VERSIONS,
    MAX_TOOL_PAGES,
    MAX_TOOLS,
    MODERN_VERSIONS,
    CallResult,
    McpClient,
    McpConnectionError,
    McpError,
    McpProtocolError,
    McpRequestError,
    McpTimeoutError,
    McpUnsupportedInteractionError,
    McpVersionError,
    RemoteTool,
)
from nanoclaude.mcp.jsonrpc import METHOD_NOT_FOUND, ErrorResponse, RpcError, encode
from nanoclaude.mcp.transport import TransportError
from tests.mcp.helpers import (
    eventually,
    group_is_gone,
    kill_helpers,
    received,
    received_methods,
)

VERSION = "io.modelcontextprotocol/protocolVersion"
MODERN_META = {
    VERSION: "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
    "io.modelcontextprotocol/clientInfo": {"name": "nano-claude-code", "version": __version__},
}
CLIENT_INFO = {"name": "nano-claude-code", "version": __version__}


T = TypeVar("T")


async def bounded(awaitable: Awaitable[T], seconds: float = 10.0) -> T:
    """Await ``awaitable``, failing the test instead of hanging it when a guard is missing."""
    return await asyncio.wait_for(awaitable, seconds)


def requests(log: Path, method: str) -> list[dict[str, Any]]:
    return [m for m in received(log) if m.get("method") == method]


# --------------------------------------------------------------------------
# What the client is made of
# --------------------------------------------------------------------------


def test_the_limits_and_versions_are_the_designs():
    assert DISCOVER_TIMEOUT_SECONDS == 5.0
    assert MAX_TOOL_PAGES == 20
    assert MAX_TOOLS == 512
    assert MODERN_VERSIONS == ("2026-07-28",)
    assert LEGACY_VERSIONS == ("2025-11-25", "2025-06-18", "2025-03-26")


def test_a_tool_and_a_result_are_plain_frozen_values():
    tool = RemoteTool("echo", "says it back", {"type": "object"})
    result = CallResult([{"type": "text", "text": "hi"}], {"a": 1}, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.is_error = True  # type: ignore[misc]
    assert (tool.name, tool.description, tool.input_schema) == (
        "echo",
        "says it back",
        {"type": "object"},
    )
    assert (result.content, result.structured_content, result.is_error) == (
        [{"type": "text", "text": "hi"}],
        {"a": 1},
        False,
    )


async def test_a_client_says_nothing_of_its_era_before_it_has_started(make_client):
    client = make_client("modern")
    with pytest.raises(McpError, match="start"):
        _ = client.era
    with pytest.raises(McpError, match="start"):
        _ = client.protocol_version
    with pytest.raises(McpError, match="start"):
        await client.list_tools()
    with pytest.raises(McpError, match="start"):
        await client.call_tool("echo", {}, timeout=1)


async def test_a_closed_client_cannot_be_started(make_client):
    client = make_client("modern")
    await client.close()
    with pytest.raises(McpError, match="the client is closed"):
        await client.start()


async def test_a_client_starts_once(make_client):
    client = make_client("modern")
    await client.start()
    with pytest.raises(McpError, match="already"):
        await client.start()


# --------------------------------------------------------------------------
# Finding the era
# --------------------------------------------------------------------------


async def test_a_modern_server_is_found_by_one_discover_request(make_client, log_path):
    client = make_client("modern")
    await bounded(client.start())
    assert client.era == "modern"
    assert client.protocol_version == "2026-07-28"
    await client.close()
    assert received_methods(log_path) == ["server/discover"]
    assert requests(log_path, "server/discover")[0]["params"] == {"_meta": MODERN_META}


async def test_every_request_after_a_modern_discover_carries_the_same_meta(make_client, log_path):
    client = make_client("modern")
    await client.start()
    await client.list_tools()
    await client.call_tool("echo", {"text": "hi"}, timeout=5)
    await client.close()
    later = [*requests(log_path, "tools/list"), *requests(log_path, "tools/call")]
    assert len(later) == 2
    for message in later:
        assert message["params"]["_meta"] == MODERN_META, message


async def test_a_server_that_answers_both_ways_is_modern_and_is_never_initialized(
    make_client, log_path
):
    client = make_client("dual")
    await bounded(client.start())
    assert client.era == "modern"
    await client.close()
    assert received_methods(log_path) == ["server/discover"]


async def test_a_legacy_server_is_initialized_after_it_refuses_discover(make_client, log_path):
    client = make_client("legacy")
    await bounded(client.start())
    assert (client.era, client.protocol_version) == ("legacy", "2025-11-25")
    await client.list_tools()
    await client.close()
    assert received_methods(log_path) == [
        "server/discover",
        "initialize",
        "notifications/initialized",
        "tools/list",
    ]
    initialize = requests(log_path, "initialize")[0]
    assert initialize["params"] == {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": CLIENT_INFO,
    }
    initialized = requests(log_path, "notifications/initialized")[0]
    assert "id" not in initialized
    # A legacy request carries nothing of the modern handshake.
    assert "_meta" not in (requests(log_path, "tools/list")[0].get("params") or {})


@pytest.mark.parametrize("reply", LEGACY_VERSIONS)
async def test_a_legacy_reply_naming_a_version_the_client_knows_is_accepted(make_client, reply):
    client = make_client("legacy", f"legacy_version={reply}")
    await bounded(client.start())
    assert (client.era, client.protocol_version) == ("legacy", reply)


@pytest.mark.parametrize("reply", ["2024-11-05", "2099-01-01", "latest", "2026-07-28", ""])
async def test_a_legacy_reply_naming_any_other_version_fails_and_ends_the_server(
    make_client, transports, reply
):
    client = make_client("legacy", f'legacy_version="{reply}"')
    with pytest.raises(McpVersionError, match="initialize"):
        await bounded(client.start())
    pid = transports[client].pid
    assert pid is not None and await eventually(lambda: group_is_gone(pid))


@pytest.mark.parametrize("code", [-32601, -32600, -32602, -32603, -32000, 1])
async def test_any_other_error_to_discover_means_legacy(make_client, log_path, code):
    client = make_client("legacy", "discover=error", f"discover_code={code}")
    await bounded(client.start())
    assert client.era == "legacy"
    await client.close()
    assert received_methods(log_path)[:2] == ["server/discover", "initialize"]


async def test_a_result_to_discover_that_is_not_an_object_means_legacy(make_client):
    client = make_client("legacy", "discover=scalar")
    await bounded(client.start())
    assert client.era == "legacy"


async def test_no_answer_to_discover_within_the_time_means_legacy(make_client, log_path):
    client = make_client("silent-discover", discover_timeout=0.2)
    await bounded(client.start())
    assert client.era == "legacy"
    await client.list_tools()
    await client.close()
    assert received_methods(log_path)[:3] == [
        "server/discover",
        "initialize",
        "notifications/initialized",
    ]


async def test_a_discover_answer_that_comes_too_late_is_not_taken_for_a_reply(make_client):
    # The server answers discover only after the client has given up on it; what arrives is
    # a reply to nothing, and the client carries on as the legacy client it has become.
    client = make_client("legacy", "discover=late", discover_timeout=0.2)
    await bounded(client.start())
    assert client.era == "legacy"
    assert [tool.name for tool in await client.list_tools()] == ["echo"]


async def test_a_version_the_server_does_not_speak_is_retried_with_one_it_does(
    make_client, log_path
):
    client = make_client("modern", modern_versions=("2027-01-01", "2026-07-28"))
    await bounded(client.start())
    assert (client.era, client.protocol_version) == ("modern", "2026-07-28")
    await client.list_tools()
    await client.close()
    versions = [m["params"]["_meta"][VERSION] for m in requests(log_path, "server/discover")]
    assert versions == ["2027-01-01", "2026-07-28"]
    assert requests(log_path, "tools/list")[0]["params"]["_meta"][VERSION] == "2026-07-28"
    assert "initialize" not in received_methods(log_path)


async def test_a_server_with_no_version_in_common_fails_and_is_not_asked_to_initialize(
    make_client, transports, log_path
):
    client = make_client("unsupported-version")
    with pytest.raises(McpVersionError, match="2099-01-01"):
        await bounded(client.start())
    pid = transports[client].pid
    assert pid is not None and await eventually(lambda: group_is_gone(pid))
    # The client says why it failed, and says the same to whoever asks it for anything after.
    assert isinstance(client.failure, McpVersionError)
    with pytest.raises(McpVersionError, match="2099-01-01"):
        await client.list_tools()
    await client.close()
    # A client that took -32022 for "some other error" would have fallen back to legacy and
    # started, which is the thing that must not happen.
    assert received_methods(log_path) == ["server/discover"]


async def test_a_version_the_client_already_tried_is_not_tried_again(make_client, log_path):
    # The server refuses the one version this client has, and names it among those it speaks.
    client = make_client("modern", "discover=unsupported")
    with pytest.raises(McpVersionError):
        await bounded(client.start())
    await client.close()
    assert len(requests(log_path, "server/discover")) == 1


async def test_an_initialize_the_server_refuses_fails_with_its_error(make_client):
    client = make_client("legacy", "initialize=method-not-found")
    with pytest.raises(McpRequestError) as caught:
        await bounded(client.start())
    assert caught.value.error.code == -32601


async def test_an_initialize_that_is_never_answered_times_out_and_ends_the_server(
    make_client, transports
):
    client = make_client("legacy", "initialize=silent", request_timeout=0.3)
    with pytest.raises(McpTimeoutError, match="initialize"):
        await bounded(client.start())
    pid = transports[client].pid
    assert pid is not None and await eventually(lambda: group_is_gone(pid))
    assert isinstance(client.failure, McpTimeoutError)


async def test_a_command_that_cannot_be_run_fails_to_start(make_transport):
    client = McpClient(make_transport("/nonexistent/server-binary"))
    with pytest.raises(McpConnectionError, match="server-binary"):
        await client.start()
    await client.close()


async def test_a_server_that_exits_at_once_fails_to_start(make_client):
    client = make_client("crash-on-start")
    with pytest.raises(McpConnectionError):
        await bounded(client.start())
    assert isinstance(client.failure, McpConnectionError)


# --------------------------------------------------------------------------
# Listing tools
# --------------------------------------------------------------------------


async def test_the_tools_come_back_as_the_server_sent_them(make_client):
    client = make_client("modern")
    await client.start()
    assert await client.list_tools() == [
        RemoteTool(
            "echo",
            "Fake tool number 0.",
            {"type": "object", "properties": {"text": {"type": "string"}}},
        )
    ]


async def test_paginated_tools_are_followed_by_cursor_in_order(make_client, log_path):
    client = make_client("paginated")
    await client.start()
    tools = await client.list_tools()
    await client.close()
    assert [tool.name for tool in tools] == [
        "echo",
        "tool_1",
        "tool_2",
        "tool_3",
        "tool_4",
        "tool_5",
    ]
    cursors = [(m.get("params") or {}).get("cursor") for m in requests(log_path, "tools/list")]
    assert cursors == [None, "page-1", "page-2"]


async def test_twenty_pages_are_read_and_a_twenty_first_is_an_error(make_client, log_path):
    client = make_client("paginated", "pages=20", "per_page=1")
    await client.start()
    assert len(await bounded(client.list_tools())) == 20
    await client.close()
    assert len(requests(log_path, "tools/list")) == MAX_TOOL_PAGES


async def test_a_server_that_never_stops_paginating_is_cut_off_after_twenty_requests(
    make_client, log_path
):
    client = make_client("paginated", "pages=1000", "per_page=1")
    await client.start()
    with pytest.raises(McpProtocolError) as caught:
        await bounded(client.list_tools())
    assert str(caught.value) == (
        "the server offered 20 tools and still has more pages after 20 pages, "
        "past the limit of 20 pages"
    )
    await client.close()
    assert len(requests(log_path, "tools/list")) == MAX_TOOL_PAGES


async def test_five_hundred_and_twelve_tools_are_read_and_a_five_hundred_and_thirteenth_is_an_error(
    make_client,
):
    ok = make_client("paginated", "pages=1", "per_page=512")
    await ok.start()
    assert len(await ok.list_tools()) == 512
    too_many = make_client("paginated", "pages=1", "per_page=513")
    await too_many.start()
    with pytest.raises(McpProtocolError) as caught:
        await too_many.list_tools()
    assert str(caught.value) == (
        "the server offered 513 tools over 1 page(s), past the limit of 512 tools"
    )


async def test_the_tool_limit_counts_across_pages_and_stops_the_requests(make_client, log_path):
    client = make_client("paginated", "pages=50", "per_page=200")
    await client.start()
    with pytest.raises(McpProtocolError) as caught:
        await bounded(client.list_tools())
    # Six hundred is what the third page brought it to, and the server had more to give.
    assert str(caught.value) == (
        "the server offered at least 600 tools over 3 page(s), past the limit of 512 tools"
    )
    await client.close()
    assert len(requests(log_path, "tools/list")) == 3, "paging went on after the limit was passed"


async def test_a_cursor_that_is_not_a_string_is_an_error(make_client):
    client = make_client("paginated", "bad_cursor=7")
    await client.start()
    with pytest.raises(McpProtocolError, match="nextCursor"):
        await bounded(client.list_tools())


async def test_a_tools_field_that_is_not_a_list_is_an_error(make_client):
    client = make_client("modern", 'list_patch={"tools": "echo"}')
    await client.start()
    with pytest.raises(McpProtocolError, match="must be a list"):
        await client.list_tools()


async def test_a_tool_without_a_name_is_an_error(make_client):
    client = make_client("modern", "tool_without_name=true")
    await client.start()
    with pytest.raises(McpProtocolError, match="name"):
        await client.list_tools()


@pytest.mark.parametrize("method", ["initialize", "tools/list", "tools/call"])
async def test_a_result_that_is_not_an_object_is_an_error(make_client, method):
    mode = "legacy" if method == "initialize" else "modern"
    client = make_client(mode, f"non_object_result={method}")
    if method == "initialize":
        with pytest.raises(McpProtocolError, match="object"):
            await bounded(client.start())
        return
    await client.start()
    with pytest.raises(McpProtocolError, match="object"):
        if method == "tools/list":
            await client.list_tools()
        else:
            await client.call_tool("echo", {}, timeout=5)


# --------------------------------------------------------------------------
# Calling a tool
# --------------------------------------------------------------------------


async def test_a_result_is_returned_with_its_content_and_structured_content_as_sent(
    make_client, log_path
):
    client = make_client("modern")
    await client.start()
    result = await client.call_tool("echo", {"text": "hi"}, timeout=5)
    await client.close()
    assert result == CallResult(
        [{"type": "text", "text": "echo:{'text': 'hi'}"}], {"arguments": {"text": "hi"}}, False
    )
    params = requests(log_path, "tools/call")[0]["params"]
    assert (params["name"], params["arguments"]) == ("echo", {"text": "hi"})


async def test_an_error_result_is_reported_as_one_and_not_raised(make_client):
    client = make_client("is-error")
    await client.start()
    result = await client.call_tool("echo", {}, timeout=5)
    assert result.is_error is True
    assert result.content == [{"type": "text", "text": "boom"}]
    assert result.structured_content is None


async def test_a_json_rpc_error_is_raised_with_its_code_message_and_data(make_client):
    client = make_client("rpc-error")
    await client.start()
    with pytest.raises(McpRequestError) as caught:
        await client.call_tool("echo", {}, timeout=5)
    assert caught.value.error.code == -32000
    assert caught.value.error.message == "the tool exploded"
    assert caught.value.error.data == {"detail": "fake"}
    assert "the tool exploded" in str(caught.value)


async def test_a_result_with_no_result_type_is_complete(make_client):
    client = make_client("no-result-type")
    await client.start()
    assert (await client.list_tools())[0].name == "echo"
    assert (await client.call_tool("echo", {}, timeout=5)).is_error is False


async def test_a_result_type_of_complete_is_complete(make_client):
    client = make_client("modern")
    await client.start()
    assert (await client.call_tool("echo", {}, timeout=5)).is_error is False


@pytest.mark.parametrize("result_type", ["input_required", "a-kind-of-the-future", ""])
async def test_any_other_result_type_is_an_unsupported_interaction_on_that_call(
    make_client, result_type
):
    client = make_client("input-required", f'call_result_type="{result_type}"')
    await client.start()
    with pytest.raises(McpUnsupportedInteractionError) as caught:
        await client.call_tool("echo", {}, timeout=5)
    assert caught.value.result_type == result_type
    assert repr(result_type) in str(caught.value)
    # The call failed; the server did not, and the next call is a call like any other.
    assert [tool.name for tool in await client.list_tools()] == ["echo"]


async def test_a_listing_that_is_not_complete_is_an_unsupported_interaction_too(make_client):
    client = make_client("modern", "list_result_type=input_required")
    await client.start()
    with pytest.raises(McpUnsupportedInteractionError):
        await client.list_tools()


@pytest.mark.parametrize(
    "patch",
    [
        '{"content": "text"}',
        '{"content": {"type": "text"}}',
        '{"isError": "yes"}',
        '{"isError": 1}',
    ],
)
async def test_a_call_result_of_the_wrong_shape_is_an_error(make_client, patch):
    client = make_client("modern", f"call_patch={patch}")
    await client.start()
    with pytest.raises(McpProtocolError):
        await client.call_tool("echo", {}, timeout=5)


@pytest.mark.parametrize("value", [object(), float("nan"), {1, 2}], ids=["object", "nan", "set"])
async def test_arguments_that_cannot_be_written_fail_the_call_and_not_the_client(
    make_client, value
):
    client = make_client("modern")
    await client.start()
    with pytest.raises(McpProtocolError, match="encode"):
        await client.call_tool("echo", {"value": value}, timeout=5)
    assert client.failure is None
    assert (await client.call_tool("echo", {"value": 1}, timeout=5)).is_error is False


async def test_a_result_with_no_content_has_an_empty_content_list(make_client):
    client = make_client("modern", 'call_patch={"structuredContent": {"only": 1}}')
    await client.start()
    result = await client.call_tool("echo", {}, timeout=5)
    assert result.structured_content == {"only": 1}


async def test_concurrent_calls_each_get_their_own_id_and_their_own_answer(make_client, log_path):
    client = make_client("modern")
    await client.start()
    results = await bounded(
        asyncio.gather(*(client.call_tool("echo", {"n": n}, timeout=10) for n in range(20)))
    )
    await client.close()
    assert [r.structured_content for r in results] == [{"arguments": {"n": n}} for n in range(20)]
    every = [m["id"] for m in received(log_path) if "id" in m and "method" in m]
    assert all(type(i) is int and i >= 1 for i in every), every
    assert len(set(every)) == len(every), f"an id was used twice: {every}"


async def test_a_call_that_is_not_answered_in_time_is_cancelled_and_raises(make_client, log_path):
    client = make_client("slow")
    await client.start()
    with pytest.raises(McpTimeoutError) as caught:
        await bounded(client.call_tool("echo", {}, timeout=0.2))
    await client.close()
    messages = received(log_path)
    call = next(m for m in messages if m.get("method") == "tools/call")
    cancel = next(m for m in messages if m.get("method") == "notifications/cancelled")
    assert caught.value.request_id == call["id"]
    assert cancel["params"]["requestId"] == call["id"]
    assert isinstance(cancel["params"]["reason"], str) and cancel["params"]["reason"]
    assert "id" not in cancel, "a cancellation is a notification and wants no answer"
    assert messages.index(call) < messages.index(cancel)


async def test_a_legacy_call_that_is_not_answered_in_time_is_cancelled_too(make_client, log_path):
    client = make_client("slow", "discover=method-not-found", "initialize=ok")
    await client.start()
    assert client.era == "legacy"
    with pytest.raises(McpTimeoutError):
        await bounded(client.call_tool("echo", {}, timeout=0.2))
    await client.close()
    assert len(requests(log_path, "notifications/cancelled")) == 1


async def test_the_connection_survives_a_timeout_and_a_late_answer_is_ignored(
    make_client, log_path
):
    client = make_client("modern", "call_delay=0.3")
    await client.start()
    with pytest.raises(McpTimeoutError):
        await bounded(client.call_tool("echo", {"n": 1}, timeout=0.1))
    # The server answers the first call, late, before it reads anything more.
    assert [tool.name for tool in await bounded(client.list_tools())] == ["echo"]
    again = await bounded(client.call_tool("echo", {"n": 2}, timeout=10))
    assert again.structured_content == {"arguments": {"n": 2}}
    assert client.failure is None


async def test_a_reply_sent_twice_is_taken_once_and_the_second_is_ignored(make_client):
    client = make_client("modern", "duplicate_replies=true")
    await bounded(client.start())
    assert [tool.name for tool in await bounded(client.list_tools())] == ["echo"]
    assert (await bounded(client.call_tool("echo", {}, timeout=5))).is_error is False
    assert [tool.name for tool in await bounded(client.list_tools())] == ["echo"]
    assert client.failure is None


async def test_no_request_is_left_in_flight_once_answered_failed_or_given_up_on(make_client):
    slow = make_client("slow")
    await slow.start()
    await slow.list_tools()
    with pytest.raises(McpTimeoutError):
        await slow.call_tool("echo", {}, timeout=0.1)
    broken = make_client("rpc-error")
    await broken.start()
    with pytest.raises(McpRequestError):
        await broken.call_tool("echo", {}, timeout=5)
    # An id that stays in this table is one the client believes is still waiting.
    assert slow._pending == {} and broken._pending == {}, (slow._pending, broken._pending)


async def test_a_call_waits_no_longer_than_the_client_default_when_it_is_given_no_time(
    make_client,
):
    client = make_client("slow", request_timeout=0.2)
    await client.start()
    with pytest.raises(McpTimeoutError):
        await bounded(client.call_tool("echo", {}))


# --------------------------------------------------------------------------
# A server that misbehaves
# --------------------------------------------------------------------------


async def test_what_the_server_sends_that_is_not_a_message_is_skipped(make_client):
    client = make_client("modern", "stdout_noise=true")
    await bounded(client.start())
    assert [tool.name for tool in await bounded(client.list_tools())] == ["echo"]
    assert (await bounded(client.call_tool("echo", {}, timeout=5))).is_error is False
    assert client.failure is None


async def test_server_requests_get_ping_an_empty_result_and_the_rest_method_not_found(
    make_client, log_path
):
    client = make_client("server-request")
    await client.start()
    assert [tool.name for tool in await client.list_tools()] == ["echo"]
    await client.close()
    answers = {m["id"]: m for m in received(log_path) if str(m.get("id", "")).startswith("srv-")}
    assert set(answers) == {"srv-0", "srv-1", "srv-2", "srv-3"}, answers
    for name in ("srv-0", "srv-1", "srv-2"):  # sampling, roots, elicitation
        assert answers[name]["error"]["code"] == -32601, answers[name]
        assert "result" not in answers[name]
    assert answers["srv-3"]["result"] == {}, answers["srv-3"]
    assert "error" not in answers["srv-3"]


async def test_server_notifications_are_ignored_and_never_answered(make_client, log_path):
    client = make_client("server-request")
    await client.start()
    await client.list_tools()
    await client.close()
    # The server sent two notifications; everything the client wrote back is a request of
    # its own or an answer to one of the server's four requests.
    stray = [
        m
        for m in received(log_path)
        if "method" not in m and not str(m.get("id", "")).startswith("srv-")
    ]
    assert stray == []
    assert client.failure is None


async def test_a_server_that_exits_after_the_handshake_fails_every_later_request(make_client):
    client = make_client("crash")
    await bounded(client.start())
    with pytest.raises(McpConnectionError):
        await bounded(client.list_tools())
    assert isinstance(client.failure, McpConnectionError)
    with pytest.raises(McpConnectionError):
        await bounded(client.call_tool("echo", {}, timeout=5))


async def test_a_line_of_exactly_four_mebibytes_from_the_server_is_a_result(make_client):
    client = make_client("oversized-line", "line_size=4194304")
    await client.start()
    result = await bounded(client.call_tool("echo", {}, timeout=20), 30)
    assert result.is_error is False
    assert client.failure is None


async def test_a_line_over_four_mebibytes_fails_the_server_and_ends_its_process(
    make_client, transports
):
    client = make_client("oversized-line")
    await client.start()
    with pytest.raises(McpProtocolError, match="4194304"):
        await bounded(client.call_tool("echo", {}, timeout=20), 30)
    assert isinstance(client.failure, McpProtocolError)
    pid = transports[client].pid
    assert pid is not None
    # Without anyone calling close: the client ends the process it has failed.
    assert await eventually(lambda: group_is_gone(pid), within=10)
    with pytest.raises(McpProtocolError):
        await client.list_tools()


async def test_a_hundred_requests_from_the_server_are_each_answered(make_client, log_path):
    client = make_client("request-flood", "flood_requests=100", "flood_then_stop_reading=false")
    await client.start()
    result = await bounded(client.call_tool("echo", {}, timeout=20))
    await client.close()
    assert result.content == [{"type": "text", "text": "after the flood"}]
    answers = [m for m in received(log_path) if str(m.get("id", "")).startswith("srv-")]
    assert [m["id"] for m in answers] == [f"srv-{n}" for n in range(100)]
    assert all(m["result"] == {} for m in answers)


async def test_a_server_that_floods_requests_and_reads_no_answer_fails_the_client(
    make_client, transports
):
    # Twenty thousand requests, and then the answer to the call, from a server that has
    # stopped reading its input. The answers fill the pipe; a reader that waited for it to
    # empty would never get as far as the answer to the call, which would wait out its own
    # minute. This one gives up on the server, and says why.
    client = make_client("request-flood", transport_options={"stdin_grace": 0.2, "term_grace": 0.5})
    await client.start()
    with pytest.raises(McpConnectionError, match="not reading its input"):
        await bounded(client.call_tool("echo", {}, timeout=60), 20)
    assert isinstance(client.failure, McpConnectionError)
    pid = transports[client].pid
    assert pid is not None and await eventually(lambda: group_is_gone(pid), within=10)


async def test_a_server_that_exits_is_noticed_even_if_a_child_keeps_its_output_open(
    make_client, log_path
):
    client = make_client(
        "escaped-helper", "exit_after_handshake=true", transport_options={"drain_grace": 0.2}
    )
    try:
        await bounded(client.start())
        # The server has exited, and a child that left its group holds stdout and stdin open,
        # so nothing ends the stream. A client that waited for the stream would wait the
        # whole of the thirty seconds a request is given.
        began = time.monotonic()
        with pytest.raises(McpConnectionError, match="exited or crashed"):
            await bounded(client.list_tools(), 10)
        assert time.monotonic() - began < 5
        assert isinstance(client.failure, McpConnectionError)
        with pytest.raises(McpConnectionError):
            await bounded(client.call_tool("echo", {}, timeout=30), 5)
    finally:
        await client.close()
        kill_helpers(log_path)


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


async def test_closing_ends_the_server_and_a_closed_client_refuses_requests(
    make_client, transports
):
    client = make_client("modern")
    await client.start()
    pid = transports[client].pid
    assert pid is not None
    await client.close()
    assert group_is_gone(pid)
    with pytest.raises(McpError, match="closed"):
        await client.list_tools()
    with pytest.raises(McpError, match="closed"):
        await client.call_tool("echo", {}, timeout=1)
    await client.close()


async def test_closing_while_a_call_is_waiting_fails_the_call(make_client):
    client = make_client("slow")
    await client.start()
    waiting = asyncio.create_task(client.call_tool("echo", {}, timeout=30))
    await asyncio.sleep(0)
    await client.close()
    with pytest.raises(McpError, match="closed"):
        await bounded(waiting)


async def test_closing_a_client_that_never_started_is_fine(make_client):
    await make_client("modern").close()


# --------------------------------------------------------------------------
# A transport that does what the test says
# --------------------------------------------------------------------------


class ScriptedTransport:
    """Hands the client the lines it is told to, then waits; for what a real process rarely
    does: a pipe that breaks, an exception of our own, a server that dies between two requests."""

    def __init__(
        self, incoming: list[bytes | Exception | None], send_error: Exception | None = None
    ) -> None:
        self._incoming = list(incoming)
        self._send_error = send_error
        self.sent: list[bytes] = []
        self.closed = False

    async def start(self) -> None:
        pass

    async def send(self, line: bytes) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent.append(line)

    async def receive(self) -> bytes | None:
        if not self._incoming:
            await asyncio.Event().wait()  # until the client's reader is cancelled
        item = self._incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


async def test_a_pipe_that_cannot_be_written_fails_the_start_and_ends_the_server():
    transport = ScriptedTransport([], send_error=TransportError("the server is not reading"))
    client = McpClient(transport)
    with pytest.raises(McpConnectionError, match="not reading"):
        await bounded(client.start())
    assert transport.closed
    assert isinstance(client.failure, McpConnectionError)


async def test_a_pipe_that_breaks_while_the_client_waits_fails_what_is_waiting():
    transport = ScriptedTransport([TransportError("the pipe broke")])
    client = McpClient(transport)
    with pytest.raises(McpConnectionError) as caught:
        await bounded(client.start())
    # The transport's own words, and not the catch-all's account of a failure of the reader.
    assert str(caught.value) == "the pipe broke"
    assert transport.closed


async def test_a_failure_of_the_readers_own_fails_what_is_waiting_instead_of_hanging_it():
    transport = ScriptedTransport([RuntimeError("a bug of ours")])
    client = McpClient(transport)
    with pytest.raises(McpConnectionError, match="a bug of ours"):
        await bounded(client.start())
    assert transport.closed


async def test_a_server_that_goes_between_discover_and_initialize_fails_the_start_at_once():
    refusal = encode(ErrorResponse(1, RpcError(METHOD_NOT_FOUND, "Method not found")))
    transport = ScriptedTransport([refusal, None])
    # A request timeout this long would show if the client had waited for an answer that
    # nobody is left to give.
    client = McpClient(transport, request_timeout=60)
    with pytest.raises(McpConnectionError, match="closed its output"):
        await bounded(client.start(), 5)
