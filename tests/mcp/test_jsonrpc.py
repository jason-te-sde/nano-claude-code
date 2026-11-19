# tests/mcp/test_jsonrpc.py
"""The JSON-RPC 2.0 codec: one message per line, and what a line must be to count as one."""

import json

import pytest

from nanoclaude.mcp.jsonrpc import (
    MAX_LINE_BYTES,
    ErrorResponse,
    LineTooLongError,
    Notification,
    ProtocolError,
    Request,
    RequestIds,
    Response,
    RpcError,
    decode,
    encode,
)


def line(**fields: object) -> bytes:
    return json.dumps({"jsonrpc": "2.0", **fields}).encode("utf-8")


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def test_a_request_is_one_compact_line_without_a_terminator():
    encoded = encode(Request(7, "tools/list", {"cursor": "c1"}))
    assert b"\n" not in encoded and b"\r" not in encoded, encoded
    assert json.loads(encoded) == {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/list",
        "params": {"cursor": "c1"},
    }
    assert b": " not in encoded and b", " not in encoded, f"not compact: {encoded!r}"


def test_a_request_without_params_has_no_params_key():
    assert "params" not in json.loads(encode(Request(1, "ping")))
    assert "params" not in json.loads(encode(Notification("notifications/initialized")))


def test_the_other_message_kinds_encode_to_their_wire_shape():
    assert json.loads(encode(Notification("notifications/cancelled", {"requestId": 3}))) == {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 3},
    }
    assert json.loads(encode(Response("srv-1", {}))) == {
        "jsonrpc": "2.0",
        "id": "srv-1",
        "result": {},
    }
    assert json.loads(encode(Response(2, None))) == {"jsonrpc": "2.0", "id": 2, "result": None}
    error = ErrorResponse(4, RpcError(-32601, "Method not found"))
    assert json.loads(encode(error)) == {
        "jsonrpc": "2.0",
        "id": 4,
        "error": {"code": -32601, "message": "Method not found"},
    }
    with_data = ErrorResponse(5, RpcError(-32022, "unsupported", {"supported": ["2026-07-28"]}))
    assert json.loads(encode(with_data))["error"]["data"] == {"supported": ["2026-07-28"]}


def test_text_that_would_break_a_line_stays_on_one():
    text = "a\nb\r\nc\u2028d\u2029e \ud800 f \u00e9 \u4e2d"
    encoded = encode(Request(1, "tools/call", {"arguments": {"text": text}}))
    assert b"\n" not in encoded and b"\r" not in encoded, encoded
    assert json.loads(encoded)["params"]["arguments"]["text"] == text


@pytest.mark.parametrize("bad_id", [None, True, False, 1.5, ["x"]])
def test_a_request_id_is_never_null_or_anything_but_an_integer_or_a_string(bad_id):
    with pytest.raises(ProtocolError, match="id"):
        encode(Request(bad_id, "ping"))
    with pytest.raises(ProtocolError, match="id"):
        encode(Response(bad_id, {}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), {1, 2}])
def test_a_value_json_cannot_carry_is_refused_not_written_as_something_else(value):
    with pytest.raises(ProtocolError, match="cannot encode"):
        encode(Request(1, "tools/call", {"arguments": {"value": value}}))


def test_a_line_over_the_limit_is_refused_when_it_is_written_too():
    ok = encode(Request(1, "x", {"p": "a" * (MAX_LINE_BYTES - 60)}))
    assert len(ok) <= MAX_LINE_BYTES
    with pytest.raises(LineTooLongError):
        encode(Request(1, "x", {"p": "a" * MAX_LINE_BYTES}))


def test_the_limit_is_four_mebibytes():
    assert MAX_LINE_BYTES == 4 * 1024 * 1024


# --------------------------------------------------------------------------
# Request ids
# --------------------------------------------------------------------------


def test_request_ids_are_distinct_positive_integers():
    ids = RequestIds()
    issued = [ids.next() for _ in range(5_000)]
    assert len(set(issued)) == len(issued), "an id was handed out twice"
    assert all(type(i) is int and i >= 1 for i in issued), issued[:5]


def test_two_sources_do_not_share_state():
    assert RequestIds().next() == RequestIds().next() == 1


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def test_a_request_decodes_with_its_id_method_and_params():
    assert decode(line(id=3, method="ping")) == Request(3, "ping", None)
    assert decode(line(id="a", method="sampling/createMessage", params={"x": 1})) == Request(
        "a", "sampling/createMessage", {"x": 1}
    )


def test_a_message_without_an_id_is_a_notification():
    assert decode(line(method="notifications/message", params={"level": "info"})) == Notification(
        "notifications/message", {"level": "info"}
    )
    assert decode(line(method="notifications/tools/list_changed")) == Notification(
        "notifications/tools/list_changed", None
    )


def test_a_result_decodes_even_when_it_is_null():
    assert decode(line(id=1, result={"ok": True})) == Response(1, {"ok": True})
    assert decode(line(id=1, result=None)) == Response(1, None)


def test_an_error_carries_code_message_and_optional_data():
    decoded = decode(line(id=9, error={"code": -32022, "message": "m", "data": {"supported": []}}))
    assert decoded == ErrorResponse(9, RpcError(-32022, "m", {"supported": []}))
    assert decode(line(id=9, error={"code": -1, "message": "m"})) == ErrorResponse(
        9, RpcError(-1, "m", None)
    )


def test_an_error_answering_no_request_may_have_a_null_id():
    assert decode(line(id=None, error={"code": -32700, "message": "Parse error"})) == ErrorResponse(
        None, RpcError(-32700, "Parse error", None)
    )


def test_a_trailing_line_terminator_is_not_part_of_the_message():
    assert decode(line(id=1, result={}) + b"\n") == Response(1, {})
    assert decode(line(id=1, result={}) + b"\r\n") == Response(1, {})


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        (b"", "an empty line"),
        (b"   ", "a blank line"),
        (b"not json", "text that is not JSON"),
        (b'{"jsonrpc": "2.0", "id": 1', "a truncated message"),
        (b"\xff\xfe{}", "bytes that are not UTF-8"),
        (b"[1, 2]", "a batch"),
        (b'"text"', "a bare string"),
        (b"null", "null"),
        (b'{"id": 1, "result": {}}', "no version"),
        (b'{"jsonrpc": "1.0", "id": 1, "result": {}}', "the wrong version"),
        (b'{"jsonrpc": 2.0, "id": 1, "result": {}}', "a numeric version"),
        (line(id=1), "neither method, result nor error"),
        (line(id=1, result={}, error={"code": 1, "message": "m"}), "both result and error"),
        (line(id=1, method="m", result={}), "a method and a result"),
        (line(id=None, method="m"), "a request whose id is null"),
        (line(id=None, result={}), "a result whose id is null"),
        (line(result={}), "a result with no id"),
        (line(id=True, method="m"), "a boolean id"),
        (line(id=1.5, method="m"), "a fractional id"),
        (line(id=[1], result={}), "a list as an id"),
        (line(id=1, method=5), "a method that is not a string"),
        (line(id=1, method="m", params=[1]), "params that are a list"),
        (line(id=1, method="m", params="x"), "params that are a string"),
        (line(id=1, error="boom"), "an error that is not an object"),
        (line(id=1, error={"message": "m"}), "an error without a code"),
        (line(id=1, error={"code": True, "message": "m"}), "a boolean code"),
        (line(id=1, error={"code": 1.5, "message": "m"}), "a fractional code"),
        (line(id=1, error={"code": 1}), "an error without a message"),
        (line(id=1, error={"code": 1, "message": 7}), "a message that is not a string"),
        (b'{"jsonrpc": "2.0", "id": 1, "result": NaN}', "NaN"),
        (b'{"jsonrpc": "2.0", "id": 1, "result": Infinity}', "Infinity"),
    ],
)
def test_a_line_that_is_not_a_jsonrpc_message_is_a_protocol_error(raw, why):
    with pytest.raises(ProtocolError) as caught:
        decode(raw)
    assert not isinstance(caught.value, LineTooLongError), f"{why} was reported as too long"


def test_deeply_nested_json_is_a_protocol_error_not_a_crash():
    nested = b"[" * 100_000 + b"]" * 100_000
    with pytest.raises(ProtocolError, match="nest"):
        decode(nested)


def pad_to(size: int) -> bytes:
    """A valid message that is exactly ``size`` bytes long."""
    base = len(line(method="m", params={"p": ""}))
    padded = line(method="m", params={"p": "x" * (size - base)})
    assert len(padded) == size
    return padded


def method_of(raw: bytes) -> str:
    message = decode(raw)
    assert isinstance(message, Notification), message
    return message.method


def test_a_line_of_exactly_four_mebibytes_is_accepted():
    assert method_of(pad_to(MAX_LINE_BYTES)) == "m"
    # The terminator is not counted: it is what ends the line, not part of it.
    assert method_of(pad_to(MAX_LINE_BYTES) + b"\n") == "m"


def test_a_line_one_byte_over_four_mebibytes_is_a_protocol_error():
    with pytest.raises(LineTooLongError, match=str(MAX_LINE_BYTES + 1)):
        decode(pad_to(MAX_LINE_BYTES + 1))
    # A terminator does not hide an excess.
    with pytest.raises(LineTooLongError):
        decode(pad_to(MAX_LINE_BYTES + 1) + b"\n")


def test_a_too_long_line_is_a_kind_of_protocol_error():
    assert issubclass(LineTooLongError, ProtocolError)
