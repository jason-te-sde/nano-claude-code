# src/nanoclaude/mcp/jsonrpc.py
"""JSON-RPC 2.0 messages, one to a line.

The MCP stdio transport frames a message as one line of UTF-8 JSON, ended by a newline.
:func:`encode` makes that line and :func:`decode` takes one apart; neither does any I/O, so
what counts as a message is decided here and nowhere else.

Three things are held by the code and not left to the caller:

* **A request id is never null and is never one that is still waiting for its answer.**
  :class:`RequestIds` counts up from 1 and never goes back, so an id is not handed out twice
  in a session, and :func:`encode` refuses an id that is not an integer or a string, so a
  message made by hand cannot carry a null one either.
* **A line over 4 MiB is an error** (:class:`LineTooLongError`), in both directions. A server
  that sends one is not slow, it is broken or hostile, and the caller ends it.
* **A line that is not a message is an error that costs only that line.** Text on stdout that
  is not JSON, a batch, an object that is neither a request, a notification nor a response:
  each is a :class:`ProtocolError`, which a caller may skip. Only the length is fatal.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

#: The longest line, in bytes without its terminator, either side will write or read.
MAX_LINE_BYTES = 4 * 1024 * 1024

#: JSON-RPC's own code for a method the receiver does not have.
METHOD_NOT_FOUND = -32601

#: MCP 2026-07-28's code for a server that does not speak the version the client named.
UNSUPPORTED_PROTOCOL_VERSION = -32022

#: What may identify a request: JSON-RPC allows a number or a string, and never null.
RequestId: TypeAlias = int | str


class ProtocolError(Exception):
    """A line is not a JSON-RPC message, or a message cannot be written as one."""


class LineTooLongError(ProtocolError):
    """A line is over :data:`MAX_LINE_BYTES`."""


# The messages hold decoded JSON (a mapping, a list), which is unhashable, so a hash made
# from their fields would only raise when somebody called it. Declared unhashable instead,
# which is what ``isinstance(x, Hashable)`` then says.


@dataclass(frozen=True, slots=True)
class RpcError:
    """The error object of a failed response."""

    code: int
    message: str
    data: Any = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class Request:
    id: RequestId
    method: str
    params: Mapping[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class Notification:
    method: str
    params: Mapping[str, Any] | None = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class Response:
    id: RequestId
    result: Any

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    #: ``None`` only for an error that answers no request: a peer that could not read the
    #: request's id at all says so with a null.
    id: RequestId | None
    error: RpcError

    __hash__ = None  # type: ignore[assignment]


Message: TypeAlias = Request | Notification | Response | ErrorResponse


class RequestIds:
    """The ids of one session's requests: integers from 1, each handed out once."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def next(self) -> int:
        return next(self._counter)


def _valid_id(value: object) -> bool:
    # bool is an int in Python and not a JSON-RPC id.
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _wire(message: Message) -> dict[str, Any]:
    match message:
        case Request(id=request_id, method=method, params=params):
            wire: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                wire["params"] = params
        case Notification(method=method, params=params):
            wire = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                wire["params"] = params
        case Response(id=request_id, result=result):
            wire = {"jsonrpc": "2.0", "id": request_id, "result": result}
        case ErrorResponse(id=request_id, error=error):
            body: dict[str, Any] = {"code": error.code, "message": error.message}
            if error.data is not None:
                body["data"] = error.data
            wire = {"jsonrpc": "2.0", "id": request_id, "error": body}
    return wire


def encode(message: Message) -> bytes:
    """``message`` as one line, without its terminator.

    The text is written with ASCII escapes, so that a newline, a U+2028 or a lone surrogate
    in an argument can neither split the line nor fail to encode.
    """
    if isinstance(message, Request | Response | ErrorResponse):
        request_id = message.id
        # Only an error that answers nothing in particular has no id to give.
        null_ok = isinstance(message, ErrorResponse) and request_id is None
        if not null_ok and not _valid_id(request_id):
            raise ProtocolError(
                f"a message id must be an integer or a string, not {request_id!r} "
                f"({type(request_id).__name__})"
            )
    try:
        line = json.dumps(
            _wire(message), separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProtocolError(f"cannot encode the message as JSON: {exc}") from exc
    if len(line) > MAX_LINE_BYTES:
        raise LineTooLongError(
            f"the message is {len(line)} bytes, over the limit of {MAX_LINE_BYTES}"
        )
    return line


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _error_object(raw: object) -> RpcError:
    if not isinstance(raw, Mapping):
        raise ProtocolError(f"an error must be an object, not {type(raw).__name__}")
    code, message = raw.get("code"), raw.get("message")
    if not isinstance(code, int) or isinstance(code, bool):
        raise ProtocolError(f"an error code must be an integer, not {code!r}")
    if not isinstance(message, str):
        raise ProtocolError(f"an error message must be a string, not {message!r}")
    return RpcError(code, message, raw.get("data"))


def decode(line: bytes) -> Message:
    """The message on ``line``, which may still carry its terminator.

    Raises :class:`LineTooLongError` for a line over the limit and :class:`ProtocolError` for
    anything else that is not a message.
    """
    if line.endswith(b"\n"):
        line = line[:-1]
    if len(line) > MAX_LINE_BYTES:
        raise LineTooLongError(f"a line of {len(line)} bytes is over the limit of {MAX_LINE_BYTES}")
    try:
        raw = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"the line is not UTF-8: {exc}") from exc
    except RecursionError as exc:
        raise ProtocolError("the line nests JSON too deeply to read") from exc
    except ValueError as exc:
        raise ProtocolError(f"the line is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        kind = "a batch" if isinstance(raw, list) else type(raw).__name__
        raise ProtocolError(f"a message must be a JSON object, not {kind}")
    if raw.get("jsonrpc") != "2.0":
        raise ProtocolError(f'"jsonrpc" must be "2.0", not {raw.get("jsonrpc")!r}')
    if "method" in raw:
        return _decode_call(raw)
    return _decode_reply(raw)


def _decode_call(raw: dict[str, Any]) -> Request | Notification:
    if "result" in raw or "error" in raw:
        raise ProtocolError("a message with a method cannot also carry a result or an error")
    method, params = raw["method"], raw.get("params")
    if not isinstance(method, str):
        raise ProtocolError(f"a method must be a string, not {method!r}")
    if params is not None and not isinstance(params, dict):
        raise ProtocolError(f"params must be an object, not {type(params).__name__}")
    if "id" not in raw:
        return Notification(method, params)
    if not _valid_id(raw["id"]):
        raise ProtocolError(f"a request id must be an integer or a string, not {raw['id']!r}")
    return Request(raw["id"], method, params)


def _decode_reply(raw: dict[str, Any]) -> Response | ErrorResponse:
    has_result, has_error = "result" in raw, "error" in raw
    if has_result == has_error:
        raise ProtocolError("a reply must carry exactly one of a result and an error")
    if "id" not in raw:
        raise ProtocolError("a reply must carry the id of the request it answers")
    request_id = raw["id"]
    if has_error:
        if request_id is not None and not _valid_id(request_id):
            raise ProtocolError(
                f"a reply id must be an integer, a string or null, not {request_id!r}"
            )
        return ErrorResponse(request_id, _error_object(raw["error"]))
    if not _valid_id(request_id):
        raise ProtocolError(
            f"the id of a result must be an integer or a string, not {request_id!r}"
        )
    return Response(request_id, raw["result"])
