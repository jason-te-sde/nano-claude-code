# src/nanoclaude/mcp/client.py
"""An MCP client: find out which protocol the server speaks, list its tools, call them.

**Two eras, one way to tell.** MCP 2026-07-28 ("modern") starts with ``server/discover``;
every earlier revision ("legacy") starts with ``initialize``. Servers in the wild are mostly
legacy, so the fallback is the main path and not the exception:

1. Send ``server/discover``, with the version, capabilities and client info in ``_meta``.
2. A result means modern, and every later request carries the same ``_meta``.
3. Error -32022 means the server speaks other versions than the one named: retry with one of
   those this client speaks that it has not yet tried, or fail.
4. Any other error, a result that is not an object, or no answer in
   :data:`DISCOVER_TIMEOUT_SECONDS`, means legacy: ``initialize``, then
   ``notifications/initialized``. A reply naming a version outside :data:`LEGACY_VERSIONS`
   fails.

**What a client does not do.** It declares no capabilities, so a server has no business
asking it for anything: a request from the server is answered ``-32601``, except ``ping``;
its notifications are ignored; a result that is not ``complete`` is an error on that call.
A server's tool list is read when asked for and is not watched. No tool is named, described
or wrapped here: :class:`RemoteTool` and :class:`CallResult` hold what the server sent, and
turning that into something safe to show a model is the adapter's job.

**Failing for good.** A line over 4 MiB, or a server that has gone, fails the client: every
request that is waiting and every later one raises the same error, and the process is ended.
Anything less (a line that is not JSON, a reply to nothing, a result of the wrong shape)
costs only that line or that call.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from nanoclaude import __version__
from nanoclaude.mcp.jsonrpc import (
    METHOD_NOT_FOUND,
    UNSUPPORTED_PROTOCOL_VERSION,
    ErrorResponse,
    LineTooLongError,
    Notification,
    ProtocolError,
    Request,
    RequestId,
    RequestIds,
    Response,
    RpcError,
    decode,
    encode,
)
from nanoclaude.mcp.transport import Transport, TransportError

#: The versions this client speaks as a modern client, newest first.
MODERN_VERSIONS = ("2026-07-28",)

#: The versions it accepts from a legacy server, newest first; it offers the first.
LEGACY_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")

#: How long ``server/discover`` has to be answered before the server is taken for a legacy one.
DISCOVER_TIMEOUT_SECONDS = 5.0

#: How long any other request has to be answered, unless the caller says otherwise.
REQUEST_TIMEOUT_SECONDS = 30.0

#: The most pages of ``tools/list`` followed, and the most tools taken from them.
MAX_TOOL_PAGES = 20
MAX_TOOLS = 512

#: How long to wait for a cancellation to be written before giving up on it.
_CANCEL_SEND_SECONDS = 2.0

_META_VERSION = "io.modelcontextprotocol/protocolVersion"
_META_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
_META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"

Era = Literal["modern", "legacy"]


class McpError(Exception):
    """Anything that goes wrong between this client and a server."""


class McpConnectionError(McpError):
    """The server did not start, or has gone: its input or its output is closed."""


class McpProtocolError(McpError):
    """The server broke the protocol: a line that is too long, a result of the wrong shape,
    more pages or tools than are taken."""


class McpVersionError(McpProtocolError):
    """The server speaks no protocol version this client does."""


class McpRequestError(McpError):
    """The server answered a request with a JSON-RPC error."""

    def __init__(self, error: RpcError) -> None:
        super().__init__(f"{error.message} (code {error.code})")
        self.error = error


class McpTimeoutError(McpError):
    """A request was not answered in time."""

    def __init__(self, message: str, request_id: RequestId) -> None:
        super().__init__(message)
        self.request_id = request_id


class McpUnsupportedInteractionError(McpError):
    """The server answered with something other than a complete result, which would need a
    kind of interaction (input from the user, a task to poll) this client does not have."""

    def __init__(self, method: str, result_type: object) -> None:
        super().__init__(
            f"the server's result for {method} is {result_type!r}, not 'complete': "
            "the interaction it asks for is not supported"
        )
        self.result_type = result_type


# Plain values holding what the server sent, which includes decoded JSON (mappings and
# lists), so a hash made from the fields would only raise when called; declared unhashable
# instead, as tools/base.py does for the same reason.


@dataclass(frozen=True, slots=True)
class RemoteTool:
    """A tool as ``tools/list`` gave it. Only the name is checked; the rest is as sent."""

    name: str
    description: Any = None
    input_schema: Any = None

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class CallResult:
    """What ``tools/call`` returned, with ``content`` a list and ``is_error`` a bool."""

    content: list[Any]
    structured_content: Any = None
    is_error: bool = False

    __hash__ = None  # type: ignore[assignment]


def _client_info() -> dict[str, str]:
    return {"name": "nano-claude-code", "version": __version__}


def _object(result: object, method: str) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        raise McpProtocolError(
            f"the result of {method} must be an object, not {type(result).__name__}"
        )
    return result


def _complete(result: object, method: str) -> Mapping[str, Any]:
    """``result`` as an object, which says nothing but ``complete`` about what it is."""
    body = _object(result, method)
    result_type = body.get("resultType", "complete")
    if result_type != "complete":
        raise McpUnsupportedInteractionError(method, result_type)
    return body


class McpClient:
    """One server, over one transport. Call :meth:`start` once, and :meth:`close` when done."""

    def __init__(
        self,
        transport: Transport,
        *,
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
        discover_timeout: float = DISCOVER_TIMEOUT_SECONDS,
        modern_versions: Sequence[str] = MODERN_VERSIONS,
    ) -> None:
        self._transport = transport
        self._request_timeout = request_timeout
        self._discover_timeout = discover_timeout
        self._modern_versions = tuple(modern_versions)
        self._ids = RequestIds()
        self._pending: dict[RequestId, asyncio.Future[Response | ErrorResponse]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False
        self._failure: McpError | None = None
        self._era: Era | None = None
        self._protocol_version: str | None = None
        #: What every request carries once the server is known to be modern.
        self._meta: dict[str, Any] | None = None

    # ---- state -----------------------------------------------------------------------

    @property
    def era(self) -> Era:
        if self._era is None:
            raise McpError("the client has not been started")
        return self._era

    @property
    def protocol_version(self) -> str:
        """The version in use: the one named in ``_meta`` (modern) or the server's reply."""
        if self._protocol_version is None:
            raise McpError("the client has not been started")
        return self._protocol_version

    @property
    def failure(self) -> McpError | None:
        """Why the client can no longer be used, if it has failed; ``None`` while it is well."""
        return self._failure

    # ---- starting --------------------------------------------------------------------

    async def start(self) -> None:
        """Start the server and settle which protocol it speaks.

        On any failure the server is ended before the error is raised, and :attr:`failure`
        says why.
        """
        if self._closed:
            raise McpError("the client is closed")
        if self._started:
            raise McpError("the client was already started")
        self._started = True
        try:
            try:
                await self._transport.start()
            except TransportError as exc:
                raise McpConnectionError(str(exc)) from exc
            self._reader = asyncio.create_task(self._read_loop(), name="mcp-reader")
            if not await self._discover():
                await self._initialize()
        except BaseException as exc:
            if isinstance(exc, McpError) and self._failure is None:
                self._failure = exc
            await self.close()
            raise

    async def _discover(self) -> bool:
        """Try ``server/discover``; True if the server is modern, False if it is not."""
        tried: set[str] = set()
        version = self._modern_versions[0]
        while True:
            tried.add(version)
            meta = {
                _META_VERSION: version,
                _META_CAPABILITIES: {},
                _META_CLIENT_INFO: _client_info(),
            }
            try:
                reply = await self._exchange(
                    "server/discover", {"_meta": meta}, self._discover_timeout
                )
            except McpTimeoutError:
                return False
            if isinstance(reply, Response):
                if not isinstance(reply.result, Mapping):
                    return False
                self._era, self._protocol_version, self._meta = "modern", version, meta
                return True
            if reply.error.code != UNSUPPORTED_PROTOCOL_VERSION:
                return False
            version = self._another_version(reply.error.data, tried)

    def _another_version(self, data: object, tried: set[str]) -> str:
        """A version this client speaks, the server's -32022 names, and has not been tried."""
        supported = data.get("supported") if isinstance(data, Mapping) else None
        offered = (
            [v for v in supported if isinstance(v, str)] if isinstance(supported, list) else []
        )
        for version in self._modern_versions:
            if version in offered and version not in tried:
                return version
        raise McpVersionError(
            f"the server speaks protocol versions {offered}; this client speaks "
            f"{list(self._modern_versions)} (and {list(LEGACY_VERSIONS)} to legacy servers)"
        )

    async def _initialize(self) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": LEGACY_VERSIONS[0],
                "capabilities": {},
                "clientInfo": _client_info(),
            },
            self._request_timeout,
        )
        version = _object(result, "initialize").get("protocolVersion")
        if version not in LEGACY_VERSIONS:
            raise McpVersionError(
                f"the server's reply to initialize names protocol version {version!r}; "
                f"this client speaks {list(LEGACY_VERSIONS)}"
            )
        await self._send(Notification("notifications/initialized"))
        self._era, self._protocol_version = "legacy", version

    # ---- tools -----------------------------------------------------------------------

    async def list_tools(self) -> list[RemoteTool]:
        """Every tool the server lists, following ``nextCursor`` to the limits in the module."""
        self._require_started()
        tools: list[RemoteTool] = []
        cursor: str | None = None
        for _ in range(MAX_TOOL_PAGES):
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = await self._request(
                "tools/list", self._with_meta(params), self._request_timeout
            )
            page = _complete(result, "tools/list")
            entries = page.get("tools", [])
            if not isinstance(entries, list):
                raise McpProtocolError(
                    f"tools/list: 'tools' must be a list, not {type(entries).__name__}"
                )
            for entry in entries:
                tools.append(self._remote_tool(entry, len(tools)))
                if len(tools) > MAX_TOOLS:
                    raise McpProtocolError(f"the server lists more than {MAX_TOOLS} tools")
            cursor = page.get("nextCursor")
            if cursor is None:
                return tools
            if not isinstance(cursor, str):
                raise McpProtocolError(
                    f"tools/list: nextCursor must be a string, not {type(cursor).__name__}"
                )
        raise McpProtocolError(f"the server lists more than {MAX_TOOL_PAGES} pages of tools")

    @staticmethod
    def _remote_tool(entry: object, position: int) -> RemoteTool:
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not isinstance(entry, Mapping) or not isinstance(name, str):
            raise McpProtocolError(f"tools/list: entry {position} has no name that is a string")
        return RemoteTool(name, entry.get("description"), entry.get("inputSchema"))

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        timeout: float | None = None,  # noqa: ASYNC109 - enforced here, with asyncio.timeout
    ) -> CallResult:
        """Call a tool. If the server has not answered in ``timeout`` seconds (the client's
        default when ``None``), tell it to stop and raise :class:`McpTimeoutError`."""
        self._require_started()
        seconds = self._request_timeout if timeout is None else timeout
        params = self._with_meta({"name": name, "arguments": dict(arguments)})
        try:
            result = await self._request("tools/call", params, seconds)
        except McpTimeoutError as exc:
            await self._cancel(exc.request_id, seconds)
            raise
        body = _complete(result, "tools/call")
        content = body.get("content", [])
        if not isinstance(content, list):
            raise McpProtocolError(
                f"tools/call: 'content' must be a list, not {type(content).__name__}"
            )
        is_error = body.get("isError", False)
        if not isinstance(is_error, bool):
            raise McpProtocolError(
                f"tools/call: 'isError' must be a boolean, not {type(is_error).__name__}"
            )
        return CallResult(content, body.get("structuredContent"), is_error)

    async def _cancel(self, request_id: RequestId, seconds: float) -> None:
        """Tell the server to stop working on a request nobody is waiting for. Best effort:
        the caller wants its timeout, not an error about the cancellation."""
        notice = Notification(
            "notifications/cancelled",
            {"requestId": request_id, "reason": f"no answer within {seconds:g} seconds"},
        )
        with contextlib.suppress(McpError, TimeoutError):
            async with asyncio.timeout(_CANCEL_SEND_SECONDS):
                await self._send(notice)

    # ---- requests --------------------------------------------------------------------

    def _require_started(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise McpError("the client is closed")
        if self._era is None:
            raise McpError("the client has not been started")

    def _with_meta(self, params: dict[str, Any]) -> dict[str, Any]:
        return params if self._meta is None else {**params, "_meta": self._meta}

    async def _send(self, message: Request | Notification | Response | ErrorResponse) -> None:
        try:
            line = encode(message)
        except ProtocolError as exc:
            raise McpProtocolError(str(exc)) from exc
        try:
            await self._transport.send(line)
        except TransportError as exc:
            raise McpConnectionError(str(exc)) from exc

    async def _exchange(
        self, method: str, params: dict[str, Any], seconds: float
    ) -> Response | ErrorResponse:
        """Send a request and wait for whatever answers it, a JSON-RPC error included."""
        if self._failure is not None:
            raise self._failure
        request_id = self._ids.next()
        future: asyncio.Future[Response | ErrorResponse] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        try:
            try:
                async with asyncio.timeout(seconds):
                    await self._send(Request(request_id, method, params))
                    return await future
            except TimeoutError:
                raise McpTimeoutError(
                    f"{method} was not answered within {seconds:g} seconds", request_id
                ) from None
        finally:
            self._pending.pop(request_id, None)

    async def _request(self, method: str, params: dict[str, Any], seconds: float) -> Any:
        """The result of a request; a JSON-RPC error is raised as :class:`McpRequestError`."""
        reply = await self._exchange(method, params, seconds)
        if isinstance(reply, ErrorResponse):
            raise McpRequestError(reply.error)
        return reply.result

    # ---- what the server sends -------------------------------------------------------

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    line = await self._transport.receive()
                    message = None if line is None else decode(line)
                except LineTooLongError as exc:
                    await self._fail(McpProtocolError(f"{exc}; the server is ended"))
                    return
                except TransportError as exc:
                    await self._fail(McpConnectionError(str(exc)))
                    return
                except ProtocolError:
                    continue  # not a message: skip the line, which costs nothing else
                if message is None:
                    await self._fail(
                        McpConnectionError("the server closed its output: it has exited or crashed")
                    )
                    return
                await self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a failure of our own must not leave requests waiting
            await self._fail(
                McpConnectionError(f"the reader of the server's output failed: {exc!r}")
            )

    async def _dispatch(self, message: Request | Notification | Response | ErrorResponse) -> None:
        if isinstance(message, Response | ErrorResponse):
            future = self._pending.get(message.id) if message.id is not None else None
            if future is not None and not future.done():
                future.set_result(message)
            # Otherwise it answers nothing we are waiting for (a call that timed out, an id
            # nobody sent): ignored.
        elif isinstance(message, Request):
            await self._refuse_or_ping(message)
        # A notification is ignored: this client acts on none of them.

    async def _refuse_or_ping(self, request: Request) -> None:
        reply: Response | ErrorResponse
        if request.method == "ping":
            reply = Response(request.id, {})
        else:
            error = RpcError(METHOD_NOT_FOUND, f"Method not found: {request.method}")
            reply = ErrorResponse(request.id, error)
        with contextlib.suppress(McpError):  # the server may be going; its reader says so
            await self._send(reply)

    async def _fail(self, error: McpError) -> None:
        """Fail the client for good: wake what is waiting with ``error``, and end the server."""
        if self._failure is None:
            self._failure = error
        for future in self._pending.values():
            if not future.done():
                future.set_exception(self._failure)
        await self._transport.close()

    # ---- closing ---------------------------------------------------------------------

    async def close(self) -> None:
        """End the server and everything waiting on it. Safe to call more than once."""
        self._closed = True
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.cancel()
            await asyncio.wait({reader})
        gone = self._failure or McpError("the client is closed")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(gone)
        await self._transport.close()
