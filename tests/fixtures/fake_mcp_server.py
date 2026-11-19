#!/usr/bin/env python3
# tests/fixtures/fake_mcp_server.py
"""A stdio MCP server that behaves however the command line says, for the tests.

Usage: ``fake_mcp_server.py MODE [KEY=VALUE ...]``

``MODE`` names a row of :data:`MODES`, a bundle of settings laid over :data:`DEFAULTS`; each
``KEY=VALUE`` after it changes one setting (the value is read as JSON when it is JSON, and
as text when it is not). A test that needs a server nobody has described yet adds a row, or
overrides a setting for one run, and the server's code does not change: every behaviour is a
setting, and every setting is read in one place.

It speaks only the standard library, so it runs under any interpreter the tests do, and it
reads and answers one message at a time, which is what makes a mode like ``slow`` (never
answers a call, keeps reading) and ``call_delay`` (answers late) possible to state.

What the tests learn from it, they learn from stderr, which the client's transport appends to
a log file: every message the server receives is written there as ``RECV <json>``, and the
moments the lifecycle tests care about as ``EVENT <name> <time>``. A test closes the client
first and reads the log after, when the transport has drained it.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

MAX_LINE_BYTES = 4 * 1024 * 1024
MODERN_VERSION = "2026-07-28"
VERSION_KEY = "io.modelcontextprotocol/protocolVersion"

METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
UNSUPPORTED_PROTOCOL_VERSION = -32022

Config = dict[str, Any]

#: Every setting, with the value a mode gets unless it says otherwise.
DEFAULTS: Config = {
    # How `server/discover` is answered: ok (a DiscoverResult, or -32022 for a version not in
    # `supported`), unsupported (-32022 whatever the version), method-not-found, silent (no
    # answer), late (a result after `discover_delay` seconds, from a server that stays
    # legacy), error (code `discover_code`) or scalar (a result that is not an object).
    "discover": "ok",
    "discover_delay": 0.5,
    "discover_code": -32603,
    "supported": [MODERN_VERSION],
    # How `initialize` is answered: ok (naming `legacy_version`), method-not-found or silent.
    "initialize": "ok",
    "legacy_version": "2025-11-25",
    # `tools/list`: `pages` pages of `per_page` tools, linked by cursors.
    "pages": 1,
    "per_page": 1,
    # A `nextCursor` that is not a string, on the first page, when set to a number.
    "bad_cursor": None,
    # The `resultType` the results carry; None leaves the field out.
    "list_result_type": None,
    "call_result_type": None,
    # The method whose result is a list and not an object: initialize, tools/list or tools/call.
    "non_object_result": "",
    # Fields merged into the result of `tools/list` and of an echoed `tools/call`, to make
    # them malformed.
    "list_patch": {},
    "call_patch": {},
    # The first tool has no name, when true.
    "tool_without_name": False,
    # Lines that are not messages (text, an empty line, a batch) before every message.
    "stdout_noise": False,
    # Every result and error is written twice.
    "duplicate_replies": False,
    # `tools/call`: echo, never (no answer), error (a JSON-RPC error), is-error (a result
    # with isError) or big (an answer of exactly `line_size` bytes).
    "call": "echo",
    "line_size": MAX_LINE_BYTES + 1,
    "call_delay": 0.0,
    # Leave as soon as the handshake is done, or before reading a byte.
    "exit_after_handshake": False,
    "exit_on_start": False,
    # Once the handshake is done: requests of these methods and a few notifications.
    "server_requests": [],
    "server_notifications": False,
    # What SIGTERM does (default, ignore, or log-exit), what the end of stdin does (exit or
    # linger), and whether a helper process is left in the group (none or child).
    "on_term": "default",
    "on_stdin_eof": "exit",
    "helper": "none",
}

_MODERN: Config = {
    "initialize": "method-not-found",
    "list_result_type": "complete",
    "call_result_type": "complete",
}
_LEGACY: Config = {"discover": "method-not-found"}
_STUBBORN: Config = {"on_term": "ignore", "on_stdin_eof": "linger", "helper": "child"}

MODES: dict[str, Config] = {
    "modern": _MODERN,
    "legacy": _LEGACY,
    "dual": {**_MODERN, "initialize": "ok"},
    "unsupported-version": {"supported": ["2099-01-01"]},
    "silent-discover": {**_LEGACY, "discover": "silent"},
    "slow": {**_MODERN, "call": "never"},
    "crash": {**_MODERN, "exit_after_handshake": True},
    "crash-on-start": {"exit_on_start": True},
    "oversized-line": {**_MODERN, "call": "big"},
    "paginated": {**_MODERN, "pages": 3, "per_page": 2},
    "server-request": {
        **_LEGACY,
        "server_requests": ["sampling/createMessage", "roots/list", "elicitation/create", "ping"],
        "server_notifications": True,
    },
    "is-error": {**_MODERN, "call": "is-error"},
    "no-result-type": {**_MODERN, "list_result_type": None, "call_result_type": None},
    "input-required": {**_MODERN, "call_result_type": "input_required"},
    "rpc-error": {**_MODERN, "call": "error"},
    # Ends at the end of stdin, and leaves a helper behind that nothing but SIGKILL stops.
    "orphan-helper": {**_MODERN, "on_term": "ignore", "helper": "child"},
    # Does not end at the end of stdin, and says when SIGTERM arrives.
    "ignore-stdin-eof": {**_MODERN, "on_term": "log-exit", "on_stdin_eof": "linger"},
    # Ignores the end of stdin and SIGTERM, and has a helper that does too.
    "ignore-term": {**_MODERN, **_STUBBORN},
}


def parse_settings(argv: list[str]) -> Config:
    """``DEFAULTS``, then the mode's row, then each ``KEY=VALUE`` of the rest."""
    if not argv:
        raise SystemExit(f"usage: fake_mcp_server.py MODE [KEY=VALUE ...]; modes: {sorted(MODES)}")
    mode, overrides = argv[0], argv[1:]
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; modes: {sorted(MODES)}")
    settings = {**DEFAULTS, **MODES[mode]}
    for item in overrides:
        key, separator, text = item.partition("=")
        if not separator or key not in DEFAULTS:
            raise SystemExit(f"not a setting: {item!r}; settings: {sorted(DEFAULTS)}")
        try:
            settings[key] = json.loads(text)
        except ValueError:
            settings[key] = text
    return settings


def log(tag: str, payload: object) -> None:
    print(f"{tag} {payload}", file=sys.stderr, flush=True)


def event(name: str) -> None:
    log("EVENT", f"{name} {time.time():.6f}")


class Server:
    def __init__(self, settings: Config) -> None:
        self.settings = settings
        #: The version the client named, once a modern handshake has been answered.
        self.modern_version: str | None = None
        self.handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "server/discover": self.on_discover,
            "initialize": self.on_initialize,
            "tools/list": self.on_tools_list,
            "tools/call": self.on_tools_call,
            "ping": lambda request: self.reply(request, {}),
        }
        self.notifications: dict[str, Callable[[], None]] = {
            "notifications/initialized": self.handshake_done,
        }

    # ---- wire ------------------------------------------------------------------------

    @staticmethod
    def write(line: bytes) -> None:
        sys.stdout.buffer.write(line + b"\n")
        sys.stdout.buffer.flush()

    def send(self, message: dict[str, Any]) -> None:
        if self.settings["stdout_noise"]:
            for noise in (b"starting up: this line is not JSON", b"", b"[1, 2]"):
                self.write(noise)
        line = json.dumps({"jsonrpc": "2.0", **message}, separators=(",", ":")).encode()
        self.write(line)
        if self.settings["duplicate_replies"] and "method" not in message:
            self.write(line)

    def reply(self, request: dict[str, Any], result: Any) -> None:
        if request["method"] == self.settings["non_object_result"]:
            result = []
        self.send({"id": request["id"], "result": result})

    def fail(self, request: dict[str, Any], code: int, message: str, data: Any = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        self.send({"id": request["id"], "error": error})

    # ---- lifecycle -------------------------------------------------------------------

    def start(self) -> None:
        settings = self.settings
        if settings["exit_on_start"]:
            log("EVENT", "exiting-on-start")
            raise SystemExit(3)
        if settings["on_term"] != "default":
            signal.signal(signal.SIGTERM, self.on_sigterm)
        if settings["helper"] == "child":
            self.leave_helper()

    def on_sigterm(self, _signum: int, _frame: object) -> None:
        if self.settings["on_term"] == "ignore":
            event("sigterm-ignored")
            return
        event("sigterm")
        os._exit(0)

    @staticmethod
    def leave_helper() -> None:
        pid = os.fork()
        if pid == 0:
            # The helper stays in the server's process group, ignores what would end it, and
            # keeps the inherited pipes open, as a worker a real server spawned would.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            while True:
                time.sleep(1)
        log("HELPER", pid)

    def run(self) -> None:
        self.start()
        for raw in iter(sys.stdin.buffer.readline, b""):
            self.handle(raw)
        event("stdin-eof")
        while self.settings["on_stdin_eof"] == "linger":
            time.sleep(0.05)

    def handle(self, raw: bytes) -> None:
        try:
            message = json.loads(raw)
        except ValueError:
            log("BAD", raw[:200])
            return
        log("RECV", json.dumps(message, separators=(",", ":")))
        method = message.get("method")
        if method is None:
            return
        if "id" not in message:
            handler = self.notifications.get(method)
            if handler is not None:
                handler()
            return
        request_handler = self.handlers.get(method)
        if request_handler is None:
            self.fail(message, METHOD_NOT_FOUND, f"Method not found: {method}")
        else:
            request_handler(message)

    def handshake_done(self) -> None:
        settings = self.settings
        if settings["exit_after_handshake"]:
            event("exiting-after-handshake")
            raise SystemExit(3)
        if settings["server_notifications"]:
            for method in ("notifications/message", "notifications/tools/list_changed"):
                self.send({"method": method, "params": {}})
        for number, method in enumerate(settings["server_requests"]):
            self.send({"id": f"srv-{number}", "method": method, "params": {}})

    # ---- methods ---------------------------------------------------------------------

    def meta_is_valid(self, request: dict[str, Any]) -> bool:
        """After a modern handshake every request must carry the version it was made with."""
        if self.modern_version is None:
            return True
        meta = (request.get("params") or {}).get("_meta")
        if isinstance(meta, dict) and meta.get(VERSION_KEY) == self.modern_version:
            return True
        self.fail(request, INVALID_PARAMS, "missing or wrong _meta")
        return False

    def on_discover(self, request: dict[str, Any]) -> None:
        settings = self.settings
        answer = settings["discover"]
        if answer == "silent":
            return
        if answer == "late":
            # A server too slow to be waited for: it answers, but it is not a modern server.
            time.sleep(settings["discover_delay"])
            self.reply(request, self.discover_result())
        elif answer == "unsupported":
            self.fail(
                request,
                UNSUPPORTED_PROTOCOL_VERSION,
                "unsupported protocol version",
                {"supported": settings["supported"]},
            )
        elif answer == "method-not-found":
            self.fail(request, METHOD_NOT_FOUND, "Method not found: server/discover")
        elif answer == "error":
            self.fail(request, settings["discover_code"], "discover failed")
        elif answer == "scalar":
            self.send({"id": request["id"], "result": "not an object"})
        else:
            meta = (request.get("params") or {}).get("_meta") or {}
            version = meta.get(VERSION_KEY)
            if version not in settings["supported"]:
                self.fail(
                    request,
                    UNSUPPORTED_PROTOCOL_VERSION,
                    f"unsupported protocol version {version}",
                    {"supported": settings["supported"]},
                )
                return
            self.modern_version = version
            self.reply(request, self.discover_result())
            self.handshake_done()

    def discover_result(self) -> dict[str, Any]:
        return {
            "supportedVersions": self.settings["supported"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp-server", "version": "0"},
        }

    def on_initialize(self, request: dict[str, Any]) -> None:
        answer = self.settings["initialize"]
        if answer == "silent":
            return
        if answer == "method-not-found":
            self.fail(request, METHOD_NOT_FOUND, "Method not found: initialize")
            return
        self.reply(
            request,
            {
                "protocolVersion": self.settings["legacy_version"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp-server", "version": "0"},
            },
        )

    def with_result_type(self, result: dict[str, Any], key: str) -> dict[str, Any]:
        result_type = self.settings[key]
        return result if result_type is None else {**result, "resultType": result_type}

    def tool(self, number: int) -> dict[str, Any]:
        tool = {
            "name": "echo" if number == 0 else f"tool_{number}",
            "description": f"Fake tool number {number}.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
        if number == 0 and self.settings["tool_without_name"]:
            del tool["name"]
        return tool

    def on_tools_list(self, request: dict[str, Any]) -> None:
        if not self.meta_is_valid(request):
            return
        settings = self.settings
        cursor = (request.get("params") or {}).get("cursor")
        page = 0 if cursor is None else int(str(cursor).removeprefix("page-"))
        per_page = settings["per_page"]
        result: dict[str, Any] = {
            "tools": [self.tool(page * per_page + n) for n in range(per_page)],
            **settings["list_patch"],
        }
        if page + 1 < settings["pages"]:
            result["nextCursor"] = f"page-{page + 1}"
        if page == 0 and settings["bad_cursor"] is not None:
            result["nextCursor"] = settings["bad_cursor"]
        self.reply(request, self.with_result_type(result, "list_result_type"))

    def on_tools_call(self, request: dict[str, Any]) -> None:
        if not self.meta_is_valid(request):
            return
        settings = self.settings
        what = settings["call"]
        if what == "never":
            return
        time.sleep(settings["call_delay"])
        params = request.get("params") or {}
        if what == "error":
            self.fail(request, -32000, "the tool exploded", {"detail": "fake"})
        elif what == "is-error":
            self.reply(request, {"content": [{"type": "text", "text": "boom"}], "isError": True})
        elif what == "big":
            self.write_exact(request, settings["line_size"])
        else:
            arguments = params.get("arguments")
            result = {
                "content": [{"type": "text", "text": f"{params.get('name')}:{arguments}"}],
                "structuredContent": {"arguments": arguments},
                **settings["call_patch"],
            }
            self.reply(request, self.with_result_type(result, "call_result_type"))

    def write_exact(self, request: dict[str, Any], size: int) -> None:
        """Answer with a valid result in a line of exactly ``size`` bytes."""

        def line(text: str) -> bytes:
            message = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"content": [{"type": "text", "text": text}]},
            }
            return json.dumps(message, separators=(",", ":")).encode()

        padding = size - len(line(""))
        self.write(line("x" * padding))


def main() -> None:
    Server(parse_settings(sys.argv[1:])).run()


if __name__ == "__main__":
    main()
