# src/nanoclaude/mcp/servers.py
"""The MCP servers of one session: started together, kept as one set, ended together.

A session that has servers configured starts every one that is ``enabled`` at the same time
and waits for all of them, at most :data:`START_BUDGET_SECONDS` in all. What a server that does
not make it costs is one warning line and nothing else: its tools are not registered, the
others' are, and the session goes on. A server fails to start for any of these reasons, and
each is the same to the session:

* the program cannot be started, or it exits, or it breaks the protocol (the client's
  errors);
* it speaks no protocol version this client does, which is found out when its era is;
* its tool list cannot be had;
* the pins file is unreadable (:class:`~nanoclaude.mcp.pins.PinStoreError`). That fails
  every enabled server, closed: a tool that cannot be checked against what the person last
  approved is not registered, and no server is started whose tools would be thrown away;
* it did not finish within the budget.

**The budget includes the time to end a server.** A start that is still going when its time
is up is cancelled, and a cancelled start ends its server (a close can take
:data:`~nanoclaude.mcp.transport.STDIN_GRACE_SECONDS` and
:data:`~nanoclaude.mcp.transport.TERM_GRACE_SECONDS` before it is a kill), so waiting
for starts is not allowed the whole budget. The last :data:`CLOSE_RESERVE_SECONDS` of it are
kept for the closes; a close that is still going when the budget ends is cut short, the group
killed outright, and the last half second is for the pipes of what was killed. Nothing here
waits past the budget, and nothing that was started is left running.

The tools of all the servers are one set before any is registered
(:func:`~nanoclaude.mcp.tool.combine_tools`): a name that two servers come to is dropped from
both, with a warning, and never raises.

What a server says about why it failed is its own text and goes to a terminal, so every reason
is one short line with nothing in it a terminal acts on.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nanoclaude.config.load import CONFIG_DIRNAME
from nanoclaude.config.schema import Config, McpServerConfig
from nanoclaude.mcp.client import Era, McpClient, McpError, McpRequestError
from nanoclaude.mcp.pins import PinStore, PinStoreError, pins_path
from nanoclaude.mcp.tool import AdaptedTools, McpTool, adapt_tools, combine_tools
from nanoclaude.mcp.transport import StdioTransport
from nanoclaude.tools.base import sanitize, without_lone_surrogates

#: The most the servers of a session are given, all together, to start and, when one does not,
#: to be ended.
START_BUDGET_SECONDS = 15.0

#: What of the budget is kept for ending the servers whose start is given up on: more than a
#: close that waits out both of its periods needs.
CLOSE_RESERVE_SECONDS = 5.0

#: The end of the budget is for the pipes of a server that was killed, which is quick.
_KILL_TIDY_SECONDS = 0.5

#: The longest a reason is, and the longest one the person's own pins-file message may be: it
#: names a path and says what to do about it.
MAX_REASON_CHARS = 200
_MAX_PINS_REASON_CHARS = 400

#: Where the stderr of the servers is kept, under ncc's own directory.
LOG_DIRNAME = "logs"

State = Literal["running", "failed", "disabled"]


def short_reason(text: str, limit: int = MAX_REASON_CHARS) -> str:
    """``text`` as one line of at most ``limit`` characters that no terminal acts on.

    Controls and escape sequences are removed, every run of white space, a line break
    included, is one space, and what is over the limit is cut and marked. The reason a
    server failed can hold what the server wrote, and it is printed.
    """
    line = " ".join(without_lone_surrogates(sanitize(text)).split())
    if not line:
        return "no reason was given"
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _reason_of(error: BaseException) -> str:
    """Why ``error`` stopped a server from starting, or from going on, in a short line."""
    if isinstance(error, PinStoreError):
        return short_reason(str(error), _MAX_PINS_REASON_CHARS)
    if isinstance(error, McpRequestError):
        return short_reason(f"the server answered with an error: {error}")
    if isinstance(error, McpError):
        return short_reason(str(error))
    # A bug, and not something a server did: the class says where to look, and what the
    # exception holds is not guessed at.
    return f"unexpected error ({type(error).__name__})"


def failed_line(name: str, reason: str) -> str:
    """The one warning a server that failed to start adds: which server, and why."""
    return f"MCP server '{name}' failed to start ({reason}) — its tools are not registered"


@dataclass(frozen=True, slots=True)
class ServerReport:
    """What ``/mcp`` says about one configured server.

    ``reason`` is why it failed, and is None otherwise. ``era`` is the protocol era it
    settled on, None for a server that never got that far. ``tools`` are the ones registered,
    which a server that failed later keeps: they are in the registry still, and a call to one
    is an error that says so.
    """

    name: str
    state: State
    reason: str | None
    era: Era | None
    tools: tuple[McpTool, ...]


@dataclass(slots=True)
class _Entry:
    config: McpServerConfig
    client: McpClient | None = None
    transport: StdioTransport | None = None
    tools: tuple[McpTool, ...] = ()
    #: Why it did not start; None for one that did, and for one that is disabled.
    failure: str | None = None


@dataclass(slots=True)
class _Started:
    client: McpClient
    transport: StdioTransport
    adapted: AdaptedTools


@dataclass(slots=True)
class _Failed:
    reason: str


class McpServers:
    """The servers a session started: their tools, the warnings, and a way to end them all."""

    def __init__(
        self,
        entries: Mapping[str, _Entry],
        tools: Sequence[McpTool],
        warnings: Sequence[str],
    ) -> None:
        self._entries = dict(entries)
        self._tools = tuple(tools)
        self._warnings = tuple(warnings)

    @property
    def tools(self) -> tuple[McpTool, ...]:
        """Every tool to register, from every server, with no name held twice."""
        return self._tools

    @property
    def warnings(self) -> tuple[str, ...]:
        """One line for each server that failed to start, and one for each tool left out."""
        return self._warnings

    def reports(self) -> tuple[ServerReport, ...]:
        """Every configured server, in the order the configuration gives them."""
        return tuple(self._report(entry) for entry in self._entries.values())

    def report(self, name: str) -> ServerReport | None:
        entry = self._entries.get(name)
        return None if entry is None else self._report(entry)

    @staticmethod
    def _report(entry: _Entry) -> ServerReport:
        name = entry.config.name
        if not entry.config.enabled:
            return ServerReport(name, "disabled", None, None, ())
        if entry.failure is not None or entry.client is None:
            return ServerReport(name, "failed", entry.failure or "it was not started", None, ())
        failure = entry.client.failure
        if failure is not None:
            return ServerReport(name, "failed", _reason_of(failure), entry.client.era, entry.tools)
        return ServerReport(name, "running", None, entry.client.era, entry.tools)

    async def aclose(self) -> None:
        """End every server, all at once. Safe to call again.

        Each is ended as :meth:`~nanoclaude.mcp.client.McpClient.close` ends one. If this is
        itself cancelled, the groups are killed before the cancellation goes on, so that a
        second Ctrl+C leaves nothing running either.
        """
        entries = [entry for entry in self._entries.values() if entry.client is not None]
        try:
            # A failure of one close does not stop the others, which have all run when this
            # returns: nothing here is raised for it, and the server is gone either way.
            await asyncio.gather(
                *(entry.client.close() for entry in entries if entry.client is not None),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            for entry in entries:
                if entry.transport is not None:
                    entry.transport.kill()
            raise


async def _start_one(
    server: McpServerConfig,
    *,
    root: str,
    log_dir: Path,
    pins: PinStore,
    cap: int,
    transport_options: Mapping[str, Any],
    transports: dict[str, StdioTransport],
) -> _Started | _Failed:
    """Start one server and adapt its tools; what went wrong is the answer, not an exception.

    The server is ended on every way out but success, a cancellation included, which is how a
    start that is given up on closes its server. The transport is put in ``transports`` before
    anything is started, so that whoever is waiting can kill it if this is not ending.
    """
    transport = StdioTransport(
        server.command,
        server.args,
        cwd=root,
        log_path=log_dir / f"mcp-{server.name}.log",
        env=server.env,
        **transport_options,
    )
    client = McpClient(transport, request_timeout=float(server.timeout))
    transports[server.name] = transport
    try:
        await client.start()
        adapted = await adapt_tools(client, server, pins, output_cap=cap)
    except asyncio.CancelledError:
        await client.close()
        raise
    except Exception as exc:
        await client.close()
        return _Failed(_reason_of(exc))
    return _Started(client, transport, adapted)


@dataclass(slots=True)
class _Starts:
    """The starts of a session, by server name, and the transport each one is using."""

    tasks: dict[str, asyncio.Task[_Started | _Failed]] = field(default_factory=dict)
    transports: dict[str, StdioTransport] = field(default_factory=dict)

    def unfinished(self) -> set[asyncio.Task[_Started | _Failed]]:
        return {task for task in self.tasks.values() if not task.done()}

    def kill_unfinished(self) -> list[StdioTransport]:
        """Kill the group of every start that is still going, and return the transports."""
        killed = []
        for name, task in self.tasks.items():
            transport = self.transports.get(name)
            if not task.done() and transport is not None:
                transport.kill()
                killed.append(transport)
        return killed


async def _wait_for(starts: _Starts, budget: float, close_reserve: float) -> None:
    """Wait for the starts, within ``budget`` seconds in all, as the module docstring says."""
    loop = asyncio.get_running_loop()
    end = loop.time() + budget
    start_period = max(budget - close_reserve, 0.0)
    pending = starts.unfinished()
    if pending:
        _, pending = await asyncio.wait(pending, timeout=start_period)
    if not pending:
        return
    # Time is up for these. Cancelling a start ends its server, in the order close() has.
    for task in pending:
        task.cancel()
    close_period = max(end - _KILL_TIDY_SECONDS - loop.time(), 0.0)
    _, pending = await asyncio.wait(pending, timeout=close_period)
    if not pending:
        return
    # Out of time for ending them in order. SIGKILL cannot be ignored, and is sent from
    # here and not left to the close that was cancelled, which may not have got as far.
    killed = starts.kill_unfinished()
    for task in pending:
        task.cancel()
    # A task that was cancelled a moment ago has not yet had the cancellation delivered, and
    # this one adds nothing to it: it ends its server in order, which is quick now that the
    # group is dead. Waiting is held to what is left of the budget all the same.
    await asyncio.wait(pending, timeout=max(end - loop.time(), 0.0))
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(max(end - loop.time(), 0.0)):
            await asyncio.gather(*(t.close() for t in killed), return_exceptions=True)


async def _abandon(starts: _Starts) -> None:
    """End every server of a start that is itself being cancelled: Ctrl+C while they start.

    A start that has not finished is cancelled, which ends its server in order, and a server
    that did start is closed the same way, all at once. A second cancellation, in the middle
    of that, kills every group before it goes on. The wait is ``asyncio.wait`` and not
    ``gather`` for that reason: ``gather`` hands a cancellation to what it waits for and then
    waits for it to finish, and a start that has already had its cancellation, and is now
    closing its server in order, would finish in ten seconds and not at once.
    """
    pending = starts.unfinished()
    started = [
        outcome.client
        for task in starts.tasks.values()
        if task.done() and not task.cancelled() and isinstance(outcome := task.result(), _Started)
    ]
    for task in pending:
        task.cancel()
    closing = {asyncio.ensure_future(client.close()) for client in started}
    waiting = {*pending, *closing}
    try:
        if waiting:
            await asyncio.wait(waiting)
    except asyncio.CancelledError:  # only the wait above can raise it, so ``waiting`` is not empty
        for transport in starts.transports.values():
            transport.kill()
        # What is left is ending quickly now that its group is dead; give it a moment, so that
        # nothing is still closing a pipe when this goes on.
        await asyncio.wait(waiting, timeout=_KILL_TIDY_SECONDS)
        raise


async def start_servers(
    config: Config,
    *,
    root: str,
    home: Path,
    budget: float = START_BUDGET_SECONDS,
    close_reserve: float = CLOSE_RESERVE_SECONDS,
    transport_options: Mapping[str, Any] | None = None,
) -> McpServers:
    """Start the servers ``config`` enables, in ``root``, and return what came of it.

    Each is the program of its ``[mcp.<name>]``, started in ``root`` with its stderr in
    ``<home>/.nanoclaude/logs``; its tools are adapted with ``config.limits.output_cap_bytes``
    as the cap a result is held to, and pinned in ``<home>/.nanoclaude/mcp-pins.json``.
    ``budget`` and ``close_reserve`` are the module docstring's two numbers, and
    ``transport_options`` are passed to each :class:`~nanoclaude.mcp.transport.StdioTransport`
    (its grace periods), for a test that cannot wait out the real ones.

    Never raises for a server: every way one can fail is a warning and a report.
    """
    entries = {name: _Entry(server) for name, server in config.mcp.items()}
    enabled = [server for server in config.mcp.values() if server.enabled]
    if not enabled:
        return McpServers(entries, (), ())
    try:
        pins = PinStore.load(pins_path(home))
    except PinStoreError as error:
        reason = _reason_of(error)
        for server in enabled:
            entries[server.name].failure = reason
        return McpServers(entries, (), [failed_line(server.name, reason) for server in enabled])

    log_dir = Path(home) / CONFIG_DIRNAME / LOG_DIRNAME
    starts = _Starts()
    for server in enabled:
        starts.tasks[server.name] = asyncio.create_task(
            _start_one(
                server,
                root=str(root),
                log_dir=log_dir,
                pins=pins,
                cap=config.limits.output_cap_bytes,
                transport_options=transport_options or {},
                transports=starts.transports,
            ),
            name=f"mcp-start-{server.name}",
        )
    try:
        await _wait_for(starts, budget, close_reserve)
    except BaseException:
        # Ctrl+C, most likely, while they start: nothing that was started stays running.
        await _abandon(starts)
        raise

    warnings: list[str] = []
    adapted: list[AdaptedTools] = []
    for server in enabled:
        entry = entries[server.name]
        task = starts.tasks[server.name]
        outcome = task.result() if task.done() and not task.cancelled() else None
        if isinstance(outcome, _Started):
            entry.client, entry.transport = outcome.client, outcome.transport
            adapted.append(outcome.adapted)
            continue
        entry.failure = (
            outcome.reason
            if isinstance(outcome, _Failed)
            else f"it did not finish starting within {budget:g} seconds"
        )
        warnings.append(failed_line(server.name, entry.failure))
    combined = combine_tools(adapted)
    for entry in entries.values():
        entry.tools = tuple(
            tool for tool in combined.tools if tool.server_name == entry.config.name
        )
    return McpServers(entries, combined.tools, [*warnings, *combined.warnings])
