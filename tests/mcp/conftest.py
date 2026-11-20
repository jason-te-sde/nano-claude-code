"""Fixtures for the MCP tests: transports and clients that are closed when the test ends."""

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.config.schema import McpServerConfig
from nanoclaude.mcp.client import McpClient
from nanoclaude.mcp.servers import McpServers, start_servers
from nanoclaude.mcp.transport import StdioTransport
from tests.mcp.helpers import config_of, fake_server


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "mcp-test.log"


@pytest.fixture
async def make_transport(
    tmp_path: Path, log_path: Path
) -> AsyncIterator[Callable[..., StdioTransport]]:
    """Build a transport in the test's directory, and close every one when the test ends.

    Python 3.13 turns a process, a pipe or a transport that is garbage collected unclosed
    into an error, and a test that failed halfway would otherwise leave it for whichever
    test is running when it is collected.
    """
    made: list[StdioTransport] = []

    def make(command: str, args: Sequence[str] = (), **kwargs: Any) -> StdioTransport:
        kwargs.setdefault("cwd", str(tmp_path))
        kwargs.setdefault("log_path", log_path)
        transport = StdioTransport(command, args, **kwargs)
        made.append(transport)
        return transport

    yield make
    for transport in made:
        await transport.close()


@pytest.fixture
def transports() -> dict[McpClient, StdioTransport]:
    """The transport of each client ``make_client`` built, for a test that needs its pid."""
    return {}


@pytest.fixture
async def make_client(
    make_transport: Callable[..., StdioTransport],
    transports: dict[McpClient, StdioTransport],
) -> AsyncIterator[Callable[..., McpClient]]:
    """Build a client for the fake server in ``mode``, and close every one at the end."""
    made: list[McpClient] = []

    def make(
        mode: str, *settings: str, transport_options: dict[str, Any] | None = None, **options: Any
    ) -> McpClient:
        command, args = fake_server(mode, *settings)
        transport = make_transport(command, args, **(transport_options or {}))
        client = McpClient(transport, **options)
        transports[client] = transport
        made.append(client)
        return client

    yield make
    for client in made:
        await client.close()


@pytest.fixture
def mcp_home(tmp_path: Path) -> Path:
    """The home ncc keeps its state in for a test of the servers of a session."""
    return tmp_path / "home"


@pytest.fixture
def mcp_root(tmp_path: Path) -> Path:
    """The project directory: where a server is started."""
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
async def start(
    mcp_home: Path, mcp_root: Path
) -> AsyncIterator[Callable[..., Awaitable[McpServers]]]:
    """Start the servers of a session, and end every one that a test left running.

    ``start(server, ...)`` takes the servers the configuration has and what
    :func:`~nanoclaude.mcp.servers.start_servers` takes; the cap is the configuration's
    ``output_cap_bytes``.
    """
    started: list[McpServers] = []

    async def go(*servers: McpServerConfig, cap: int = 100_000, **options: Any) -> McpServers:
        result = await start_servers(
            config_of(*servers, cap=cap), root=str(mcp_root), home=mcp_home, **options
        )
        started.append(result)
        return result

    yield go
    for servers in started:
        await servers.aclose()
