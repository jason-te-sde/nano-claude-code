"""Fixtures for the MCP tests: transports and clients that are closed when the test ends."""

from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.mcp.transport import StdioTransport


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
