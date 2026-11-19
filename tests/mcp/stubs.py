"""A stand-in for a connected client, for the tests that are about what the adapter makes of
what a client says and not about the client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nanoclaude.mcp.client import CallResult, RemoteTool

OBJECT: dict[str, Any] = {"type": "object"}


class StubClient:
    """Lists the tools it was given, and answers a call with what it was told to.

    ``calls`` records ``(name, arguments, timeout)`` of every call, and ``listed`` counts the
    times the tools were listed, so that a test can say what was and was not asked of it.
    """

    def __init__(
        self,
        tools: Sequence[RemoteTool] = (),
        *,
        result: CallResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.tools = list(tools)
        self.result = result if result is not None else CallResult([{"type": "text", "text": "ok"}])
        self.error = error
        self.calls: list[tuple[str, Mapping[str, Any], float | None]] = []
        self.listed = 0

    async def list_tools(self) -> list[RemoteTool]:
        self.listed += 1
        return list(self.tools)

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        timeout: float | None = None,  # noqa: ASYNC109 - the client's own parameter
    ) -> CallResult:
        self.calls.append((name, arguments, timeout))
        if self.error is not None:
            raise self.error
        return self.result


def tool(name: str, description: object = "A tool.", schema: object = OBJECT) -> RemoteTool:
    return RemoteTool(name, description, schema)
