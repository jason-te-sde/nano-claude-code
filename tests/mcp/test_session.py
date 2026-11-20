# tests/mcp/test_session.py
"""A session that has MCP servers: what its model is told, what it registers, what closing it
does to the servers, and what a sub-agent is not given."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.agent.subagent import run_subagent
from nanoclaude.mcp.servers import McpServers
from nanoclaude.prompts import MCP_RESULTS_ARE_DATA
from nanoclaude.subagents import SubAgentCall
from nanoclaude.testing.scripted import calls, says
from nanoclaude.testing.session import build_session
from tests.mcp.helpers import all_gone, group_is_gone, pids_of, server

Start = Callable[..., Awaitable[McpServers]]


async def test_a_session_with_a_server_registers_its_tools_and_tells_the_model_what_they_are(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(server("one", "modern"))
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    await session.run("hello")
    (request,) = session.model.requests
    assert "mcp__one__echo" in [spec.name for spec in request.tools]
    assert request.system.count(MCP_RESULTS_ARE_DATA) == 1
    await session.aclose()


async def test_a_session_with_no_mcp_tool_does_not_mention_them(mcp_root: Path) -> None:
    session = build_session(mcp_root, [says("hi")])
    await session.run("hello")
    (request,) = session.model.requests
    assert MCP_RESULTS_ARE_DATA not in request.system
    assert not [spec.name for spec in request.tools if spec.name.startswith("mcp__")]
    await session.aclose()


async def test_closing_the_session_closes_every_server_it_started(
    start: Start, mcp_root: Path, mcp_home: Path
) -> None:
    servers = await start(server("one", "modern"), server("two", "legacy"))
    pids = [*pids_of(mcp_home, "one"), *pids_of(mcp_home, "two")]
    assert len(pids) == 2 and not any(group_is_gone(pid) for pid in pids)
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    await session.aclose()
    assert await all_gone(pids), "a server was left running when the session was closed"
    assert session.router is not None


async def test_a_step_of_closing_that_fails_does_not_leave_a_server_running(
    start: Start, mcp_root: Path, mcp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers = await start(server("one", "modern"))
    session = build_session(mcp_root, [says("hi")], mcp=servers)

    async def fails() -> None:
        raise RuntimeError("the provider's client would not close")

    monkeypatch.setattr(session.router, "aclose", fails)
    with pytest.raises(RuntimeError, match="would not close"):
        await session.aclose()
    assert await all_gone(pids_of(mcp_home, "one")), "the server outlived a failed close"


async def test_the_session_closes_the_providers_then_the_servers_and_the_store_last(
    start: Start, mcp_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers = await start(server("one", "modern"))
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    order: list[str] = []

    def noting(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            order.append(name)
            return real(*args, **kwargs)

        return wrapper

    def noting_async(name: str, real: Callable[..., Awaitable[None]]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> None:
            order.append(name)
            await real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(session, "_persist", noting("conversation", session._persist))
    monkeypatch.setattr(session, "_finish", noting("spend", session._finish))
    monkeypatch.setattr(session.router, "aclose", noting_async("providers", session.router.aclose))
    monkeypatch.setattr(servers, "aclose", noting_async("servers", servers.aclose))
    monkeypatch.setattr(session, "_close_store", noting("store", session._close_store))
    await session.aclose()
    # docs/architecture.md says the servers go after the providers' clients and before the store.
    assert order == ["conversation", "spend", "providers", "servers", "store"]


async def test_the_servers_are_closed_by_the_session_once_and_a_second_close_does_nothing(
    start: Start, mcp_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers = await start(server("one", "modern"))
    closed: list[int] = []
    real = servers.aclose

    async def counting() -> None:
        closed.append(1)
        await real()

    monkeypatch.setattr(servers, "aclose", counting)
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    await session.aclose()
    await session.aclose()
    assert closed == [1]


# --------------------------------------------------------------------------
# A sub-agent is never given an MCP tool
# --------------------------------------------------------------------------


async def test_a_sub_agent_never_sees_an_mcp_tool_the_session_has_registered(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(server("one", "modern"))
    session = build_session(
        mcp_root,
        [says("unused")],
        explore_script=[calls("Read", {"path": "a.py"}, call_id="r1"), says("found it")],
        mcp=servers,
    )
    (mcp_root / "a.py").write_text("x = 1\n")
    assert "mcp__one__echo" in session.registry, "the parent does not hold the MCP tool"
    result = await run_subagent(session, SubAgentCall("explore", "where?", "task_1"), audit_turn=0)
    assert result.text == "found it"
    assert session.explore_model is not None
    assert session.explore_model.requests, "the sub-agent sent no request"
    for request in session.explore_model.requests:
        names = [spec.name for spec in request.tools]
        assert sorted(names) == ["Glob", "Grep", "Read"], names
        assert MCP_RESULTS_ARE_DATA not in request.system
    await session.aclose()
