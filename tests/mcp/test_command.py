# tests/mcp/test_command.py
"""The ``/mcp`` command: what it says of each server, and of the tools of one."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from nanoclaude.agent.session import Session
from nanoclaude.cli.commands import COMMANDS, dispatch
from nanoclaude.mcp.servers import McpServers, ServerReport
from nanoclaude.testing.scripted import says
from nanoclaude.testing.session import build_session
from tests.cli.helpers import plain_console
from tests.mcp.helpers import server

Start = Callable[..., Awaitable[McpServers]]


async def mcp_says(session: Session, *args: str) -> str:
    console, buffer = plain_console(200)
    assert await dispatch("mcp", list(args), session, console), "/mcp ended the REPL"
    return buffer.getvalue()


def test_mcp_is_a_command_with_a_summary_that_says_what_it_does():
    assert COMMANDS["mcp"].summary.strip()
    assert not COMMANDS["mcp"].ends_repl


async def test_mcp_lists_every_configured_server_with_its_state_its_era_and_its_tool_count(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(
        server("alpha", "modern"),
        server("beta", "legacy"),
        server("broken", "crash-on-start"),
        server("off", "modern", enabled=False),
    )
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    lines = (await mcp_says(session)).splitlines()
    assert len(lines) == 4, lines
    alpha, beta, broken, off = lines
    assert re.fullmatch(r"alpha  +running  +modern  +1 tool", alpha), alpha
    assert re.fullmatch(r"beta  +running  +legacy  +1 tool", beta), beta
    assert re.fullmatch(r"broken  +failed  +-  +0 tools  .+", broken), broken
    failure = servers.report("broken")
    assert failure is not None and failure.reason is not None
    assert broken.endswith(failure.reason), "the reason a server failed is not on its line"
    assert re.fullmatch(r"off  +disabled  +-  +0 tools", off), off
    await session.aclose()


async def test_mcp_names_the_tools_of_one_server_with_whether_each_changed_and_a_fingerprint(
    start: Start, mcp_root: Path
) -> None:
    first = await start(server("one", "modern"))
    await first.aclose()  # the pins now hold what it looked like
    servers = await start(server("one", "changed-definition"))
    (tool,) = servers.tools
    assert tool.changed
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    text = await mcp_says(session, "one")
    (line,) = [ln for ln in text.splitlines() if ln.strip().startswith("mcp__one__echo")]
    name, short, flag = line.split()
    assert (name, short, flag) == ("mcp__one__echo", tool.fingerprint[:12], "changed"), line
    assert re.fullmatch(r"[0-9a-f]{12}", short)
    assert tool.fingerprint[12:] not in text, "more of the fingerprint than 12 characters"
    assert "asks before every call" in text, "nothing says what a changed tool means"
    await session.aclose()


async def test_a_tool_that_is_not_changed_says_so_and_there_is_no_note_about_changes(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(server("one", "modern"))
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    text = await mcp_says(session, "one")
    (line,) = [ln for ln in text.splitlines() if ln.strip().startswith("mcp__one__echo")]
    assert line.split()[2] == "unchanged" and "asks before every call" not in text
    await session.aclose()


async def test_the_tools_of_one_server_are_listed_by_name_in_order(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(server("p", "paginated"))
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    names = [ln.split()[0] for ln in (await mcp_says(session, "p")).splitlines()[1:]]
    assert names == sorted(t.name for t in servers.tools) and len(names) == 6
    await session.aclose()


async def test_a_server_that_is_not_running_has_no_tools_to_list_and_says_why(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(server("broken", "crash-on-start"), server("off", enabled=False))
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    broken = await mcp_says(session, "broken")
    assert broken.splitlines()[0].split()[:2] == ["broken", "failed"]
    assert "no tools are registered" in broken
    off = await mcp_says(session, "off")
    assert off.splitlines()[0].split()[:2] == ["off", "disabled"]
    assert "no tools are registered" in off
    await session.aclose()


async def test_a_server_nobody_configured_is_an_error_that_names_the_ones_there_are(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(server("one", "modern"), server("two", "modern"))
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    assert await mcp_says(session, "three") == (
        "error: no MCP server named 'three' — the servers are one, two\n"
    )
    await session.aclose()


async def test_more_than_one_server_name_is_an_error_and_not_the_first_of_them(
    start: Start, mcp_root: Path
) -> None:
    servers = await start(server("one", "modern"), server("two", "modern"))
    session = build_session(mcp_root, [says("hi")], mcp=servers)
    text = await mcp_says(session, "one", "two")
    assert text.startswith("error: ") and "one server" in text and "running" not in text
    await session.aclose()


async def test_with_no_server_configured_it_says_where_to_configure_one(mcp_root: Path) -> None:
    session = build_session(mcp_root, [says("hi")])
    text = await mcp_says(session)
    assert text == (
        "no MCP servers are configured — add an [mcp.<name>] table to "
        "~/.nanoclaude/config.toml, as docs/configuration.md describes\n"
    )
    assert await mcp_says(session, "one") == text
    await session.aclose()


class _Reports:
    """What a session holds of its servers, as far as ``/mcp`` reads it."""

    def __init__(self, *reports: ServerReport) -> None:
        self._reports = reports

    def reports(self) -> tuple[ServerReport, ...]:
        return self._reports

    def report(self, name: str) -> ServerReport | None:
        return next((r for r in self._reports if r.name == name), None)

    async def aclose(self) -> None:
        return None


async def test_nothing_a_server_wrote_reaches_the_terminal_as_anything_but_text(
    mcp_root: Path,
) -> None:
    hostile = ServerReport(
        "evil", "failed", "oops \x1b[2J\x1b]0;owned\x07 [bold]x[/bold]", None, ()
    )
    session = build_session(mcp_root, [says("hi")])
    session.mcp = _Reports(hostile)  # type: ignore[assignment]
    for args in ((), ("evil",)):
        text = await mcp_says(session, *args)
        assert "\x1b" not in text and "\x07" not in text, repr(text)
        assert "[bold]x[/bold]" in text, "text in brackets was read as markup"
    await session.aclose()
