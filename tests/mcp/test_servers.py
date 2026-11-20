# tests/mcp/test_servers.py
"""The servers of a session: started together, held to a budget, one warning for each that
fails, their tools merged, and every one of them ended when the session is."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.config.schema import McpServerConfig
from nanoclaude.mcp.pins import LOCK_SUFFIX, PINS_FILENAME, pins_path
from nanoclaude.mcp.servers import (
    CLOSE_RESERVE_SECONDS,
    MAX_TOOL_WARNINGS_PER_SERVER,
    START_BUDGET_SECONDS,
    McpServers,
    short_reason,
    start_servers,
)
from nanoclaude.mcp.transport import STDIN_GRACE_SECONDS, TERM_GRACE_SECONDS
from nanoclaude.private import create_private_file
from nanoclaude.tools.registry import ToolRegistry
from tests.mcp.helpers import (
    all_gone,
    config_of,
    events,
    eventually,
    group_is_gone,
    helper_pid,
    kill_helpers,
    log_of,
    pids_of,
    process_is_gone,
    received_methods,
    server,
)

Start = Callable[..., Awaitable[McpServers]]


def one(servers: McpServers, name: str) -> Any:
    report = servers.report(name)
    assert report is not None, f"no report for {name}: {servers.reports()}"
    return report


# --------------------------------------------------------------------------
# Starting
# --------------------------------------------------------------------------


async def test_every_enabled_server_is_started_and_its_tools_are_the_registrys(start):
    servers = await start(server("one", "modern"), server("two", "legacy"))
    assert [tool.name for tool in servers.tools] == ["mcp__one__echo", "mcp__two__echo"]
    assert servers.warnings == ()
    first, second = servers.reports()
    assert (first.name, first.state, first.era, len(first.tools), first.reason) == (
        "one",
        "running",
        "modern",
        1,
        None,
    )
    assert (second.name, second.state, second.era, len(second.tools)) == (
        "two",
        "running",
        "legacy",
        1,
    )


async def test_a_disabled_server_is_never_started_and_is_reported_as_disabled(start, mcp_home):
    servers = await start(server("off", "modern", enabled=False), server("on", "modern"))
    assert pids_of(mcp_home, "off") == [], "a server that is disabled was started"
    assert len(pids_of(mcp_home, "on")) == 1
    off = one(servers, "off")
    assert (off.state, off.era, off.tools, off.reason) == ("disabled", None, (), None)
    assert servers.warnings == ()
    assert [tool.name for tool in servers.tools] == ["mcp__on__echo"]


async def test_a_session_with_no_server_configured_starts_nothing_and_says_nothing(start):
    servers = await start()
    assert (servers.tools, servers.warnings, servers.reports()) == ((), (), ())


async def test_the_servers_are_started_together_and_not_one_after_another(start):
    # Each answers server/discover a second after it is asked. Started one after another,
    # four of them would take four seconds.
    late = ("discover=late", "discover_delay=1.0")
    began = time.monotonic()
    servers = await start(*[server(f"s{n}", "modern", *late) for n in range(4)])
    elapsed = time.monotonic() - began
    assert len(servers.tools) == 4, servers.warnings
    assert elapsed < 3.0, f"four servers that take a second each started in {elapsed:.1f}s"


async def test_a_server_is_started_in_the_project_root_and_keeps_its_stderr_in_the_logs(
    start, mcp_home, mcp_root
):
    # A program that writes where it is to stderr and ends is no server, so it fails; what
    # matters here is where it was run and where what it wrote went.
    command = sys.executable
    args = ("-I", "-S", "-c", "import os, sys; print(os.getcwd(), file=sys.stderr)")
    await start(McpServerConfig(name="where", command=command, args=args))
    assert str(mcp_root.resolve()) in log_of(mcp_home, "where").read_text()


async def test_the_tools_are_pinned_in_the_homes_pins_file_and_a_changed_one_is_marked(
    start, mcp_home
):
    first = await start(server("one", "modern"))
    assert not first.tools[0].changed
    await first.aclose()
    pinned = json.loads(pins_path(mcp_home).read_text())
    assert list(pinned["one"]) == ["echo"], pinned
    second = await start(server("one", "changed-definition"))
    assert second.tools[0].changed, "a definition that differs from its pin is not marked"


# --------------------------------------------------------------------------
# What a server that fails costs: one line, and nothing else
# --------------------------------------------------------------------------

FAILURES = [
    pytest.param(
        lambda: McpServerConfig(name="bad", command="/nonexistent/mcp-server"),
        ("cannot start",),
        id="the-program-cannot-be-started",
    ),
    pytest.param(
        lambda: server("bad", "crash-on-start"),
        ("closed its output", "not reading its input"),
        id="the-server-exits-at-once",
    ),
    pytest.param(
        lambda: server("bad", "unsupported-version"),
        ("protocol versions",),
        id="no-protocol-version-in-common",
    ),
    pytest.param(
        lambda: server("bad", "legacy", "initialize=method-not-found"),
        ("the server answered with an error", "Method not found"),
        id="the-server-answers-initialize-with-an-error",
    ),
    pytest.param(
        lambda: server("bad", "modern", "non_object_result=tools/list"),
        ("tools/list must be an object",),
        id="the-tool-list-is-not-one",
    ),
]


@pytest.mark.parametrize(("make", "reasons"), FAILURES)
async def test_a_server_that_fails_adds_one_warning_and_registers_nothing_and_the_rest_start(
    start, mcp_home, make, reasons
):
    servers = await start(make(), server("good", "modern"))
    (warning,) = servers.warnings
    assert warning.startswith("MCP server 'bad' failed to start"), warning
    assert "\n" not in warning and any(reason in warning for reason in reasons), warning
    bad = one(servers, "bad")
    assert (bad.state, bad.tools, bad.era) == ("failed", (), None)
    assert bad.reason is not None and any(reason in bad.reason for reason in reasons), bad.reason
    assert [tool.name for tool in servers.tools] == ["mcp__good__echo"]
    assert one(servers, "good").state == "running"
    started = pids_of(mcp_home, "bad")
    if started:
        assert await all_gone(started), "the server that failed to start was left running"


async def test_a_bug_while_one_server_starts_is_that_servers_warning_and_not_a_crash(
    start, mcp_home, monkeypatch
):
    async def breaks(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("a defect of ours, with a detail that is not shown")

    monkeypatch.setattr("nanoclaude.mcp.servers.adapt_tools", breaks)
    servers = await start(server("one", "modern"))
    (warning,) = servers.warnings
    assert "failed to start (unexpected error (RuntimeError))" in warning, warning
    assert "detail" not in warning
    assert servers.tools == ()
    assert await all_gone(pids_of(mcp_home, "one")), "the server was left running by the bug"


@pytest.mark.parametrize(
    "contents",
    ["{this is not json", "[" * 200_000],
    ids=["not-json", "a-file-the-parser-gives-out-on"],
)
async def test_a_corrupt_pins_file_fails_every_enabled_server_closed_and_starts_none(
    start, mcp_home, contents
):
    path = pins_path(mcp_home)
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    servers = await start(server("one", "modern"), server("two", "legacy"))
    assert servers.tools == (), "tools were registered with no pins to check them against"
    assert len(servers.warnings) == 2, servers.warnings
    for name, warning in zip(("one", "two"), servers.warnings, strict=True):
        assert f"MCP server '{name}' failed to start" in warning and "pins file" in warning
        assert one(servers, name).state == "failed"
    assert pids_of(mcp_home, "one") == [] and pids_of(mcp_home, "two") == [], (
        "a server was started whose tools could not be registered"
    )
    assert path.read_text() == contents, "the pins file somebody wrote was replaced"


async def test_a_corrupt_pins_file_is_not_looked_at_when_no_server_is_enabled(start, mcp_home):
    path = pins_path(mcp_home)
    path.parent.mkdir(parents=True)
    path.write_text("{this is not json")
    servers = await start(server("off", "modern", enabled=False))
    assert servers.warnings == ()


def test_a_reason_is_one_short_line_that_no_terminal_acts_on():
    assert short_reason("one\ntwo\x1b[2J  \x07three") == "one two three"
    long = short_reason("x" * 1000)
    assert len(long) <= 200 and long.endswith("…")
    assert short_reason("") == "no reason was given"


# --------------------------------------------------------------------------
# The tools of all the servers are one set
# --------------------------------------------------------------------------

_TOOL = '[{{"name":"{name}","description":"d","inputSchema":{{"type":"object"}}}}]'


async def test_a_name_two_servers_come_to_is_dropped_from_both_with_a_warning(start):
    # `b__t` of the server `a` and `t` of the server `a__b` are the same registry name, which
    # the registry would refuse to hold twice.
    servers = await start(
        server("a", "modern", "tools=" + _TOOL.format(name="b__t")),
        server("a__b", "modern", "tools=" + _TOOL.format(name="t")),
        server("c", "modern"),
    )
    assert [tool.name for tool in servers.tools] == ["mcp__c__echo"]
    (warning,) = servers.warnings
    assert "mcp__a__b__t" in warning and "'a'" in warning and "'a__b'" in warning, warning
    assert [(r.name, r.state, len(r.tools)) for r in servers.reports()] == [
        ("a", "running", 0),
        ("a__b", "running", 0),
        ("c", "running", 1),
    ]
    ToolRegistry(list(servers.tools))  # raises on a name held twice


def _bad_tools(count: int) -> str:
    """The setting of a server whose ``count`` tools all have an input schema that is no object."""
    tools = [
        {"name": f"bad{n}", "description": "d", "inputSchema": {"type": "string"}}
        for n in range(count)
    ]
    return "tools=" + json.dumps(tools)


@pytest.mark.parametrize("count", [MAX_TOOL_WARNINGS_PER_SERVER, MAX_TOOL_WARNINGS_PER_SERVER + 1])
async def test_warnings_about_one_servers_tools_are_one_line_with_a_count_past_a_few(start, count):
    servers = await start(server("bad", "modern", _bad_tools(count)), server("good", "modern"))
    if count <= MAX_TOOL_WARNINGS_PER_SERVER:
        assert len(servers.warnings) == count
        assert all("'bad" in line and "is dropped" in line for line in servers.warnings)
        return
    (line,) = servers.warnings
    assert line.startswith("MCP server 'bad': "), line
    assert (
        f"{count} warnings about its tools" in line and "the first: tool 'bad0' is dropped" in line
    )
    assert "\n" not in line
    assert [t.name for t in servers.tools] == ["mcp__good__echo"]


async def test_the_warnings_of_each_server_are_counted_apart_from_those_of_another(start):
    servers = await start(
        server("one", "modern", _bad_tools(5)), server("two", "modern", _bad_tools(5))
    )
    assert len(servers.warnings) == 2
    assert "'one'" in servers.warnings[0] and "'two'" in servers.warnings[1]


# --------------------------------------------------------------------------
# The cap
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("cap", "cut"), [(300, True), (100_000, False)])
async def test_a_result_is_held_to_the_configurations_output_cap(start, ctx, cap, cut):
    servers = await start(server("one", "modern"), cap=cap)
    (tool,) = servers.tools
    outcome = await tool.run(ctx, "call-1", {"text": "x" * 2000})
    assert ("[output cut:" in outcome.content) is cut, outcome.content[:200]
    if cut:
        assert len(outcome.content.encode()) < 300 + 100


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------


def test_the_budget_is_fifteen_seconds_and_a_close_that_waits_out_both_periods_fits_in_it():
    assert START_BUDGET_SECONDS == 15.0
    assert CLOSE_RESERVE_SECONDS > STDIN_GRACE_SECONDS + TERM_GRACE_SECONDS
    assert CLOSE_RESERVE_SECONDS < START_BUDGET_SECONDS


async def started_within(
    start: Start, home: Path, *servers: McpServerConfig, bound: float, **options: Any
) -> tuple[McpServers, float]:
    """Start ``servers``, and fail rather than wait when a guard is missing.

    A server that never answers and has a long timeout is stopped only by the budget, and a
    budget that is not enforced would leave this test waiting for ten minutes.
    """
    began = time.monotonic()
    try:
        async with asyncio.timeout(bound):
            result = await start(*servers, **options)
    except TimeoutError:
        for name in {s.name for s in servers}:
            for pid in pids_of(home, name):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
        raise AssertionError(f"starting was still going after {bound} seconds") from None
    return result, time.monotonic() - began


async def test_a_server_that_never_answers_is_given_up_on_at_the_budget_and_one_line_says_so(
    start, mcp_home
):
    graces = {"stdin_grace": 0.3, "term_grace": 0.3}
    servers, elapsed = await started_within(
        start,
        mcp_home,
        server("mute", "never-answers", timeout=600),
        server("good", "modern"),
        bound=20,
        budget=2.0,
        close_reserve=1.0,
        transport_options=graces,
    )
    (warning,) = servers.warnings
    assert "MCP server 'mute' failed to start" in warning and "within 2 seconds" in warning
    assert [tool.name for tool in servers.tools] == ["mcp__good__echo"]
    assert elapsed >= 1.0 - 0.1, f"gave up after {elapsed:.2f}s, before the start period ended"
    assert elapsed < 2.0 + 0.5, f"waited {elapsed:.2f}s with a budget of 2"
    assert await all_gone(pids_of(mcp_home, "mute")), "the server that never answered is running"


async def test_the_budget_includes_the_time_a_cancelled_start_needs_to_close_its_server(
    start, mcp_home
):
    # It ignores the end of its stdin and SIGTERM, with a helper that does too: ending it
    # takes both periods and then SIGKILL. The reserve is what is left for that.
    graces = {"stdin_grace": 0.4, "term_grace": 0.4}
    servers, elapsed = await started_within(
        start,
        mcp_home,
        server("stubborn", "never-answers-stubborn", timeout=600),
        bound=20,
        budget=3.0,
        close_reserve=1.5,
        transport_options=graces,
    )
    assert len(servers.warnings) == 1
    # Given up on at 1.5 s; its graceful ending then takes about 0.8 s more.
    assert 1.5 - 0.1 <= elapsed < 3.0 + 0.5, f"{elapsed:.2f}s"
    pids = pids_of(mcp_home, "stubborn")
    assert await all_gone(pids), "a process of the stubborn server is still running"
    # It was ended in order and not killed outright: SIGTERM was sent, and ignored, before the
    # SIGKILL. A start that was given the whole budget would have had no time left for that.
    assert "sigterm-ignored" in events(log_of(mcp_home, "stubborn")), (
        "the server was killed at once, with no room left to close it in order"
    )


async def test_with_the_shipped_budget_and_periods_the_worst_server_is_gone_within_fifteen_seconds(
    start, mcp_home
):
    # The case the budget was sized for, in real time and with nothing shortened: a server that
    # never answers, ignores the end of its stdin and SIGTERM, and has a helper that does too.
    # Ten seconds of starting, then about four to end it in order, which is inside fifteen.
    servers, elapsed = await started_within(
        start, mcp_home, server("stubborn", "never-answers-stubborn", timeout=600), bound=60
    )
    assert len(servers.warnings) == 1
    assert elapsed >= START_BUDGET_SECONDS - CLOSE_RESERVE_SECONDS - 0.1
    assert elapsed < START_BUDGET_SECONDS, f"{elapsed:.2f}s with a budget of 15"
    pids = pids_of(mcp_home, "stubborn")
    assert await all_gone(pids), "a process of the stubborn server is still running"
    assert "sigterm-ignored" in events(log_of(mcp_home, "stubborn")), (
        "the server was killed at once and not ended in order"
    )


async def test_what_is_left_to_close_after_the_kill_is_held_to_what_is_left_of_the_budget(
    start, mcp_home
):
    # A helper that left the server's group and holds its pipes: SIGKILL to the group does not
    # end it, so closing the transport waits for the end of stderr (five seconds here). The
    # budget is two, and what is left after the kill must not wait for that.
    log = log_of(mcp_home, "stuck")
    graces = {"stdin_grace": 5.0, "term_grace": 5.0, "drain_grace": 5.0}
    try:
        servers, elapsed = await started_within(
            start,
            mcp_home,
            server("stuck", "never-answers-stubborn", "helper=escaped", timeout=600),
            bound=30,
            budget=2.0,
            close_reserve=1.0,
            transport_options=graces,
        )
        assert len(servers.warnings) == 1
        assert elapsed < 2.0, f"waited {elapsed:.2f}s with a budget of 2, which is not to be passed"
    finally:
        # Only now is the pipe let go, and what was still closing can finish.
        kill_helpers(log)
        leftovers = [t for t in asyncio.all_tasks() if t.get_name().startswith("mcp-start-")]
        if leftovers:
            await asyncio.wait(leftovers, timeout=10)


@pytest.mark.parametrize("reserve", [0.5, 1.0], ids=["no-room-to-close", "room-for-half-a-second"])
async def test_a_close_that_does_not_finish_in_the_reserve_is_cut_short_with_a_kill(
    start, mcp_home, reserve
):
    # Its graceful ending would take ten seconds. The budget is two, so the rest is a kill.
    graces = {"stdin_grace": 5.0, "term_grace": 5.0}
    servers, elapsed = await started_within(
        start,
        mcp_home,
        server("stubborn", "never-answers-stubborn", timeout=600),
        bound=30,
        budget=2.0,
        close_reserve=reserve,
        transport_options=graces,
    )
    assert len(servers.warnings) == 1
    assert elapsed < 2.0, f"waited {elapsed:.2f}s with a budget of 2, which is not to be passed"
    pids = pids_of(mcp_home, "stubborn")
    assert await all_gone(pids), "the kill did not end the server and its helper"


_HOLD_THE_LOCK = (
    "import fcntl, os, sys, time\n"
    "descriptor = os.open(sys.argv[1], os.O_RDWR)\n"
    "fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
    "print('locked', flush=True)\n"
    "time.sleep(60)\n"
)


async def test_a_pins_lock_held_by_another_process_does_not_stall_the_other_servers(
    start, mcp_home, monkeypatch
):
    # Another ncc is saving pins and holds the lock. Each server waits for it to write its own,
    # and the waits are together, not one after another: with the loop blocked in each, four
    # servers would take four times as long as the wait, and spend a budget that is for the
    # starts. They are started, with a warning each that their pins could not be written.
    monkeypatch.setattr("nanoclaude.mcp.pins.LOCK_WAIT_SECONDS", 1.5)
    lock = pins_path(mcp_home).with_name(PINS_FILENAME + LOCK_SUFFIX)
    lock.parent.mkdir(parents=True)
    create_private_file(lock)
    holder = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-S",
        "-c",
        _HOLD_THE_LOCK,
        str(lock),
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None
        assert (await holder.stdout.readline()).decode().strip() == "locked"
        began = time.monotonic()
        servers = await start(
            *[server(f"s{n}", "modern") for n in range(4)], budget=4.0, close_reserve=1.0
        )
        elapsed = time.monotonic() - began
    finally:
        holder.terminate()
        await holder.wait()
    assert [r.state for r in servers.reports()] == ["running"] * 4, servers.warnings
    assert len(servers.tools) == 4
    assert len(servers.warnings) == 4 and all("could not write" in w for w in servers.warnings)
    assert elapsed < 4.0, f"four servers took {elapsed:.1f}s with a lock to wait for"


async def test_a_start_that_is_cancelled_ends_every_server_it_had_started(mcp_home, mcp_root):
    # The person presses Ctrl+C while one server is still starting and another is up: by the
    # time the cancellation has gone through, nothing of either is left, and it did go through.
    config = config_of(
        server("good", "modern"), server("mute", "never-answers-stubborn", timeout=600)
    )
    starting = asyncio.create_task(
        start_servers(
            config,
            root=str(mcp_root),
            home=mcp_home,
            transport_options={"stdin_grace": 0.3, "term_grace": 0.3},
        )
    )
    good = log_of(mcp_home, "good")
    try:
        assert await eventually(
            lambda: (
                good.exists()
                and "tools/list" in received_methods(good)
                and bool(pids_of(mcp_home, "mute"))
            )
        ), "the servers were not both started before the test cancelled the start"
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, timeout=10)
        # Not polled for: a start that left its work to the loop would see it done a moment
        # later, and that is not the same as having ended the servers before it returned.
        for name in ("good", "mute"):
            (pid,) = pids_of(mcp_home, name)
            assert group_is_gone(pid), (
                f"the server {name} was still running when start_servers left"
            )
    finally:
        for name in ("good", "mute"):
            for pid in pids_of(mcp_home, name):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# What a server that fails later is, and ending them all
# --------------------------------------------------------------------------


async def test_a_server_that_goes_while_the_session_runs_is_reported_failed(start, mcp_home):
    servers = await start(server("one", "modern"))
    assert one(servers, "one").state == "running"
    (pid,) = pids_of(mcp_home, "one")
    os.killpg(pid, signal.SIGKILL)
    assert await eventually(lambda: one(servers, "one").state == "failed"), servers.reports()
    report = one(servers, "one")
    assert report.reason and "\n" not in report.reason
    assert len(report.tools) == 1, "the tool a failed server registered is not forgotten"


async def test_every_server_is_closed_together_and_nothing_of_one_is_left(start, mcp_home):
    # Each waits out its stdin period before it can be ended: four, one after another, would
    # take four of them.
    graces = {"stdin_grace": 1.0, "term_grace": 1.0}
    names = [f"s{n}" for n in range(4)]
    servers = await start(
        *[server(name, "ignore-stdin-eof") for name in names], transport_options=graces
    )
    pids = [pid for name in names for pid in pids_of(mcp_home, name)]
    assert len(pids) == 4 and not any(group_is_gone(pid) for pid in pids)
    began = time.monotonic()
    await servers.aclose()
    elapsed = time.monotonic() - began
    assert elapsed < 2.5, f"closing four servers that take a second each took {elapsed:.1f}s"
    assert await all_gone(pids), "a server was left running"
    await servers.aclose()  # a second close is harmless


@pytest.mark.parametrize("delay", [0, 0.2], ids=["before-any-close-began", "mid-close"])
async def test_a_close_that_is_cancelled_kills_every_server_before_it_lets_go(
    start, mcp_home, delay
):
    # Cancelled at once, the closes of the clients have not begun and the transports have had
    # no chance to kill what they hold; cancelled later, each transport is waiting out its period.
    servers = await start(
        server("one", "ignore-term"),
        server("two", "ignore-term"),
        transport_options={"stdin_grace": 30.0, "term_grace": 30.0},
    )
    pids = [*pids_of(mcp_home, "one"), *pids_of(mcp_home, "two")]
    assert len(pids) == 2 and not any(group_is_gone(pid) for pid in pids)
    closing = asyncio.create_task(servers.aclose())
    await asyncio.sleep(delay)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert await all_gone(pids), "cancelling the close left a server running"


@pytest.mark.parametrize("gap", [0, 0.3], ids=["at-once", "later"])
async def test_a_second_cancellation_while_a_start_is_being_abandoned_kills_every_server(
    mcp_home, mcp_root, gap
):
    config = config_of(server("mute", "never-answers-stubborn", timeout=600))
    starting = asyncio.create_task(
        start_servers(
            config,
            root=str(mcp_root),
            home=mcp_home,
            transport_options={"stdin_grace": 30.0, "term_grace": 30.0},
        )
    )
    try:
        assert await eventually(lambda: bool(pids_of(mcp_home, "mute")))
        starting.cancel()  # the first Ctrl+C: the server is being ended in order, slowly
        await asyncio.sleep(gap)
        starting.cancel()  # the second
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, timeout=10)
        assert await all_gone(pids_of(mcp_home, "mute")), "the second Ctrl+C left a server"
    finally:
        for pid in pids_of(mcp_home, "mute"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)


async def test_closing_ends_a_helper_a_server_left_behind_too(start, mcp_home):
    servers = await start(server("one", "orphan-helper"), transport_options={"term_grace": 0.3})
    log = log_of(mcp_home, "one")
    helper = await helper_pid(log)
    await servers.aclose()
    assert await all_gone(pids_of(mcp_home, "one"))
    assert await eventually(lambda: process_is_gone(helper)), "the helper outlived the server"
