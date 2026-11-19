# tests/mcp/test_transport.py
"""The stdio transport: how the server is started, how its lines come and go, what happens to
its stderr, and that closing it leaves nothing behind."""

import asyncio
import os
import signal
import time

import pytest

from nanoclaude.mcp.jsonrpc import MAX_LINE_BYTES, LineTooLongError
from nanoclaude.mcp.transport import (
    ENV_ALLOWLIST,
    STDERR_LOG_CAP,
    STDIN_GRACE_SECONDS,
    TERM_GRACE_SECONDS,
    CappedLog,
    StdioTransport,
    TransportError,
    build_environment,
)
from tests.mcp.helpers import (
    events,
    eventually,
    fake_server,
    group_is_gone,
    logged_helper_pid,
    open_descriptors,
    process_is_gone,
    python_child,
)

PING = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'


async def drain(transport: StdioTransport) -> list[bytes]:
    """Everything the server writes to stdout until it ends."""
    lines = []
    while (line := await transport.receive()) is not None:
        lines.append(line)
    return lines


async def when_ready(transport: StdioTransport) -> None:
    """Return once the fake server is reading, so that its signal handlers are in place."""
    await transport.send(PING)
    assert await transport.receive() == b'{"jsonrpc":"2.0","id":1,"result":{}}'


# --------------------------------------------------------------------------
# The environment
# --------------------------------------------------------------------------


def test_the_allowlist_is_the_nine_names_the_design_gives():
    assert ENV_ALLOWLIST == (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TERM",
    )


def test_the_environment_is_the_allowlist_from_the_parent_plus_what_is_configured():
    parent = {
        "PATH": "/bin",
        "HOME": "/home/me",
        "LANG": "C",
        "ANTHROPIC_API_KEY": "not-a-real-key",
        "OPENAI_API_KEY": "not-a-real-key",
        "SSH_AUTH_SOCK": "agent-socket",
    }
    built = build_environment({"TOKEN_FOR_THE_SERVER": "chosen"}, parent)
    assert built == {
        "PATH": "/bin",
        "HOME": "/home/me",
        "LANG": "C",
        "TOKEN_FOR_THE_SERVER": "chosen",
    }


def test_a_configured_value_wins_over_the_parents_and_an_absent_name_stays_absent():
    built = build_environment({"HOME": "/elsewhere"}, {"HOME": "/home/me"})
    assert built == {"HOME": "/elsewhere"}
    assert "TERM" not in build_environment({}, {})


async def test_the_child_sees_no_variable_outside_the_allowlist_and_the_configured_env(
    make_transport, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key-openai")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setenv("LANG", "C")
    transport = make_transport("/usr/bin/env", env={"CHOSEN_BY_THE_USER": "yes"})
    await transport.start()
    seen = dict(line.decode().split("=", 1) for line in await drain(transport))
    allowed = {*ENV_ALLOWLIST, "CHOSEN_BY_THE_USER"}
    assert set(seen) <= allowed, f"the child was given {sorted(set(seen) - allowed)}"
    assert seen["CHOSEN_BY_THE_USER"] == "yes"
    assert seen["LANG"] == "C", "an allowlisted name present in the parent was not passed"
    assert "PATH" in seen
    assert not any("not-a-real-key" in value for value in seen.values()), seen


# --------------------------------------------------------------------------
# Starting the process
# --------------------------------------------------------------------------


async def test_the_server_starts_in_the_directory_it_is_given(make_transport, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    transport = make_transport("/bin/pwd", ["-P"], cwd=str(root))
    await transport.start()
    assert await drain(transport) == [str(root.resolve()).encode()]


async def test_the_command_is_not_run_through_a_shell(make_transport):
    arguments = ["a b;c", "$(echo expanded)", "`echo expanded`", "*", "~"]
    transport = make_transport("/bin/echo", arguments)
    await transport.start()
    assert await drain(transport) == [" ".join(arguments).encode()]


async def test_the_server_leads_a_process_group_of_its_own(make_transport):
    command, args = python_child("import sys; sys.stdin.read()")
    transport = make_transport(command, args)
    await transport.start()
    assert transport.pid is not None
    assert os.getpgid(transport.pid) == transport.pid, "the server is not a group leader"
    assert transport.pid != os.getpgrp(), "the server shares the group of this process"
    assert os.getsid(transport.pid) == transport.pid, "the server is in this session"


async def test_a_command_that_cannot_be_run_is_a_transport_error(make_transport):
    transport = make_transport("/nonexistent/server-binary")
    with pytest.raises(TransportError, match="/nonexistent/server-binary"):
        await transport.start()
    assert transport.pid is None


async def test_nothing_can_be_sent_or_received_before_the_transport_starts(make_transport):
    transport = make_transport("/bin/cat")
    with pytest.raises(TransportError, match="not started"):
        await transport.send(PING)
    with pytest.raises(TransportError, match="not started"):
        await transport.receive()


async def test_a_transport_starts_once(make_transport):
    transport = make_transport("/bin/cat")
    assert transport.pid is None
    await transport.start()
    with pytest.raises(TransportError, match="already"):
        await transport.start()


# --------------------------------------------------------------------------
# Lines in and out
# --------------------------------------------------------------------------


async def test_a_line_sent_is_ended_by_a_newline_and_one_received_is_not(make_transport):
    transport = make_transport("/bin/cat")
    await transport.start()
    await transport.send(b'{"a":1}')
    await transport.send(b'{"b":2}')
    assert await transport.receive() == b'{"a":1}'
    assert await transport.receive() == b'{"b":2}'


async def test_lines_sent_at_once_by_many_tasks_arrive_whole(make_transport):
    transport = make_transport("/bin/cat")
    await transport.start()
    lines = [bytes([ord("a") + n]) * 200_000 for n in range(10)]
    received_back: list[bytes] = []

    async def read() -> None:
        for _ in lines:
            line = await transport.receive()
            assert line is not None
            received_back.append(line)

    reader = asyncio.create_task(read())
    await asyncio.wait_for(asyncio.gather(*(transport.send(line) for line in lines)), timeout=20)
    await asyncio.wait_for(reader, timeout=20)
    assert sorted(received_back) == sorted(lines), "a line was cut or two were mixed"


async def test_receive_says_none_when_the_server_has_closed_its_output(make_transport):
    command, args = python_child("pass")
    transport = make_transport(command, args)
    await transport.start()
    assert await transport.receive() is None
    assert await transport.receive() is None


async def test_sending_to_a_server_that_has_gone_is_a_transport_error(make_transport):
    command, args = python_child("pass")
    transport = make_transport(command, args)
    await transport.start()
    assert await transport.receive() is None
    with pytest.raises(TransportError):
        await transport.send(PING)


_WRITE_A_LINE = "import sys; sys.stdout.buffer.write(b'x' * int(sys.argv[1]) + b'\\n')"


async def test_a_line_of_exactly_four_mebibytes_arrives(make_transport):
    command, args = python_child(_WRITE_A_LINE, str(MAX_LINE_BYTES))
    transport = make_transport(command, args)
    await transport.start()
    line = await transport.receive()
    assert line is not None and len(line) == MAX_LINE_BYTES


async def test_a_line_over_four_mebibytes_is_refused_not_buffered(make_transport):
    command, args = python_child(_WRITE_A_LINE, str(MAX_LINE_BYTES + 1))
    transport = make_transport(command, args)
    await transport.start()
    with pytest.raises(LineTooLongError):
        await transport.receive()


async def test_a_line_over_the_limit_without_a_newline_is_refused_as_it_grows(make_transport):
    source = (
        "import sys; sys.stdout.buffer.write(b'x' * (int(sys.argv[1]) + 10)); "
        "sys.stdout.flush(); sys.stdin.read()"
    )
    command, args = python_child(source, str(MAX_LINE_BYTES))
    transport = make_transport(command, args)
    await transport.start()
    with pytest.raises(LineTooLongError):
        await asyncio.wait_for(transport.receive(), timeout=10)


# --------------------------------------------------------------------------
# stderr
# --------------------------------------------------------------------------

_WRITE_TO_STDERR = (
    "import sys; sys.stderr.write(sys.argv[1]); sys.stderr.flush(); sys.stdout.write('out\\n')"
)


async def test_stderr_is_appended_to_the_log_and_not_to_stdout(make_transport, log_path):
    command, args = python_child(_WRITE_TO_STDERR, "diagnostics\n")
    transport = make_transport(command, args)
    await transport.start()
    assert await drain(transport) == [b"out"]
    await transport.close()
    assert log_path.read_text() == "diagnostics\n"


async def test_a_second_run_appends_to_what_the_first_wrote(make_transport, log_path):
    for text in ("first\n", "second\n"):
        command, args = python_child(_WRITE_TO_STDERR, text)
        transport = make_transport(command, args)
        await transport.start()
        await drain(transport)
        await transport.close()
    assert log_path.read_text() == "first\nsecond\n"


async def test_the_log_is_capped_at_one_mebibyte_and_keeps_the_newest_bytes(
    make_transport, log_path
):
    source = (
        "import sys\n"
        "for n in range(1_500_000 // 100):\n"
        "    sys.stderr.write(f'line {n:08d} ' + 'x' * 86 + '\\n')\n"
        "sys.stderr.write('THE-LAST-LINE\\n')\n"
    )
    command, args = python_child(source)
    transport = make_transport(command, args)
    await transport.start()
    await drain(transport)
    await transport.close()
    data = log_path.read_bytes()
    assert STDERR_LOG_CAP == 1024 * 1024
    assert len(data) <= STDERR_LOG_CAP, f"the log is {len(data)} bytes"
    assert len(data) > STDERR_LOG_CAP - 200, "the log was cut far below its cap"
    assert data.endswith(b"THE-LAST-LINE\n"), data[-60:]
    assert b"line 00000000 " not in data, "the oldest bytes were kept instead of the newest"


async def test_the_log_is_private_and_so_is_the_directory_made_for_it(make_transport, log_path):
    command, args = python_child(_WRITE_TO_STDERR, "x")
    transport = make_transport(command, args)
    await transport.start()
    await drain(transport)
    await transport.close()
    assert log_path.stat().st_mode & 0o777 == 0o600
    assert log_path.parent.stat().st_mode & 0o777 == 0o700


#: More than a pipe holds and more than the stream behind it buffers (two limits, 8 MiB): a
#: server that writes this much to stderr stops when nobody reads it.
_MORE_STDERR_THAN_ANYTHING_BUFFERS = (
    "import sys\n"
    "for _ in range(160):\n"
    "    sys.stderr.write('x' * 65_536)\n"
    "sys.stdout.write('done\\n')\n"
)


async def test_a_log_that_cannot_be_written_does_not_stop_the_server(make_transport, tmp_path):
    unusable = tmp_path / "a-directory-where-the-file-should-be"
    unusable.mkdir()
    command, args = python_child(_MORE_STDERR_THAN_ANYTHING_BUFFERS)
    transport = make_transport(command, args, log_path=unusable)
    await transport.start()
    assert await asyncio.wait_for(drain(transport), timeout=10) == [b"done"]
    await transport.close()


async def test_a_log_that_fails_after_it_was_opened_does_not_stop_the_server(
    make_transport, monkeypatch
):
    def full_disk(self: CappedLog, data: bytes) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(CappedLog, "append", full_disk)
    command, args = python_child(_MORE_STDERR_THAN_ANYTHING_BUFFERS)
    transport = make_transport(command, args)
    await transport.start()
    assert await asyncio.wait_for(drain(transport), timeout=10) == [b"done"]
    await transport.close()


class TestCappedLog:
    def test_what_fits_is_appended(self, tmp_path):
        log = CappedLog(tmp_path / "x.log", cap=10)
        log.append(b"abc")
        log.append(b"def")
        log.close()
        assert (tmp_path / "x.log").read_bytes() == b"abcdef"

    def test_what_does_not_fit_costs_the_oldest_bytes(self, tmp_path):
        log = CappedLog(tmp_path / "x.log", cap=10)
        log.append(b"0123456")
        log.append(b"789abc")
        log.close()
        assert (tmp_path / "x.log").read_bytes() == b"3456789abc"

    def test_a_chunk_bigger_than_the_cap_keeps_its_own_tail(self, tmp_path):
        log = CappedLog(tmp_path / "x.log", cap=4)
        log.append(b"ab")
        log.append(b"0123456789")
        log.close()
        assert (tmp_path / "x.log").read_bytes() == b"6789"

    def test_a_file_already_over_the_cap_is_cut_when_next_written(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_bytes(b"0123456789")
        log = CappedLog(path, cap=6)
        log.append(b"ab")
        log.close()
        assert path.read_bytes() == b"6789ab"

    def test_a_file_that_is_there_is_appended_to(self, tmp_path):
        path = tmp_path / "x.log"
        path.write_bytes(b"old\n")
        log = CappedLog(path, cap=100)
        log.append(b"new\n")
        log.close()
        assert path.read_bytes() == b"old\nnew\n"

    def test_nothing_can_be_appended_after_it_is_closed(self, tmp_path):
        log = CappedLog(tmp_path / "x.log", cap=10)
        log.close()
        log.close()
        with pytest.raises(ValueError, match="closed"):
            log.append(b"x")

    def test_a_file_exactly_at_the_cap_is_whole(self, tmp_path):
        log = CappedLog(tmp_path / "x.log", cap=6)
        log.append(b"abc")
        log.append(b"def")
        log.close()
        assert (tmp_path / "x.log").read_bytes() == b"abcdef"


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


def test_the_grace_periods_are_two_seconds_each():
    assert STDIN_GRACE_SECONDS == 2.0
    assert TERM_GRACE_SECONDS == 2.0


async def test_a_server_that_ends_at_the_end_of_stdin_is_never_signalled(make_transport):
    transport = make_transport("/bin/cat")
    await transport.start()
    pid = transport.pid
    assert pid is not None
    await transport.close()
    assert transport.returncode == 0, f"cat was ended by {transport.returncode}, not by its stdin"
    assert group_is_gone(pid)


async def test_closing_twice_and_closing_at_once_are_both_fine(make_transport):
    transport = make_transport("/bin/cat")
    await transport.start()
    await asyncio.gather(transport.close(), transport.close())
    await transport.close()


async def test_closing_a_transport_that_never_started_is_fine(make_transport):
    await make_transport("/bin/cat").close()


async def test_nothing_can_be_sent_or_started_after_closing(make_transport):
    transport = make_transport("/bin/cat")
    await transport.start()
    await transport.close()
    with pytest.raises(TransportError, match="closed"):
        await transport.send(PING)
    assert await transport.receive() is None
    with pytest.raises(TransportError, match="closed"):
        await transport.start()


async def test_closing_leaves_no_descriptor_open(make_transport):
    # Both ends of every pipe, and the transport that wraps them: a leaked one is an error
    # on 3.13 and a slow leak on every version.
    before = open_descriptors()
    transport = make_transport("/bin/cat")
    await transport.start()
    assert open_descriptors() > before
    await transport.close()
    assert open_descriptors() == before


async def test_stdin_is_closed_before_the_server_is_sent_a_signal(make_transport, log_path):
    command, args = fake_server("ignore-stdin-eof")
    grace = 0.5
    transport = make_transport(command, args, stdin_grace=grace, term_grace=5.0)
    await transport.start()
    await when_ready(transport)
    await transport.close()
    seen = events(log_path)
    assert "stdin-eof" in seen and "sigterm" in seen, seen
    assert seen["stdin-eof"] < seen["sigterm"], f"the signal came first: {seen}"
    waited = seen["sigterm"] - seen["stdin-eof"]
    assert waited >= grace - 0.1, f"only {waited:.2f}s between the end of stdin and SIGTERM"
    assert transport.returncode == 0, "the server did not leave on SIGTERM"


async def test_a_server_that_ignores_both_is_killed_and_so_is_everything_it_started(
    make_transport, log_path
):
    command, args = fake_server("ignore-term")
    transport = make_transport(command, args, stdin_grace=0.3, term_grace=0.3)
    await transport.start()
    await when_ready(transport)
    pgid, helper = transport.pid, logged_helper_pid(log_path)
    assert pgid is not None and not group_is_gone(pgid)
    began = time.monotonic()
    await transport.close()
    elapsed = time.monotonic() - began
    assert "sigterm-ignored" in events(log_path), "SIGTERM was never sent"
    assert transport.returncode == -signal.SIGKILL, transport.returncode
    assert group_is_gone(pgid), "a process is still in the server's group"
    assert process_is_gone(helper), "the helper the server started is still running"
    assert elapsed >= 0.6 - 0.05, f"closed in {elapsed:.2f}s, without waiting out both periods"


async def test_a_helper_left_behind_by_a_server_that_exited_is_ended_too(make_transport, log_path):
    command, args = fake_server("orphan-helper")
    transport = make_transport(command, args, stdin_grace=0.2, term_grace=0.2)
    await transport.start()
    await when_ready(transport)
    pgid, helper = transport.pid, logged_helper_pid(log_path)
    assert pgid is not None
    await transport.close()
    assert transport.returncode == 0, "the server itself was signalled"
    assert group_is_gone(pgid), "the helper outlived the server"
    assert process_is_gone(helper)


async def test_a_close_that_is_cancelled_still_ends_the_server(make_transport, log_path):
    command, args = fake_server("ignore-term")
    transport = make_transport(command, args, stdin_grace=30.0, term_grace=30.0)
    await transport.start()
    await when_ready(transport)
    pgid = transport.pid
    assert pgid is not None
    closing = asyncio.create_task(transport.close())
    await asyncio.sleep(0)  # let it begin: it is now waiting out the first period
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert await eventually(lambda: group_is_gone(pgid))
    assert group_is_gone(pgid), "cancelling close left the server running"
