# src/nanoclaude/mcp/transport.py
"""Running an MCP server as a child process and speaking to it a line at a time.

A server is somebody else's program, started with the user's authority, so what it is
given and what it is left behind with are decided here and not by the caller:

**It is given little.** The program is started directly, never through a shell, in the
project root, with an environment of nine variables from this process (:data:`ENV_ALLOWLIST`)
and whatever the user wrote in the server's ``env``. Nothing else of this process's
environment reaches it, so a provider's API key is not in a server's hands unless the user
put it there.

**It is led by nobody but us.** The child is the leader of a process group of its own, so
that one signal reaches it and everything it started, and so that a Ctrl-C at the terminal,
which goes to this process's group, does not reach it before :meth:`StdioTransport.close`
has had its say.

**It is never left running.** :meth:`StdioTransport.close` closes the server's stdin and
gives it :data:`STDIN_GRACE_SECONDS` to finish, then sends SIGTERM to the group and waits
:data:`TERM_GRACE_SECONDS`, then SIGKILL. "Finish" means that nothing is left in the group,
not that the process we started has exited: a worker the server forked and forgot is ended
too. It is idempotent, and a close that is itself cancelled kills the group before it lets
the cancellation go on.

What the server writes to stderr is not shown to the model. It is appended to a log file that
holds at most :data:`STDERR_LOG_CAP` bytes, the newest ones, and a log that cannot be written
costs the server nothing: stderr is still read, so that a server that writes a lot of it
cannot block on a full pipe.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from nanoclaude.mcp.jsonrpc import MAX_LINE_BYTES, LineTooLongError
from nanoclaude.private import create_private_file, make_private_directories

#: The only names of this process's environment a server is given (spec 3.2). Its own ``env``
#: is added to these, and wins over them.
ENV_ALLOWLIST = (
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

#: The most the stderr log of one server holds.
STDERR_LOG_CAP = 1024 * 1024

#: How long a server has to finish once its stdin is closed, and then once it has been sent
#: SIGTERM, before the next step.
STDIN_GRACE_SECONDS = 2.0
TERM_GRACE_SECONDS = 2.0

#: How long to wait for a group to disappear after SIGKILL, which cannot be ignored: this is
#: for the kernel to reap it.
_KILL_WAIT_SECONDS = 2.0

#: How long to wait for the end of stderr once the group is gone.
_STDERR_DRAIN_SECONDS = 2.0

#: How often a group is looked at while waiting for it to empty. Nothing can be awaited for
#: that: a process that is not our child cannot be waited for.
_POLL_SECONDS = 0.01

_STDERR_CHUNK = 64 * 1024


class TransportError(Exception):
    """The server could not be started, or its pipes can no longer be used."""


class Transport(Protocol):
    """What a client needs of the way it reaches a server: lines in, lines out, an end.

    Only :class:`StdioTransport` exists. Streamable HTTP is not built, and when it is, it is
    another implementation of this.
    """

    async def start(self) -> None: ...

    async def send(self, line: bytes) -> None:
        """Write one message, which holds no newline."""
        ...

    async def receive(self) -> bytes | None:
        """The next message, or ``None`` once the server will send no more."""
        ...

    async def close(self) -> None:
        """End the server and release everything of it. Idempotent."""
        ...


def build_environment(
    configured: Mapping[str, str], parent: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment of a server: the allowlisted names ``parent`` has, then ``configured``.

    ``parent`` is this process's environment unless a caller says otherwise.
    """
    source = os.environ if parent is None else parent
    built = {name: source[name] for name in ENV_ALLOWLIST if name in source}
    built.update(configured)
    return built


class CappedLog:
    """A file that is appended to and never holds more than ``cap`` bytes: the newest.

    The file is private from the moment it exists. Writes are small and synchronous: a
    server's stderr is a trickle, and the cost of the one write that has to rewrite the file
    is a megabyte.
    """

    def __init__(self, path: Path, cap: int = STDERR_LOG_CAP) -> None:
        make_private_directories(path.parent)
        create_private_file(path)
        self._cap = cap
        # O_APPEND: whatever else has the file open, a write lands at its end. O_NOFOLLOW: a
        # link put where the log should be is not written through.
        self._descriptor: int | None = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)

    def append(self, data: bytes) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            raise ValueError("the log is closed")
        size = os.fstat(descriptor).st_size
        if size + len(data) <= self._cap:
            self._write(descriptor, data)
            return
        room = self._cap - len(data)
        kept = os.pread(descriptor, size, 0)[-room:] if room > 0 else b""
        os.ftruncate(descriptor, 0)
        self._write(descriptor, kept + data[-self._cap :])

    @staticmethod
    def _write(descriptor: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]

    def close(self) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None


class StdioTransport:
    """An MCP server's stdin and stdout as lines, and the process behind them.

    Not safe to share between tasks that both ``receive``; any number may ``send``.
    """

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str,
        log_path: Path,
        env: Mapping[str, str] | None = None,
        stdin_grace: float = STDIN_GRACE_SECONDS,
        term_grace: float = TERM_GRACE_SECONDS,
    ) -> None:
        self._command = command
        self._args = tuple(args)
        self._cwd = cwd
        self._log_path = log_path
        self._env = dict(env or {})
        self._stdin_grace = stdin_grace
        self._term_grace = term_grace
        self._process: asyncio.subprocess.Process | None = None
        self._log: CappedLog | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def pid(self) -> int | None:
        """The server's process id, which is also the id of its process group."""
        return self._process.pid if self._process is not None else None

    @property
    def returncode(self) -> int | None:
        """How the server ended: its exit status, or minus the signal that ended it."""
        return self._process.returncode if self._process is not None else None

    async def start(self) -> None:
        if self._closing:
            raise TransportError("the transport is closed")
        if self._started:
            raise TransportError("the transport was already started")
        self._started = True
        try:
            self._log = CappedLog(self._log_path)
        except OSError:
            self._log = None  # the server runs without a log; its stderr is still read
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                env=build_environment(self._env),
                start_new_session=True,
                limit=MAX_LINE_BYTES,
            )
        except (OSError, ValueError) as exc:
            raise TransportError(f"cannot start {self._command!r}: {exc}") from exc
        assert self._process.stderr is not None  # noqa: S101 - PIPE was asked for
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._process.stderr))

    def _live_process(self) -> asyncio.subprocess.Process:
        if self._closing:
            raise TransportError("the transport is closed")
        if self._process is None:
            raise TransportError("the transport was not started")
        return self._process

    async def send(self, line: bytes) -> None:
        """Write ``line`` and the newline that ends it. ``line`` must hold no newline."""
        process = self._live_process()
        assert process.stdin is not None  # noqa: S101 - PIPE was asked for
        try:
            # One call to write(), so that two tasks sending at once cannot mix their lines.
            process.stdin.write(line + b"\n")
            await process.stdin.drain()
        except OSError as exc:  # BrokenPipeError and ConnectionResetError are OSErrors
            raise TransportError(f"the server is not reading its input: {exc}") from exc

    async def receive(self) -> bytes | None:
        """The next line the server wrote, without its newline; ``None`` once it has no more.

        Raises :class:`~nanoclaude.mcp.jsonrpc.LineTooLongError` for a line over the limit,
        without waiting for the rest of it.
        """
        if self._closed:
            return None
        process = self._live_process()
        assert process.stdout is not None  # noqa: S101 - PIPE was asked for
        try:
            line = await process.stdout.readline()
        except ValueError as exc:  # the stream's own limit, set to the line limit
            raise LineTooLongError(
                f"the server wrote a line of more than {MAX_LINE_BYTES} bytes"
            ) from exc
        return line.removesuffix(b"\n") if line else None

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        while chunk := await stderr.read(_STDERR_CHUNK):
            if self._log is not None:
                try:
                    self._log.append(chunk)
                except OSError:
                    self._log.close()
                    self._log = None  # keep reading: a full pipe would stop the server

    async def close(self) -> None:
        """End the server, in the order the module docstring gives, and leave nothing of it."""
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            try:
                await self._shutdown()
            except asyncio.CancelledError:
                self._signal_group(signal.SIGKILL)
                raise
            self._closed = True

    async def _shutdown(self) -> None:
        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if not await self._group_gone(process, self._stdin_grace):
                self._signal_group(signal.SIGTERM)
                if not await self._group_gone(process, self._term_grace):
                    self._signal_group(signal.SIGKILL)
                    await self._group_gone(process, _KILL_WAIT_SECONDS)
        await self._finish_stderr()
        if self._log is not None:
            self._log.close()
            self._log = None

    async def _group_gone(self, process: asyncio.subprocess.Process, seconds: float) -> bool:
        """Wait for the server to be reaped and its group to empty; False if time runs out."""
        deadline = time.monotonic() + seconds
        try:
            await asyncio.wait_for(process.wait(), seconds)
        except TimeoutError:
            return False
        while self._group_alive(process.pid):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_POLL_SECONDS)
        return True

    @staticmethod
    def _group_alive(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # a member of the group that is not ours: it is there
            return True
        return True

    def _signal_group(self, sig: signal.Signals) -> None:
        if self._process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._process.pid, sig)

    async def _finish_stderr(self) -> None:
        task, self._stderr_task = self._stderr_task, None
        if task is None:
            return
        # The end of stderr comes when the last holder of the pipe is gone, which after the
        # kill is everything in the group. A process that left the group holding the pipe
        # is not ours to end, and must not hold this up.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(task, _STDERR_DRAIN_SECONDS)
