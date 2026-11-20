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
the cancellation go on. When it returns, our ends of the three pipes and the transport that
owns them are closed, whatever the server's descendants still hold: asyncio's own wait for
a process does not end while a pipe is open, so nothing here waits for it, and a server that
floods its stdout after nobody reads it, or one whose child left the group holding a pipe,
cannot keep a descriptor open or turn into a ResourceWarning.

**It is noticed when it is gone.** :meth:`StdioTransport.receive` says there is no more
when the server's stdout ends, and also when the server has exited and nothing more arrives
in a moment, which is what a descendant holding stdout open would otherwise hide until every
request had run out its own timeout.

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

#: How long to wait for the end of stderr once the group is gone. Its end comes when the last
#: holder of the pipe lets go, which is at once unless a process left the group holding it.
STDERR_DRAIN_SECONDS = 2.0

#: How often :meth:`StdioTransport.receive` looks to see whether the server has exited while
#: it waits for a line, and how long it then gives what the server wrote before it went.
_EXIT_POLL_SECONDS = 0.05
_EXIT_SETTLE_SECONDS = 0.2

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
        drain_grace: float = STDERR_DRAIN_SECONDS,
    ) -> None:
        self._command = command
        self._args = tuple(args)
        self._cwd = cwd
        self._log_path = log_path
        self._env = dict(env or {})
        self._stdin_grace = stdin_grace
        self._term_grace = term_grace
        self._drain_grace = drain_grace
        self._process: asyncio.subprocess.Process | None = None
        self._log: CappedLog | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self._closed = False
        #: True once the server's group is known to be empty. The group id is the server's pid,
        #: and from then on the number can belong to any process at all.
        self._group_ended = False
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

        That is the end of its stdout, or, when a process the server left behind holds the pipe
        open, the server having exited and nothing more having come in a moment. Raises
        :class:`~nanoclaude.mcp.jsonrpc.LineTooLongError` for a line over the limit, without
        waiting for the rest of it.
        """
        if self._closed:
            return None
        process = self._live_process()
        assert process.stdout is not None  # noqa: S101 - PIPE was asked for
        read = asyncio.ensure_future(process.stdout.readline())
        try:
            while not read.done():
                await asyncio.wait({read}, timeout=_EXIT_POLL_SECONDS)
                if not read.done() and process.returncode is not None:
                    await asyncio.wait({read}, timeout=_EXIT_SETTLE_SECONDS)
                    if not read.done():
                        return None
            line = read.result()
        except ValueError as exc:  # the stream's own limit, set to the line limit
            raise LineTooLongError(
                f"the server wrote a line of more than {MAX_LINE_BYTES} bytes"
            ) from exc
        finally:
            read.cancel()  # does nothing to a read that is done
        return line.removesuffix(b"\n") if line else None

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        while chunk := await stderr.read(_STDERR_CHUNK):
            if self._log is not None:
                try:
                    self._log.append(chunk)
                except OSError:
                    self._log.close()
                    self._log = None  # keep reading: a full pipe would stop the server
            # A read that finds data waiting does not give up the loop, and a server that
            # writes megabytes of stderr would hold it for as long as writing them to the log
            # takes: every other server's answers would wait behind it.
            await asyncio.sleep(0)

    async def close(self) -> None:
        """End the server, in the order the module docstring gives, and leave nothing of it."""
        async with self._close_lock:
            if self._closed:
                # More than tidiness: the group id is the server's pid, and once the group is
                # gone that number can be anybody's. A second look at it would be a look at a
                # stranger's group, and a signal to it a signal to a stranger.
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
        if process is not None:
            await self._close_pipes(process)
        if self._log is not None:
            self._log.close()
            self._log = None

    async def _group_gone(self, process: asyncio.subprocess.Process, seconds: float) -> bool:
        """Wait for the server to be reaped and its group to empty; False if time runs out.

        Polled, and not awaited with ``process.wait()``: asyncio's wait, when it is started
        before the process exits, ends only when every pipe has closed, and a pipe whose
        reading was paused (a server that floods stdout) or that a descendant holds open never
        does. The return code is set when the child is reaped, whatever the pipes are doing.
        """
        deadline = time.monotonic() + seconds
        while process.returncode is None or self._group_alive(process.pid):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_POLL_SECONDS)
        self._group_ended = True
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

    def kill(self) -> None:
        """Send SIGKILL to the server's group now, and wait for nothing.

        For a caller that has run out of time to close in the order the module docstring
        gives: SIGKILL cannot be ignored, so nothing the server or a helper of it started
        survives it. It does not release the pipes; :meth:`close` after it does that, and
        finds nothing left to wait for. Nothing is signalled before the server was started
        or once its group is known to be gone.
        """
        self._signal_group(signal.SIGKILL)

    def _signal_group(self, sig: signal.Signals) -> None:
        if self._process is None or self._group_ended:
            return
        # macOS answers a group of nothing but zombies with "not permitted" and not "no such
        # process", and the end of a close must not depend on which.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self._process.pid, sig)

    @staticmethod
    async def _close_pipes(process: asyncio.subprocess.Process) -> None:
        """Close our end of every pipe and the transport that owns them.

        Nothing else does it when a pipe is still open at the other end: a descendant that left
        the group holds stdout and stderr, and a stdout that nobody reads is paused with its
        buffer full. ``Process`` has no ``close()``, so this reaches for its transport; closing
        that closes the pipes, and kills the leader if the steps before could not end it. A
        leader killed here is waited for, because the loop learns that a child has gone from a
        thread that waits on it, and a ``Popen`` whose exit the loop never heard of is a
        ResourceWarning when it is collected. The pipe transports close their descriptors in
        callbacks that closing scheduled, so the loop is then given two turns to run them.
        """
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()
        deadline = time.monotonic() + _KILL_WAIT_SECONDS
        while process.returncode is None:
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(_POLL_SECONDS)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def _finish_stderr(self) -> None:
        task, self._stderr_task = self._stderr_task, None
        if task is None:
            return
        # The end of stderr comes when the last holder of the pipe is gone, which after the
        # kill is everything in the group. A process that left the group holding the pipe
        # is not ours to end, and must not hold this up.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(task, self._drain_grace)
