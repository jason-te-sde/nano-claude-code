# tests/mcp/helpers.py
"""What the MCP tests share: how to start the fake server, and how to read what it logged."""

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanoclaude.config.schema import (
    Config,
    LimitsConfig,
    McpServerConfig,
    ModelConfig,
    RolesConfig,
)

FAKE_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "fake_mcp_server.py"


def fake_server(mode: str, *settings: str) -> tuple[str, list[str]]:
    """The command and arguments that start the fake server in ``mode``.

    ``-I -S`` start the interpreter without the site machinery: the server needs only the
    standard library, and the suite starts one for most of its tests.
    """
    return sys.executable, ["-I", "-S", str(FAKE_SERVER), mode, *settings]


def python_child(source: str, *args: str) -> tuple[str, list[str]]:
    """The command and arguments that run ``source`` as a one-off child."""
    return sys.executable, ["-I", "-S", "-c", source, *args]


def received(log: Path) -> list[dict[str, Any]]:
    """Every message the fake server logged as received, in order."""
    messages = []
    for text in log.read_text(encoding="utf-8").splitlines():
        tag, _, rest = text.partition(" ")
        if tag == "RECV":
            messages.append(json.loads(rest))
    return messages


def received_methods(log: Path) -> list[str]:
    return [message["method"] for message in received(log) if "method" in message]


def events(log: Path) -> dict[str, float]:
    """The lifecycle events the fake server logged, by name, with the time of the last."""
    found: dict[str, float] = {}
    for text in log.read_text(encoding="utf-8").splitlines():
        tag, _, rest = text.partition(" ")
        if tag == "EVENT":
            name, _, stamp = rest.partition(" ")
            found[name] = float(stamp) if stamp else 0.0
    return found


def logged_server_pids(log: Path) -> list[int]:
    """The process id the fake server logged each time it was started: its process group's too.

    Empty when there is no log, which is what a server that was never started leaves.
    """
    if not log.exists():
        return []
    return [
        int(text.split()[1])
        for text in log.read_text(encoding="utf-8").splitlines()
        if text.startswith("PID ")
    ]


def logged_helper_pid(log: Path) -> int:
    for text in log.read_text(encoding="utf-8").splitlines():
        if text.startswith("HELPER "):
            return int(text.split()[1])
    raise AssertionError(f"the fake server logged no helper:\n{log.read_text(encoding='utf-8')}")


async def helper_pid(log: Path) -> int:
    """The id the fake server logged for its helper, once the log holds it.

    The log is written by the transport as stderr arrives, which is not at the same moment as
    anything the server wrote to stdout.
    """

    def logged() -> bool:
        return log.exists() and "HELPER " in log.read_text(encoding="utf-8")

    assert await eventually(logged), "the fake server logged no helper"
    return logged_helper_pid(log)


def kill_helpers(log: Path) -> None:
    """End every helper the fake server logged: one that left the group is nobody else's to end."""
    if not log.exists():
        return
    for text in log.read_text(encoding="utf-8").splitlines():
        if text.startswith("HELPER "):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(text.split()[1]), signal.SIGKILL)


def group_is_gone(pgid: int) -> bool:
    """True when no process is left in the group, which is what ``killpg`` with 0 says.

    macOS answers "not permitted" for a group whose only members are zombies that nobody has
    reaped yet; that is a group that is not gone yet, and a poll sees it go.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def eventually(condition: Callable[[], bool], *, within: float = 5.0) -> bool:
    """Poll ``condition`` until it holds or ``within`` seconds pass; for what is reaped by a
    thread or the kernel after the call that caused it has returned."""
    deadline = time.monotonic() + within
    while not condition():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def open_descriptors() -> int:
    """How many file descriptors this process has open."""
    return len(list(Path("/dev/fd").iterdir()))


def server(name: str, mode: str = "modern", *settings: str, **fields: Any) -> McpServerConfig:
    """A configured server that is the fake server in ``mode``."""
    command, args = fake_server(mode, *settings)
    return McpServerConfig(name=name, command=command, args=tuple(args), **fields)


def log_of(home: Path, name: str) -> Path:
    """Where the stderr of the server called ``name`` is kept, under ``home``."""
    return home / ".nanoclaude" / "logs" / f"mcp-{name}.log"


def pids_of(home: Path, name: str) -> list[int]:
    """The process ids the server called ``name`` logged: one for each time it was started."""
    return logged_server_pids(log_of(home, name))


async def all_gone(pids: list[int]) -> bool:
    """Whether every group is gone, once the kernel and the loop have reaped what ended."""
    assert pids, "no server was started, so nothing was checked for being gone"
    return await eventually(lambda: all(group_is_gone(pid) for pid in pids))


def config_of(*servers: McpServerConfig, cap: int = 100_000) -> Config:
    """A configuration that has ``servers`` and a model for every role, and nothing else."""
    return Config(
        models={"m": ModelConfig("anthropic", "claude-sonnet-5")},
        roles=RolesConfig("m", "m", "m", "m", "m", "m"),
        limits=LimitsConfig(output_cap_bytes=cap),
        mcp={server.name: server for server in servers},
    )
