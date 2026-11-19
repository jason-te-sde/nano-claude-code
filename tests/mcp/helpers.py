# tests/mcp/helpers.py
"""What the MCP tests share: how to start the fake server, and how to read what it logged."""

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


def logged_helper_pid(log: Path) -> int:
    for text in log.read_text(encoding="utf-8").splitlines():
        if text.startswith("HELPER "):
            return int(text.split()[1])
    raise AssertionError(f"the fake server logged no helper:\n{log.read_text(encoding='utf-8')}")


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
