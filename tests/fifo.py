"""A FIFO, and a way to find out that opening one would have blocked without blocking the run.

Opening a FIFO for reading waits until somebody opens it for writing, so code that opens
whatever path it is given hangs on one, and a test of that code that simply calls it hangs
the whole run with it. ``call_without_blocking`` gives the call a deadline, and when the
call has not returned by then it lets it go and fails the test.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

_T = TypeVar("_T")

#: Long enough that a call which is only slow is not taken for one that is blocked.
DEADLINE_S = 5.0


def make_fifo(path: Path) -> Path:
    os.mkfifo(path)
    return path


def release(path: Path) -> None:
    """Let whoever is blocked opening ``path`` for reading go, by opening it for writing.

    The reader then finds the end of the file at once. Nobody waiting is not an error.
    """
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:  # ENXIO: nothing has it open for reading
        return
    os.close(descriptor)


def call_without_blocking(fifo: Path, function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """``function(*args, **kwargs)``, run in a thread that is given ``DEADLINE_S`` to finish.

    What it returns is returned and what it raises is raised. If it is still running when the
    time is up it was blocked on ``fifo``: it is let go, so that the run can end, and the test
    fails.
    """
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = function(*args, **kwargs)
        except BaseException as exc:
            outcome["raised"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(DEADLINE_S)
    if thread.is_alive():
        release(fifo)
        thread.join(DEADLINE_S)
        pytest.fail(f"{getattr(function, '__name__', function)} blocked on a FIFO")
    if "raised" in outcome:
        raise outcome["raised"]
    return outcome["value"]  # type: ignore[no-any-return]
