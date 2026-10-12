"""Fixtures shared by the whole suite."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoclaude.conversation.store import Store
from nanoclaude.permissions.policy import PermissionMode, Policy
from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS, Redactor
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox
from nanoclaude.tools.base import ToolContext


@pytest.fixture(autouse=True)
def _close_every_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Close every Store a test opened, when the test ends.

    Python 3.13 emits a ResourceWarning when a sqlite3 connection is garbage
    collected unclosed, and this suite turns warnings into errors. The warning
    fires at collection time, so it is reported against whichever later test
    happens to be running -- a leak in one module surfaces as a failure in an
    unrelated one. Closing deterministically keeps the attribution honest.
    """
    opened: list[Store] = []
    real_open = Store.open

    def tracking_open(self: Store) -> None:
        real_open(self)
        opened.append(self)

    monkeypatch.setattr(Store, "open", tracking_open)
    yield
    for store in opened:
        store.close()


@pytest.fixture
def tmp_repo(tmp_path):
    """A working directory that looks enough like a project."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("node_modules/\n*.pyc\n")
    return tmp_path


@pytest.fixture
def policy(tmp_repo: Path) -> Policy:
    return Policy(
        sandbox=Sandbox((str(tmp_repo),)),
        rules=RuleSet.build(allow=["Read", "Grep", "Glob"], ask=["Bash", "Write", "Edit"]),
        mode=PermissionMode.DEFAULT,
        secret_paths=SECRET_PATH_PATTERNS,
    )


@pytest.fixture
def ctx(tmp_repo: Path, policy: Policy) -> ToolContext:
    return ToolContext(
        sandbox=policy.sandbox,
        policy=policy,
        redactor=Redactor(),
        read_state={},
        root=str(tmp_repo),
    )


@pytest.fixture
def private_from_the_start(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A process in which a file or directory can only come out private if it was made so.

    The umask lets everything through and no chmod of any kind does anything, so a mode that
    is private afterwards is the mode the call that *created* it asked for, not one that
    was set on it later, which would leave it open in between.
    """
    previous = os.umask(0)
    monkeypatch.setattr(os, "chmod", lambda *_a, **_k: None)
    monkeypatch.setattr(os, "fchmod", lambda *_a, **_k: None)
    monkeypatch.setattr(Path, "chmod", lambda *_a, **_k: None)
    try:
        yield
    finally:
        os.umask(previous)


def mode_of(path: Path) -> int:
    """The permission bits of ``path``, as the octal number a person reads."""
    return path.stat().st_mode & 0o777
