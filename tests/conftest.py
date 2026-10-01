"""Fixtures shared by the whole suite."""

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
