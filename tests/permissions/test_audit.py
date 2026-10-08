"""Tests for nanoclaude.permissions.audit.

The credential sample below is assembled from parts at import time rather than
written as a contiguous literal: this repository is public, and GitHub push
protection rejects a push containing a string shaped like a live credential,
even a fake one (the same convention tests/permissions/test_redact.py uses).
"""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.conversation.store import Store
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.policy import Decision
from nanoclaude.permissions.redact import Redactor

GITHUB_TOKEN = "ghp_" + "A" * 36


def audit(tmp_path: Path) -> AuditLog:
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    return AuditLog(store)


def test_the_decision_is_recorded_before_the_tool_runs(tmp_path):
    """A crash mid-execution must still leave the decision on disk."""
    log = audit(tmp_path)
    log.record_decision(
        "s1",
        turn=0,
        tool_use_id="t1",
        tool="Bash",
        arguments={"command": "ls"},
        decision=Decision.ALLOW,
        rule="rule.allow",
    )
    row = log.store.db.execute("SELECT * FROM tool_calls").fetchone()
    assert (row["decision"], row["rule"], row["outcome"]) == ("allow", "rule.allow", None)


def test_the_outcome_fills_in_the_same_row(tmp_path):
    log = audit(tmp_path)
    log.record_decision(
        "s1",
        turn=0,
        tool_use_id="t1",
        tool="Bash",
        arguments={"command": "ls"},
        decision=Decision.ALLOW,
        rule="rule.allow",
    )
    log.record_outcome("s1", "t1", outcome="ok", duration_ms=12, bytes_out=40, error=None)
    rows = log.store.db.execute("SELECT * FROM tool_calls").fetchall()
    assert len(rows) == 1
    assert (rows[0]["outcome"], rows[0]["duration_ms"]) == ("ok", 12)


def test_a_denied_call_has_no_outcome_and_that_is_queryable(tmp_path):
    log = audit(tmp_path)
    log.record_decision(
        "s1",
        turn=0,
        tool_use_id="t1",
        tool="Read",
        arguments={"path": "/etc/passwd"},
        decision=Decision.DENY,
        rule="sandbox.outside-root",
    )
    denied = log.store.db.execute(
        "SELECT COUNT(*) AS n FROM tool_calls WHERE decision = 'deny' AND outcome IS NOT NULL"
    ).fetchone()
    assert denied["n"] == 0


def test_arguments_are_stored_redacted(tmp_path):
    """The audit table is a place secrets could survive redaction. It must not."""
    log = audit(tmp_path)
    log.record_decision(
        "s1",
        turn=0,
        tool_use_id="t1",
        tool="Bash",
        arguments={"command": "deploy --token " + GITHUB_TOKEN},
        decision=Decision.ASK,
        rule="default.ask",
    )
    row = log.store.db.execute("SELECT args_json FROM tool_calls").fetchone()
    assert "ghp_" not in row["args_json"]
    assert "redacted" in row["args_json"]


def test_a_custom_redactor_is_used_instead_of_the_default(tmp_path):
    # Proves the constructor actually wires in the redactor it is given,
    # rather than always scrubbing with a Redactor() of its own: a disabled
    # redactor must leave the secret right there in args_json.
    store = Store(tmp_path / "s.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    log = AuditLog(store, redactor=Redactor(enabled=False))

    log.record_decision(
        "s1",
        turn=0,
        tool_use_id="t1",
        tool="Bash",
        arguments={"command": "deploy --token " + GITHUB_TOKEN},
        decision=Decision.ALLOW,
        rule="rule.allow",
    )
    row = log.store.db.execute("SELECT args_json FROM tool_calls").fetchone()
    assert GITHUB_TOKEN in row["args_json"]


def _decision(log: AuditLog, call_id: str, *, session: str = "s1", **fields: object) -> None:
    given: dict[str, Any] = {
        "turn": 0,
        "tool_use_id": call_id,
        "tool": "Bash",
        "arguments": {"command": "ls"},
        "decision": Decision.ALLOW,
        "rule": "rule.allow",
    } | fields
    log.record_decision(session, **given)


def test_a_repeated_tool_use_id_is_an_error_and_the_first_decision_stands(tmp_path):
    """The audit trail only grows. Minted ids are unique, so a second decision under one is a
    bug somewhere else, and replacing the first row would hide it and the decision with it."""
    log = audit(tmp_path)
    _decision(log, "t1", decision=Decision.DENY, rule="rule.deny", arguments={"command": "rm"})
    with pytest.raises(sqlite3.IntegrityError):
        _decision(log, "t1", decision=Decision.ALLOW, rule="rule.allow")
    row = log.store.db.execute("SELECT decision, rule, args_json FROM tool_calls").fetchone()
    assert (row["decision"], row["rule"], row["args_json"]) == (
        "deny",
        "rule.deny",
        '{"command": "rm"}',
    )


def test_the_same_id_in_another_session_is_another_call(tmp_path):
    log = audit(tmp_path)
    log.store.create_session("s2", cwd="/p", roles={})
    _decision(log, "t1", session="s1")
    _decision(log, "t1", session="s2")
    count = log.store.db.execute("SELECT COUNT(*) AS n FROM tool_calls").fetchone()["n"]
    assert count == 2


def test_an_outcome_still_fills_in_the_row_it_belongs_to(tmp_path):
    """Insert-only does not mean the outcome is a second row."""
    log = audit(tmp_path)
    _decision(log, "t1")
    _decision(log, "t2")
    log.record_outcome("s1", "t2", outcome="cancelled", duration_ms=0, bytes_out=0, error=None)
    rows = {
        r["tool_use_id"]: r["outcome"] for r in log.store.db.execute("SELECT * FROM tool_calls")
    }
    assert rows == {"t1": None, "t2": "cancelled"}
