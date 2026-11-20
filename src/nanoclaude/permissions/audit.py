"""Append-only record of every tool call and what was decided about it.

The decision is written *before* the tool runs and the outcome fills in the same
row afterwards. A crash between the two leaves a row saying "this was allowed
and we do not know what happened", which is the honest state; writing the row
after execution would lose it entirely. A process that is still alive does not
leave one: whatever stops a call (a refusal, a decline, an error, the person
cancelling) records its own outcome, so a missing one means a crash and nothing else.

The trail only grows. A decision is inserted, never replaced: ids are minted to be
unique, so a second decision under the same (session, tool_use_id) is a bug somewhere
else, and replacing the first row would hide the bug and the decision with it. The
``IntegrityError`` is the report. An outcome is set once: a second one, from whatever
reported it twice, changes nothing. The database enforces both (see the triggers in
``conversation/store.py``), so a statement that bypasses this class is refused too.

Arguments pass through the redactor on the way in. The audit table is the one
place a secret could survive the transcript scrubbing, because it stores what
the model asked for rather than what the tool returned.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from nanoclaude.conversation.store import Store
from nanoclaude.permissions.policy import Decision
from nanoclaude.permissions.redact import Redactor


def _arguments_json(arguments: Mapping[str, Any]) -> str:
    """The arguments as JSON for the record, or a line that says they cannot be.

    The decision is recorded before the call is allowed to run, and what the model wrote can
    be nested past what ``json`` can walk: that costs the record its arguments and not the
    call its place in it, and does not end the turn.
    """
    try:
        return json.dumps(dict(arguments), sort_keys=True)
    except (RecursionError, ValueError, TypeError, OverflowError):
        return json.dumps(
            {"_not_recorded": "the arguments are nested too deeply, or not JSON, to be written"}
        )


class AuditLog:
    def __init__(self, store: Store, redactor: Redactor | None = None) -> None:
        self.store = store
        self._redactor = redactor or Redactor()

    def record_decision(
        self,
        session_id: str,
        *,
        turn: int,
        tool_use_id: str,
        tool: str,
        arguments: Mapping[str, Any],
        decision: Decision,
        rule: str,
        parent_call_id: str | None = None,
    ) -> None:
        """Record what was decided about a call, before it runs.

        ``parent_call_id`` is the ``Task`` call that started the sub-agent making this one, and
        None for a call the person's own conversation made.
        """
        payload, _ = self._redactor.scrub(_arguments_json(arguments))
        self.store.db.execute(
            "INSERT INTO tool_calls "
            "(session_id, turn, tool_use_id, ts, tool, args_json, decision, rule, parent_call_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn,
                tool_use_id,
                time.time(),
                tool,
                payload,
                str(decision),
                rule,
                parent_call_id,
            ),
        )
        self.store.db.commit()

    def record_outcome(
        self,
        session_id: str,
        tool_use_id: str,
        *,
        outcome: str,
        duration_ms: int,
        bytes_out: int,
        error: str | None,
    ) -> None:
        self.store.db.execute(
            "UPDATE tool_calls SET outcome = ?, duration_ms = ?, bytes_out = ?, error = ? "
            "WHERE session_id = ? AND tool_use_id = ? AND outcome IS NULL",
            (outcome, duration_ms, bytes_out, error, session_id, tool_use_id),
        )
        self.store.db.commit()
