"""Session persistence.

The schema is spec 17.8 verbatim and the column names are a contract: the audit
CLI reads them, and a database written by one version is opened by the next.
Migrations are additive only.

``total_cost_usd`` is NULL when the cost is not known: a model with no price, or
a session that has not finished. It is never 0 in those cases, because a zero
reads as free and that is the one answer known to be wrong. The column was
``NOT NULL DEFAULT 0`` before v0.1 shipped; relaxing a constraint is not an
additive change, but no build that created such a database was ever released, so
there is nothing to migrate and the schema below is simply the schema.

Blocks are stored as JSON with an explicit ``type`` discriminator rather than
pickled, so a session written today is still readable when the dataclasses gain
a field.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoclaude.conversation.transcript import (
    Block,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
)
from nanoclaude.providers.base import Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    cwd TEXT NOT NULL,
    roles_json TEXT NOT NULL,
    total_cost_usd REAL,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    blocks_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    tool_use_id TEXT NOT NULL,
    ts REAL NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    rule TEXT NOT NULL,
    outcome TEXT,
    duration_ms INTEGER,
    bytes_out INTEGER,
    error TEXT,
    PRIMARY KEY (session_id, tool_use_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
"""


def encode_blocks(blocks: Sequence[Block]) -> str:
    out: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            out.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            out.append({"type": "thinking", "text": block.text, "signature": block.signature})
        elif isinstance(block, ToolUseBlock):
            out.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "arguments": dict(block.arguments),
                }
            )
        else:
            out.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                    "is_error": block.is_error,
                }
            )
    return json.dumps(out)


def decode_blocks(payload: str) -> tuple[Block, ...]:
    blocks: list[Block] = []
    for raw in json.loads(payload):
        kind = raw["type"]
        if kind == "text":
            blocks.append(TextBlock(raw["text"]))
        elif kind == "thinking":
            blocks.append(ThinkingBlock(raw["text"], raw.get("signature", "")))
        elif kind == "tool_use":
            blocks.append(ToolUseBlock(raw["id"], raw["name"], raw["arguments"]))
        elif kind == "tool_result":
            blocks.append(
                ToolResultBlock(raw["tool_use_id"], raw["content"], raw.get("is_error", False))
            )
        else:
            raise ValueError(f"unknown block type {kind!r} in stored session")
    return tuple(blocks)


@dataclass(frozen=True, slots=True)
class SessionRow:
    id: str
    started_at: float
    ended_at: float | None
    cwd: str
    total_cost_usd: float | None
    total_input_tokens: int
    total_output_tokens: int


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: sqlite3.Connection | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("store is not open; call open() first")
        return self._db

    def create_session(self, session_id: str, *, cwd: str, roles: Mapping[str, str]) -> None:
        self.db.execute(
            "INSERT INTO sessions (id, started_at, cwd, roles_json) VALUES (?, ?, ?, ?)",
            (session_id, time.time(), cwd, json.dumps(dict(roles), sort_keys=True)),
        )
        self.db.commit()

    def append_message(self, session_id: str, seq: int, message: Message) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO messages (session_id, seq, role, blocks_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, seq, message.role, encode_blocks(message.blocks), time.time()),
        )
        self.db.commit()

    def load_transcript(self, session_id: str) -> Transcript:
        rows = self.db.execute(
            "SELECT role, blocks_json FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        return Transcript(
            tuple(Message(row["role"], decode_blocks(row["blocks_json"])) for row in rows)
        )

    def replace_transcript(self, session_id: str, transcript: Transcript) -> None:
        """Make the stored messages exactly ``transcript``.

        Compaction and ``/clear`` replace a transcript instead of extending it.
        Inserting the new rows over the old ones would leave the old tail behind,
        and the session would reload as a summary followed by the messages it
        replaced. The delete and the inserts are one transaction, so a failure
        part-way (a message that cannot be encoded, a full disk) leaves the old
        transcript in place, not an empty or half-replaced one.
        """
        now = time.time()
        with self.db:
            self.db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self.db.executemany(
                "INSERT INTO messages (session_id, seq, role, blocks_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    (session_id, seq, message.role, encode_blocks(message.blocks), now)
                    for seq, message in enumerate(transcript.messages)
                ),
            )

    def finish_session(self, session_id: str, usage: Usage, cost_usd: float | None) -> None:
        """Record what a session used. ``None`` is stored as NULL: cost not known."""
        self.db.execute(
            "UPDATE sessions SET ended_at = ?, total_cost_usd = ?, "
            "total_input_tokens = ?, total_output_tokens = ? WHERE id = ?",
            (time.time(), cost_usd, usage.input_tokens, usage.output_tokens, session_id),
        )
        self.db.commit()

    def recent_sessions(self, limit: int = 10) -> list[SessionRow]:
        rows = self.db.execute(
            "SELECT id, started_at, ended_at, cwd, total_cost_usd, total_input_tokens, "
            "total_output_tokens FROM sessions ORDER BY started_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [SessionRow(**dict(row)) for row in rows]

    def latest_session_id(self) -> str | None:
        row = self.db.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None
