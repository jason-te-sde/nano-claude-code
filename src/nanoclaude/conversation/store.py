"""Session persistence.

The schema is spec 17.8 verbatim and the column names are a contract: the audit
CLI reads them, and a database written by one version is opened by the next.
Migrations are additive only.

One table is added to it: ``messages_archive``. ``messages`` is the transcript the
model is shown, which is what a resumed session reloads, so compaction and
``/clear`` have to replace its rows. The audit trail is append-only, so a row a
replacement removes or changes is moved to the archive, with the time it was
archived, in the same transaction. Between them the two tables hold every message
ever stored.

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
import os
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
CREATE TABLE IF NOT EXISTS messages_archive (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    blocks_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    archived_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_archive_session
    ON messages_archive(session_id, archived_at);
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


class MessageConflictError(RuntimeError):
    """A different message is already stored where another was to be appended.

    Another process has written to the session: the rows are not the ones the writer
    believes they are. Nothing was written.
    """

    def __init__(self, session_id: str, seq: int) -> None:
        super().__init__(
            f"session {session_id} already holds a different message at position {seq}"
        )
        self.session_id = session_id
        self.seq = seq


@dataclass(frozen=True, slots=True)
class SessionRow:
    id: str
    started_at: float
    ended_at: float | None
    cwd: str
    total_cost_usd: float | None
    total_input_tokens: int
    total_output_tokens: int


def same_directory(first: str, second: str) -> bool:
    """Whether two paths are the same place, whatever they were spelled like.

    Through a symlink, with another capitalisation where the file system ignores it, with a
    ``..`` in it: the disk says, by comparing what each one is. A directory that is no longer
    there cannot be asked, and is the place its resolved path says it was.
    """
    if first == second:
        return True
    try:
        return Path(first).samefile(second)
    except OSError:
        return os.path.realpath(first) == os.path.realpath(second)


def _same_messages(
    old: Sequence[tuple[str, str]], new: Sequence[tuple[str, str]]
) -> list[tuple[int, int]]:
    """Which messages two transcripts share: pairs of (index in old, index in new).

    A replacement keeps messages in one of two shapes, and the shape says which. If
    the number of messages is unchanged (a micro-compaction that shrank some tool
    results) each position that holds the same message holds that message. If it
    changed (compaction writes a summary and keeps the last turns, ``/clear`` keeps
    nothing) what is kept is the run both begin with and the run both end with, matched
    from the end.

    Never by what a message says alone. The same "continue" can be there ten times, and
    matching it by its words gives one copy the time of another, or leaves a message at
    a position where an earlier copy happened to be. Position, from the front or from
    the back, is what says which copy it is.
    """
    if len(old) == len(new):
        return [
            (i, i)
            for i, (before, after) in enumerate(zip(old, new, strict=True))
            if before == after
        ]
    shortest = min(len(old), len(new))
    start = 0
    while start < shortest and old[start] == new[start]:
        start += 1
    end = 0
    while end < shortest - start and old[-1 - end] == new[-1 - end]:
        end += 1
    return [(i, i) for i in range(start)] + [
        (len(old) - end + j, len(new) - end + j) for j in range(end)
    ]


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
        """Store ``message`` as the message at ``seq``, the next one in its session.

        The same message already stored there is not an error: a message is written
        once, so an append that finds it has nothing to do. A different one is, and
        raises :class:`MessageConflictError`: another process has written to this
        session, and the rows are not the ones this writer believes they are. Nothing
        is overwritten, since the audit trail only grows.
        """
        role, blocks_json = message.role, encode_blocks(message.blocks)
        try:
            with self.db:
                self.db.execute(
                    "INSERT INTO messages (session_id, seq, role, blocks_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, seq, role, blocks_json, time.time()),
                )
        except sqlite3.IntegrityError:
            stored = self.db.execute(
                "SELECT role, blocks_json FROM messages WHERE session_id = ? AND seq = ?",
                (session_id, seq),
            ).fetchone()
            if stored is None:
                raise  # not a row in the way: the table refused the value itself
            if (stored["role"], stored["blocks_json"]) == (role, blocks_json):
                return
            raise MessageConflictError(session_id, seq) from None

    def load_transcript(self, session_id: str) -> Transcript:
        """The live messages: the transcript the model is shown, not the archive."""
        rows = self.db.execute(
            "SELECT role, blocks_json FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        return Transcript(
            tuple(Message(row["role"], decode_blocks(row["blocks_json"])) for row in rows)
        )

    def replace_transcript(self, session_id: str, transcript: Transcript) -> None:
        """Make the live messages exactly ``transcript``, and keep what that removes.

        Compaction and ``/clear`` replace a transcript instead of extending it.
        Inserting the new rows over the old ones would leave the old tail behind,
        and the session would reload as a summary followed by the messages it
        replaced. So the rows that no longer belong are deleted, and each is first
        copied to ``messages_archive``: a row is archived when its position now
        holds another message, or none. Rows that did not change are not touched,
        and a message that moves to another position keeps the time it was first
        stored (see :func:`_same_messages` for which messages those are).

        Archiving, deleting and inserting are one transaction, so a failure part-way
        (a value the table refuses, a full disk) leaves the old transcript in place
        and the archive as it was.
        """
        wanted = [(m.role, encode_blocks(m.blocks)) for m in transcript.messages]
        now = time.time()
        with self.db:
            live = self.db.execute(
                "SELECT seq, role, blocks_json, created_at FROM messages "
                "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
            pairs = _same_messages([(row["role"], row["blocks_json"]) for row in live], wanted)
            # A pair whose row is already at its position stays; the others are moves.
            stays = [(i, at) for i, at in pairs if live[i]["seq"] == at]
            moved = {at: live[i]["created_at"] for i, at in pairs if live[i]["seq"] != at}
            staying_rows = {i for i, _ in stays}
            staying_positions = {at for _, at in stays}
            removed = [(row["seq"], row) for i, row in enumerate(live) if i not in staying_rows]
            self._archive(session_id, removed, now)
            self.db.executemany(
                "DELETE FROM messages WHERE session_id = ? AND seq = ?",
                [(session_id, seq) for seq, _ in removed],
            )
            self.db.executemany(
                "INSERT INTO messages (session_id, seq, role, blocks_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (session_id, at, role, blocks_json, moved.get(at, now))
                    for at, (role, blocks_json) in enumerate(wanted)
                    if at not in staying_positions
                ],
            )

    def _archive(
        self, session_id: str, rows: Sequence[tuple[int, sqlite3.Row]], archived_at: float
    ) -> None:
        self.db.executemany(
            "INSERT INTO messages_archive "
            "(session_id, seq, role, blocks_json, created_at, archived_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (session_id, seq, row["role"], row["blocks_json"], row["created_at"], archived_at)
                for seq, row in rows
            ],
        )

    def finish_session(self, session_id: str, usage: Usage, cost_usd: float | None) -> None:
        """Record what a session used. ``None`` is stored as NULL: cost not known."""
        self.db.execute(
            "UPDATE sessions SET ended_at = ?, total_cost_usd = ?, "
            "total_input_tokens = ?, total_output_tokens = ? WHERE id = ?",
            (time.time(), cost_usd, usage.input_tokens, usage.output_tokens, session_id),
        )
        self.db.commit()

    def next_tool_turn(self, session_id: str) -> int:
        """The first turn number no audited call of this session has been recorded under.

        One past the largest, 0 when nothing has been audited. The audit's turn is
        the session's own count and a resumed session carries on from it.
        """
        row = self.db.execute(
            "SELECT MAX(turn) AS latest FROM tool_calls WHERE session_id = ?", (session_id,)
        ).fetchone()
        return 0 if row["latest"] is None else int(row["latest"]) + 1

    def session_row(self, session_id: str) -> SessionRow | None:
        """One session's row, or None when no such session is stored."""
        row = self.db.execute(
            "SELECT id, started_at, ended_at, cwd, total_cost_usd, total_input_tokens, "
            "total_output_tokens FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return SessionRow(**dict(row)) if row else None

    def recent_sessions(self, limit: int = 10) -> list[SessionRow]:
        rows = self.db.execute(
            "SELECT id, started_at, ended_at, cwd, total_cost_usd, total_input_tokens, "
            "total_output_tokens FROM sessions ORDER BY started_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [SessionRow(**dict(row)) for row in rows]

    def latest_session_id(self, *, cwd: str | None = None) -> str | None:
        """The session started most recently: in the whole store, or the one to continue in ``cwd``.

        With ``cwd`` it is the latest session started in that directory, as a place (see
        :func:`same_directory`), in which something was said: a row left by a session that was
        closed at once holds nothing to continue, and is passed over. Without it, the latest
        row there is, whatever it holds.
        """
        if cwd is None:
            row = self.db.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
            return row["id"] if row else None
        talked = self.db.execute(
            "SELECT id, cwd FROM sessions "
            "WHERE EXISTS (SELECT 1 FROM messages WHERE messages.session_id = sessions.id) "
            "ORDER BY started_at DESC, rowid DESC"
        )
        known: dict[str, bool] = {}
        for row in talked:
            if row["cwd"] not in known:
                known[row["cwd"]] = same_directory(row["cwd"], cwd)
            if known[row["cwd"]]:
                return str(row["id"])
        return None
