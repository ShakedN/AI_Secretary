"""SQLite log of every call the assistant took."""
from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .agent.summary import CallSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    call_sid         TEXT UNIQUE,
    from_number      TEXT,
    started_at       TEXT,
    ended_at         TEXT,
    caller_name      TEXT,
    callback_phone   TEXT,
    lawyer_requested TEXT,
    topic            TEXT,
    summary          TEXT,
    urgency          TEXT,
    existing_client  TEXT,
    callback_time    TEXT,
    next_step        TEXT,
    caller_mood      TEXT,
    missing_info     TEXT,
    call_review      TEXT,
    duration_seconds INTEGER,
    transcript       TEXT,
    emailed_to       TEXT
)
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def save_call(
    path: Path,
    *,
    call_sid: str,
    from_number: str | None,
    started_at: datetime,
    ended_at: datetime,
    summary: CallSummary,
    transcript: str,
    emailed_to: list[str],
) -> None:
    with connect(path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO calls
               (call_sid, from_number, started_at, ended_at, caller_name, callback_phone,
                lawyer_requested, topic, summary, urgency, existing_client, callback_time,
                next_step, caller_mood, missing_info, call_review, duration_seconds,
                transcript, emailed_to)
               VALUES (:call_sid, :from_number, :started_at, :ended_at, :caller_name, :callback_phone,
                :lawyer_requested, :topic, :summary, :urgency, :existing_client, :callback_time,
                :next_step, :caller_mood, :missing_info, :call_review, :duration_seconds,
                :transcript, :emailed_to)""",
            {
                **asdict(summary),
                "call_sid": call_sid,
                "from_number": from_number,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "duration_seconds": int((ended_at - started_at).total_seconds()),
                "transcript": transcript,
                "emailed_to": ", ".join(emailed_to),
            },
        )


def recent_calls(path: Path, limit: int = 50) -> list[dict]:
    with connect(path) as conn:
        rows = conn.execute("SELECT * FROM calls ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_call(path: Path, call_sid: str) -> dict | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,)).fetchone()
    return dict(row) if row else None
