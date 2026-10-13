"""Состояние обзоров и сохранённый отчёт в общей базе памяти."""

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from protogen_delta.core.async_completion import finish_operation


@dataclass(frozen=True, slots=True)
class CodeReviewState:
    fingerprints: dict[str, str]
    attempted_at: float
    report: str
    notified: bool


class CodeReviewsRepository:
    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "memories.db"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS code_reviews (user_id INTEGER PRIMARY KEY, "
                "fingerprints TEXT NOT NULL, attempted_at REAL NOT NULL, "
                "report TEXT NOT NULL, notified INTEGER NOT NULL)"
            )

    async def load(self, user_id: int) -> CodeReviewState | None:
        return await asyncio.to_thread(self._load, user_id)

    def _load(self, user_id: int) -> CodeReviewState | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT fingerprints,attempted_at,report,notified FROM code_reviews WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        fingerprints = json.loads(row[0])
        if not isinstance(fingerprints, dict) or any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in fingerprints.items()
        ):
            raise ValueError("Некорректное состояние обзора")
        return CodeReviewState(fingerprints, row[1], row[2], bool(row[3]))

    async def save(self, user_id: int, state: CodeReviewState) -> None:
        await finish_operation(asyncio.to_thread(self._save, user_id, state))

    def _save(self, user_id: int, state: CodeReviewState) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO code_reviews VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
                "fingerprints=excluded.fingerprints,attempted_at=excluded.attempted_at,"
                "report=excluded.report,notified=excluded.notified",
                (
                    user_id,
                    json.dumps(state.fingerprints),
                    state.attempted_at,
                    state.report,
                    state.notified,
                ),
            )
