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


@dataclass(frozen=True, slots=True)
class CodeReviewDelivery:
    """Последний действительно отправленный текст, отдельно от результата генерации."""

    text: str
    delivered_at: float
    automatic: bool = False


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
            db.execute(
                "CREATE TABLE IF NOT EXISTS code_review_deliveries ("
                "user_id INTEGER PRIMARY KEY, text TEXT NOT NULL, delivered_at REAL NOT NULL, "
                "automatic INTEGER NOT NULL)"
            )

    async def record_delivery(
        self, user_id: int, text: str, delivered_at: float, automatic: bool = False
    ) -> None:
        """Записать подтверждённую отправку без нового вызова модели."""
        await finish_operation(
            asyncio.to_thread(
                self._record_delivery, user_id, text, delivered_at, automatic
            )
        )

    def _record_delivery(
        self, user_id: int, text: str, delivered_at: float, automatic: bool
    ) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT INTO code_review_deliveries VALUES (?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET text=excluded.text, "
                "delivered_at=excluded.delivered_at, automatic=excluded.automatic",
                (user_id, text, delivered_at, int(automatic)),
            )

    async def delivery(self, user_id: int) -> CodeReviewDelivery | None:
        """Получить только последнюю отправку указанного владельца."""
        return await asyncio.to_thread(self._delivery, user_id)

    def _delivery(self, user_id: int) -> CodeReviewDelivery | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT text,delivered_at,automatic FROM code_review_deliveries WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return CodeReviewDelivery(row[0], row[1], bool(row[2])) if row else None

    async def clear_delivery(self, user_id: int) -> None:
        """Убрать разговорный контекст обзора при удалении памяти."""
        await finish_operation(asyncio.to_thread(self._clear_delivery, user_id))

    def _clear_delivery(self, user_id: int) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM code_review_deliveries WHERE user_id=?", (user_id,))

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
