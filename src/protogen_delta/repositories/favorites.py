"""Отдельная SQLite-база выбранных сообщений с полнотекстовым поиском."""

import asyncio
import hashlib
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.favorite_messages import (
    FAVORITE_LABELS,
    FavoriteMessage,
    FavoriteReason,
)
from protogen_delta.core.memory_search import memory_terms


class FavoritesRepository:
    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "favorites.db"
        with closing(self._connect()) as db, db:
            indexed = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='favorites_fts'"
            ).fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS favorite_messages (
                    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, digest TEXT NOT NULL,
                    text TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL,
                    search_text TEXT NOT NULL, UNIQUE(user_id,digest)
                );
                CREATE INDEX IF NOT EXISTS favorites_owner_time ON favorite_messages(user_id,created_at,id);
                CREATE VIRTUAL TABLE IF NOT EXISTS favorites_fts USING fts5(
                    search_text,content='favorite_messages',content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS favorites_insert AFTER INSERT ON favorite_messages
                BEGIN INSERT INTO favorites_fts(rowid,search_text) VALUES(new.id,new.search_text); END;
                CREATE TRIGGER IF NOT EXISTS favorites_delete AFTER DELETE ON favorite_messages
                BEGIN INSERT INTO favorites_fts(favorites_fts,rowid,search_text)
                    VALUES('delete',old.id,old.search_text); END;
            """)
            if not indexed:
                db.execute("INSERT INTO favorites_fts(favorites_fts) VALUES('rebuild')")
        self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def _owner(user_id: int) -> None:
        if type(user_id) is not int or not 0 < user_id < 2**63:
            raise ValueError("Нужен Telegram ID владельца")

    async def add(
        self, user_id: int, text: str, reason: FavoriteReason, at: float
    ) -> bool:
        self._owner(user_id)
        if (
            not isinstance(text, str)
            or not 5 <= len(text.strip()) <= 8000
            or not isinstance(reason, str)
            or reason not in FAVORITE_LABELS
            or type(at) not in {int, float}
            or not math.isfinite(at)
            or not 0 <= at <= 253402289999
        ):
            raise ValueError("Некорректное избранное сообщение")
        return await finish_operation(
            asyncio.to_thread(self._add, user_id, text, reason, at)
        )

    def _add(self, user_id: int, text: str, reason: FavoriteReason, at: float) -> bool:
        day = (
            datetime.fromtimestamp(at, timezone(timedelta(hours=3)))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )
        digest = hashlib.sha256(text.encode()).hexdigest()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            with db:
                if db.execute(
                    "SELECT 1 FROM favorite_messages WHERE user_id=? AND digest=?",
                    (user_id, digest),
                ).fetchone():
                    return False
                count, latest = db.execute(
                    "SELECT COALESCE(SUM(CASE WHEN created_at>=? THEN 1 ELSE 0 END),0),MAX(created_at) "
                    "FROM favorite_messages WHERE user_id=?",
                    (day, user_id),
                ).fetchone()
                if count >= 5 or latest is not None and at - latest < 900:
                    return False
                db.execute(
                    "INSERT INTO favorite_messages(user_id,digest,text,reason,created_at,search_text) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        user_id,
                        digest,
                        text,
                        reason,
                        at,
                        " ".join(memory_terms(text, limit=2048)),
                    ),
                )
        return True

    @staticmethod
    def _message(row: tuple) -> FavoriteMessage:
        return FavoriteMessage(
            row[0], row[1], row[2], cast(FavoriteReason, row[3]), row[4]
        )

    async def recent(self, user_id: int, *, limit: int = 5) -> list[FavoriteMessage]:
        self._owner(user_id)
        return await asyncio.to_thread(self._read, user_id, "", min(max(limit, 0), 10))

    async def search(
        self, user_id: int, query: str, *, limit: int = 3
    ) -> list[FavoriteMessage]:
        self._owner(user_id)
        terms = memory_terms(query, limit=16)
        if not terms or limit <= 0:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        return await asyncio.to_thread(self._read, user_id, expression, min(limit, 10))

    def _read(self, user_id: int, expression: str, limit: int) -> list[FavoriteMessage]:
        with closing(self._connect()) as db:
            if expression:
                rows = db.execute(
                    "SELECT m.id,m.user_id,m.text,m.reason,m.created_at FROM favorites_fts "
                    "JOIN favorite_messages m ON m.id=favorites_fts.rowid WHERE favorites_fts MATCH ? "
                    "AND m.user_id=? ORDER BY bm25(favorites_fts),m.created_at DESC,m.id DESC LIMIT ?",
                    (expression, user_id, limit),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id,user_id,text,reason,created_at FROM favorite_messages WHERE user_id=? "
                    "ORDER BY created_at DESC,id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        return [self._message(row) for row in rows]

    async def get(self, user_id: int, message_id: int) -> FavoriteMessage | None:
        self._owner(user_id)
        if type(message_id) is not int or not 0 < message_id < 2**63:
            raise ValueError("Некорректный номер избранного сообщения")
        return await asyncio.to_thread(self._get, user_id, message_id)

    def _get(self, user_id: int, message_id: int) -> FavoriteMessage | None:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT id,user_id,text,reason,created_at FROM favorite_messages WHERE user_id=? AND id=?",
                (user_id, message_id),
            ).fetchone()
        return self._message(row) if row else None

    async def delete_user(self, user_id: int) -> None:
        self._owner(user_id)
        await finish_operation(asyncio.to_thread(self._delete, user_id, None))

    async def remove(self, user_id: int, message_id: int) -> bool:
        self._owner(user_id)
        if type(message_id) is not int or not 0 < message_id < 2**63:
            raise ValueError("Некорректный номер избранного сообщения")
        return await finish_operation(
            asyncio.to_thread(self._delete, user_id, message_id)
        )

    def _delete(self, user_id: int, message_id: int | None) -> bool:
        with closing(self._connect()) as db, db:
            if message_id is None:
                cursor = db.execute(
                    "DELETE FROM favorite_messages WHERE user_id=?", (user_id,)
                )
            else:
                cursor = db.execute(
                    "DELETE FROM favorite_messages WHERE user_id=? AND id=?",
                    (user_id, message_id),
                )
        return cursor.rowcount > 0

    async def forget_sources(self, user_id: int, sources: tuple[str, ...]) -> None:
        self._owner(user_id)
        await finish_operation(asyncio.to_thread(self._forget, user_id, sources))

    def _forget(self, user_id: int, sources: tuple[str, ...]) -> None:
        needles = tuple(source.casefold() for source in sources if source)
        terms = tuple(
            set(memory_terms(source, limit=2048)) for source in sources if source
        )
        with closing(self._connect()) as db, db:
            for message_id, text in db.execute(
                "SELECT id,text FROM favorite_messages WHERE user_id=?", (user_id,)
            ).fetchall():
                words = set(memory_terms(text, limit=2048))
                if any(value in text.casefold() for value in needles) or any(
                    value and value <= words for value in terms
                ):
                    db.execute(
                        "DELETE FROM favorite_messages WHERE user_id=? AND id=?",
                        (user_id, message_id),
                    )
