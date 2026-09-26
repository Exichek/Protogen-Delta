"""SQLite-хранилище эпизодической памяти и проактивного общения."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MemoryKind = Literal["grievance", "funny", "topic"]


@dataclass(frozen=True, slots=True)
class Memory:
    """Один значимый эпизод общения с пользователем."""

    kind: MemoryKind
    text: str
    created_at: float


@dataclass(frozen=True, slots=True)
class ProactiveCandidate:
    """Пользователь, которому можно написать проактивно."""

    user_id: int
    unanswered_count: int


class MemoriesRepository:
    """Хранить эпизоды, активность и настройки проактивных сообщений."""

    def __init__(self, data_dir: Path, *, max_memories_per_user: int = 50) -> None:
        if max_memories_per_user <= 0:
            raise ValueError("max_memories_per_user должен быть больше нуля")
        self._path = data_dir / "memories.db"
        self._max_memories = max_memories_per_user
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    async def remember(
        self, user_id: int, kind: MemoryKind, text: str, created_at: float
    ) -> None:
        """Сохранить уникальный эпизод и удалить самые старые сверх лимита."""
        await asyncio.to_thread(self._remember_sync, user_id, kind, text, created_at)

    async def recent(self, user_id: int, limit: int = 8) -> list[Memory]:
        """Вернуть последние эпизоды от новых к старым."""
        return await asyncio.to_thread(self._recent_sync, user_id, limit)

    async def note_user_activity(self, user_id: int, at: float) -> None:
        """Зафиксировать сообщение пользователя и сбросить счётчик молчания."""
        await asyncio.to_thread(self._note_user_activity_sync, user_id, at)

    async def set_proactive(self, user_id: int, enabled: bool, at: float) -> None:
        """Изменить согласие пользователя на проактивные сообщения."""
        await asyncio.to_thread(self._set_proactive_sync, user_id, enabled, at)

    async def proactive_enabled(self, user_id: int) -> bool:
        """Вернуть текущую настройку; для нового пользователя она включена."""
        return await asyncio.to_thread(self._proactive_enabled_sync, user_id)

    async def due_candidates(
        self, *, now: float, idle_seconds: float, cooldown_seconds: float, limit: int
    ) -> list[ProactiveCandidate]:
        """Найти пользователей, которым пора написать с учётом backoff."""
        return await asyncio.to_thread(
            self._due_candidates_sync,
            now,
            idle_seconds,
            cooldown_seconds,
            limit,
        )

    async def note_proactive_sent(self, user_id: int, at: float) -> None:
        """Зафиксировать доставленное проактивное сообщение."""
        await asyncio.to_thread(self._note_proactive_sent_sync, user_id, at)

    async def delete_user(self, user_id: int) -> None:
        """Удалить всю дополнительную память пользователя."""
        await asyncio.to_thread(self._delete_user_sync, user_id)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _remember_sync(
        self, user_id: int, kind: MemoryKind, text: str, created_at: float
    ) -> None:
        clean = " ".join(text.split())[:500]
        if not clean:
            return
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO memories(user_id, kind, text, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, kind, text) DO UPDATE SET
                    created_at = excluded.created_at
                """,
                (user_id, kind, clean, created_at),
            )
            connection.execute(
                """
                DELETE FROM memories
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM memories WHERE user_id = ?
                    ORDER BY created_at DESC, id DESC LIMIT ?
                )
                """,
                (user_id, user_id, self._max_memories),
            )

    def _recent_sync(self, user_id: int, limit: int) -> list[Memory]:
        if limit <= 0:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT kind, text, created_at FROM memories
                WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [Memory(kind=row[0], text=row[1], created_at=row[2]) for row in rows]

    def _note_user_activity_sync(self, user_id: int, at: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO engagement(user_id, last_user_message_at)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_user_message_at = excluded.last_user_message_at,
                    unanswered_count = 0
                """,
                (user_id, at),
            )

    def _set_proactive_sync(self, user_id: int, enabled: bool, at: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO engagement(user_id, proactive_enabled, last_user_message_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET proactive_enabled = excluded.proactive_enabled
                """,
                (user_id, int(enabled), at),
            )

    def _proactive_enabled_sync(self, user_id: int) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT proactive_enabled FROM engagement WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return True if row is None else bool(row[0])

    def _due_candidates_sync(
        self, now: float, idle_seconds: float, cooldown_seconds: float, limit: int
    ) -> list[ProactiveCandidate]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT user_id, unanswered_count FROM engagement
                WHERE proactive_enabled = 1
                  AND last_user_message_at > 0
                  AND ? - last_user_message_at >= ?
                  AND (last_proactive_at IS NULL OR
                       ? - last_proactive_at >= ? * (1 << MIN(unanswered_count, 6)))
                ORDER BY COALESCE(last_proactive_at, 0), last_user_message_at
                LIMIT ?
                """,
                (now, idle_seconds, now, cooldown_seconds, limit),
            ).fetchall()
        return [
            ProactiveCandidate(user_id=row[0], unanswered_count=row[1]) for row in rows
        ]

    def _note_proactive_sent_sync(self, user_id: int, at: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE engagement SET last_proactive_at = ?,
                    unanswered_count = unanswered_count + 1
                WHERE user_id = ?
                """,
                (at, user_id),
            )

    def _delete_user_sync(self, user_id: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM engagement WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engagement (
                    user_id INTEGER PRIMARY KEY,
                    proactive_enabled INTEGER NOT NULL DEFAULT 1,
                    last_user_message_at REAL NOT NULL DEFAULT 0,
                    last_proactive_at REAL,
                    unanswered_count INTEGER NOT NULL DEFAULT 0
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('grievance', 'funny', 'topic')),
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(user_id, kind, text)
                )
                """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_user_time ON memories(user_id, created_at DESC)"
            )
