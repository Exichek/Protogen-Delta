"""SQLite-хранилище эпизодической памяти и проактивного общения."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from protogen_delta.core.memory_search import (
    SEARCH_VERSION,
    memory_index_text,
    memory_terms,
)

MemoryKind = Literal["grievance", "funny", "topic"]


@dataclass(frozen=True, slots=True)
class Memory:
    """Один значимый эпизод общения с пользователем."""

    kind: MemoryKind
    text: str
    created_at: float
    importance: int = 1


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
        self,
        user_id: int,
        kind: MemoryKind,
        text: str,
        created_at: float,
        *,
        importance: int = 1,
    ) -> None:
        """Сохранить эпизод; при переполнении сначала удалять менее важные."""
        if type(importance) is not int or not 1 <= importance <= 3:
            raise ValueError("importance должен быть целым числом от 1 до 3")
        await asyncio.to_thread(
            self._remember_sync, user_id, kind, text, created_at, importance
        )

    async def recent(self, user_id: int, limit: int = 8) -> list[Memory]:
        """Вернуть последние эпизоды от новых к старым."""
        return await asyncio.to_thread(self._recent_sync, user_id, limit)

    async def search(self, user_id: int, query: str, limit: int = 3) -> list[Memory]:
        """Искать во всех эпизодах этого пользователя, учитывая важность."""
        terms = memory_terms(query, limit=16)
        if limit <= 0 or not terms:
            return []
        return await asyncio.to_thread(self._search_sync, user_id, terms, limit)

    def _search_sync(
        self, user_id: int, terms: tuple[str, ...], limit: int
    ) -> list[Memory]:
        # Только нормализованные слова; операторы пользователя не становятся MATCH.
        expression = " OR ".join(f'"{term}"' for term in terms)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT m.kind, m.text, m.created_at, m.importance
                FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ? AND m.user_id = ?
                ORDER BY bm25(memories_fts) * (1 + 0.15 * (m.importance - 1)),
                         m.created_at DESC, m.id DESC
                LIMIT ?
                """,
                (expression, user_id, limit),
            ).fetchall()
        return [Memory(*row) for row in rows]

    async def note_user_activity(self, user_id: int, at: float) -> None:
        """Зафиксировать сообщение пользователя и сбросить счётчик молчания."""
        await asyncio.to_thread(self._note_user_activity_sync, user_id, at)

    async def set_proactive(self, user_id: int, enabled: bool, at: float) -> None:
        """Изменить согласие пользователя на проактивные сообщения."""
        await asyncio.to_thread(self._set_proactive_sync, user_id, enabled, at)

    async def proactive_enabled(self, user_id: int) -> bool:
        """Вернуть текущую настройку; для нового пользователя она включена."""
        return await asyncio.to_thread(self._proactive_enabled_sync, user_id)

    async def suspend_proactive_delivery(self, user_id: int) -> None:
        """Приостановить недоступный чат до новой активности, сохранив настройку."""
        await asyncio.to_thread(self._suspend_proactive_delivery_sync, user_id)

    def _suspend_proactive_delivery_sync(self, user_id: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE engagement SET delivery_suspended = 1 WHERE user_id = ?",
                (user_id,),
            )

    async def enable_proactive_for_all(self) -> None:
        """Включить фоновые сообщения всем известным пользователям."""
        await asyncio.to_thread(self._enable_proactive_for_all_sync)

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

    async def is_due(
        self, user_id: int, *, now: float, idle_seconds: float, cooldown_seconds: float
    ) -> bool:
        """Проверить активность и настройку ещё раз после генерации сообщения."""
        candidates = await asyncio.to_thread(
            self._due_candidates_sync, now, idle_seconds, cooldown_seconds, 1, user_id
        )
        return bool(candidates)

    async def delete_user(self, user_id: int) -> None:
        """Удалить всю дополнительную память пользователя."""
        await asyncio.to_thread(self._delete_user_sync, user_id)

    async def forget_sources(self, user_id: int, sources: tuple[str, ...]) -> None:
        """Не возвращать удалённые/устаревшие факты через эпизодическую память."""
        await asyncio.to_thread(self._forget_sources_sync, user_id, sources)

    def _forget_sources_sync(self, user_id: int, sources: tuple[str, ...]) -> None:
        with closing(self._connect()) as connection, connection:
            needles = tuple(source.casefold() for source in sources if source)
            term_sets = tuple(set(memory_terms(source)) for source in sources if source)
            rows = connection.execute(
                "SELECT id,text FROM memories WHERE user_id=?", (user_id,)
            ).fetchall()
            for row_id, text in rows:
                words = set(memory_terms(text))
                if not any(source in text.casefold() for source in needles) and not any(
                    terms and terms <= words for terms in term_sets
                ):
                    continue
                connection.execute(
                    "DELETE FROM memories WHERE user_id=? AND id=?",
                    (user_id, row_id),
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _remember_sync(
        self,
        user_id: int,
        kind: MemoryKind,
        text: str,
        created_at: float,
        importance: int,
    ) -> None:
        clean = " ".join(text.split())[:500]
        if not clean:
            return
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO memories(user_id, kind, text, created_at, search_text, importance)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, kind, text) DO UPDATE SET
                    created_at = excluded.created_at,
                    importance = MAX(memories.importance, excluded.importance)
                """,
                (
                    user_id,
                    kind,
                    clean,
                    created_at,
                    memory_index_text(clean),
                    importance,
                ),
            )
            connection.execute(
                """
                DELETE FROM memories
                WHERE user_id = ? AND kind = ? AND id NOT IN (
                    SELECT id FROM memories WHERE user_id = ? AND kind = ?
                    ORDER BY importance DESC, created_at DESC, id DESC LIMIT ?
                )
                """,
                (user_id, kind, user_id, kind, self._max_memories),
            )

    def _recent_sync(self, user_id: int, limit: int) -> list[Memory]:
        if limit <= 0:
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT kind, text, created_at, importance FROM memories
                WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [Memory(*row) for row in rows]

    def _note_user_activity_sync(self, user_id: int, at: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO engagement(user_id, last_user_message_at)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_user_message_at = excluded.last_user_message_at,
                    unanswered_count = 0,
                    delivery_suspended = 0
                """,
                (user_id, at),
            )

    def _set_proactive_sync(self, user_id: int, enabled: bool, at: float) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO engagement(user_id, proactive_enabled, last_user_message_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    proactive_enabled = excluded.proactive_enabled,
                    delivery_suspended = CASE WHEN excluded.proactive_enabled = 1
                        THEN 0 ELSE engagement.delivery_suspended END
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

    def _enable_proactive_for_all_sync(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("UPDATE engagement SET proactive_enabled = 1")

    def _due_candidates_sync(
        self,
        now: float,
        idle_seconds: float,
        cooldown_seconds: float,
        limit: int,
        user_id: int | None = None,
    ) -> list[ProactiveCandidate]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT user_id, unanswered_count FROM engagement
                WHERE proactive_enabled = 1
                  AND delivery_suspended = 0
                  AND (? IS NULL OR user_id = ?)
                  AND last_user_message_at > 0
                  AND ? - last_user_message_at >= ?
                  AND (last_proactive_at IS NULL OR
                       ? - last_proactive_at >= ? * (1 << MIN(unanswered_count, 6)))
                ORDER BY COALESCE(last_proactive_at, 0), last_user_message_at
                LIMIT ?
                """,
                (user_id, user_id, now, idle_seconds, now, cooldown_seconds, limit),
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
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engagement (
                    user_id INTEGER PRIMARY KEY,
                    proactive_enabled INTEGER NOT NULL DEFAULT 1,
                    last_user_message_at REAL NOT NULL DEFAULT 0,
                    last_proactive_at REAL,
                    unanswered_count INTEGER NOT NULL DEFAULT 0,
                    delivery_suspended INTEGER NOT NULL DEFAULT 0
                )
                """)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(engagement)")
            }
            if "delivery_suspended" not in columns:
                connection.execute(
                    "ALTER TABLE engagement ADD COLUMN delivery_suspended INTEGER NOT NULL DEFAULT 0"
                )
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
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(memories)")
            }
            if "search_text" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN search_text TEXT NOT NULL DEFAULT ''"
                )
            if "importance" not in columns:
                connection.execute(
                    "ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 1"
                )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memory_search_version "
                "(id INTEGER PRIMARY KEY CHECK(id = 1), version INTEGER NOT NULL)"
            )
            version = connection.execute(
                "SELECT version FROM memory_search_version WHERE id = 1"
            ).fetchone()
            indexed = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'memories_fts'"
            ).fetchone()
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "search_text, content='memories', content_rowid='id', tokenize='unicode61')"
            )
            if not indexed or version != (SEARCH_VERSION,):
                for trigger in (
                    "memories_search_insert",
                    "memories_search_delete",
                    "memories_search_update",
                ):
                    connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
                rows = connection.execute("SELECT id, text FROM memories").fetchall()
                connection.executemany(
                    "UPDATE memories SET search_text = ? WHERE id = ?",
                    [(memory_index_text(text), row_id) for row_id, text in rows],
                )
                connection.execute(
                    "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"
                )
                connection.execute(
                    "INSERT INTO memory_search_version VALUES(1, ?) "
                    "ON CONFLICT(id) DO UPDATE SET version=excluded.version",
                    (SEARCH_VERSION,),
                )
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_search_insert AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, search_text) VALUES(new.id, new.search_text);
                END
                """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_search_delete AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, search_text)
                    VALUES('delete', old.id, old.search_text);
                END
                """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_search_update AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, search_text)
                    VALUES('delete', old.id, old.search_text);
                    INSERT INTO memories_fts(rowid, search_text) VALUES(new.id, new.search_text);
                END
                """)
            # После временного отката старый код мог добавить записи без нормализации.
            rows = connection.execute(
                "SELECT id, text FROM memories WHERE search_text = ''"
            ).fetchall()
            connection.executemany(
                "UPDATE memories SET search_text = ? WHERE id = ?",
                [
                    (normalized, row_id)
                    for row_id, text in rows
                    if (normalized := memory_index_text(text))
                ],
            )
