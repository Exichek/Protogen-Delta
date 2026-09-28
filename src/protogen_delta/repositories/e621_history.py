"""SQLite-история уже показанных пользователю постов e621."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path


class E621HistoryRepository:
    """Не допускать повторов между перезапусками бота."""

    def __init__(self, data_dir: Path, *, max_posts_per_user: int = 2000) -> None:
        if max_posts_per_user <= 0:
            raise ValueError("max_posts_per_user должен быть больше нуля")
        self._path = data_dir / "e621.db"
        self._max_posts = max_posts_per_user
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS shown_posts (
                user_id INTEGER NOT NULL,
                post_id INTEGER NOT NULL,
                shown_at REAL NOT NULL,
                PRIMARY KEY(user_id, post_id)
                )""")

    async def seen_ids(self, user_id: int) -> set[int]:
        """Вернуть ID ранее показанных постов."""
        return await asyncio.to_thread(self._seen_ids_sync, user_id)

    async def mark_seen(self, user_id: int, post_id: int, shown_at: float) -> None:
        """Запомнить показ и удалить самые старые записи сверх лимита."""
        await asyncio.to_thread(self._mark_seen_sync, user_id, post_id, shown_at)

    def _seen_ids_sync(self, user_id: int) -> set[int]:
        with closing(sqlite3.connect(self._path)) as connection:
            rows = connection.execute(
                "SELECT post_id FROM shown_posts WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {int(row[0]) for row in rows}

    def _mark_seen_sync(self, user_id: int, post_id: int, shown_at: float) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                """INSERT INTO shown_posts(user_id, post_id, shown_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, post_id) DO UPDATE SET shown_at = excluded.shown_at""",
                (user_id, post_id, shown_at),
            )
            connection.execute(
                """DELETE FROM shown_posts WHERE user_id = ? AND post_id NOT IN (
                SELECT post_id FROM shown_posts WHERE user_id = ?
                ORDER BY shown_at DESC LIMIT ?
                )""",
                (user_id, user_id, self._max_posts),
            )
