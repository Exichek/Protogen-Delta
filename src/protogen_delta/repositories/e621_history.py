"""SQLite-история уже показанных пользователю постов e621."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

MediaFilter = Literal["all", "images", "videos"]
SearchOrder = Literal["site", "favcount", "random"]


@dataclass(frozen=True, slots=True)
class E621Preferences:
    """Настройки выдачи одного пользователя, независимо от диалога."""

    media_filter: MediaFilter = "all"
    order: SearchOrder = "site"
    count: int = 1


@dataclass(frozen=True, slots=True)
class CachedMedia:
    """Повторно используемый файл именно текущего Telegram-бота."""

    file_id: str
    kind: str
    notice: str = ""


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
            connection.execute("""CREATE TABLE IF NOT EXISTS search_preferences (
                user_id INTEGER PRIMARY KEY,
                media_filter TEXT NOT NULL, search_order TEXT NOT NULL,
                result_count INTEGER NOT NULL
                )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS media_cache (
                bot_id INTEGER NOT NULL, asset_key TEXT NOT NULL,
                file_id TEXT NOT NULL, kind TEXT NOT NULL, notice TEXT NOT NULL,
                updated_at REAL NOT NULL, PRIMARY KEY(bot_id, asset_key)
                )""")

    async def preferences(self, user_id: int) -> E621Preferences:
        return await asyncio.to_thread(self._preferences_sync, user_id)

    def _preferences_sync(self, user_id: int) -> E621Preferences:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                "SELECT media_filter, search_order, result_count FROM search_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return (
            E621Preferences(
                cast(MediaFilter, row[0]), cast(SearchOrder, row[1]), row[2]
            )
            if row
            else E621Preferences()
        )

    async def save_preferences(
        self, user_id: int, preferences: E621Preferences
    ) -> None:
        if preferences.media_filter not in {"all", "images", "videos"}:
            raise ValueError("Неизвестный тип медиа")
        if (
            preferences.order not in {"site", "favcount", "random"}
            or not 1 <= preferences.count <= 10
        ):
            raise ValueError("Некорректная сортировка или количество")
        await asyncio.to_thread(self._save_preferences_sync, user_id, preferences)

    def _save_preferences_sync(
        self, user_id: int, preferences: E621Preferences
    ) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO search_preferences VALUES (?, ?, ?, ?)",
                (
                    user_id,
                    preferences.media_filter,
                    preferences.order,
                    preferences.count,
                ),
            )

    async def cached_media(self, bot_id: int, key: str) -> CachedMedia | None:
        return await asyncio.to_thread(self._cached_media_sync, bot_id, key)

    def _cached_media_sync(self, bot_id: int, key: str) -> CachedMedia | None:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                "SELECT file_id, kind, notice FROM media_cache WHERE bot_id = ? AND asset_key = ?",
                (bot_id, key),
            ).fetchone()
        return CachedMedia(*row) if row else None

    async def cache_media(
        self, bot_id: int, key: str, media: CachedMedia, at: float
    ) -> None:
        await asyncio.to_thread(self._cache_media_sync, bot_id, key, media, at)

    def _cache_media_sync(
        self, bot_id: int, key: str, media: CachedMedia, at: float
    ) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO media_cache VALUES (?, ?, ?, ?, ?, ?)",
                (bot_id, key, media.file_id, media.kind, media.notice, at),
            )
            connection.execute("""DELETE FROM media_cache WHERE rowid NOT IN
                (SELECT rowid FROM media_cache ORDER BY updated_at DESC LIMIT 2000)""")

    async def forget_media(self, bot_id: int, key: str) -> None:
        await asyncio.to_thread(self._forget_media_sync, bot_id, key)

    def _forget_media_sync(self, bot_id: int, key: str) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                "DELETE FROM media_cache WHERE bot_id = ? AND asset_key = ?",
                (bot_id, key),
            )

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
