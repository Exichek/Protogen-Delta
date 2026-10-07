"""Счётчики сообщений без текста, файлов и содержимого профиля."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time

STATS_ZONE = timezone(timedelta(hours=3), "MSK")


@dataclass(frozen=True, slots=True)
class UserStatistics:
    user_id: int
    name: str
    username: str
    first_seen: float
    last_seen: float
    total: int
    tracking_started: float
    daily: tuple[tuple[str, int], ...]
    kinds: tuple[tuple[str, int], ...]


class UserStatisticsRepository:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "user_statistics.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS stats_users (
                user_id INTEGER PRIMARY KEY, name TEXT NOT NULL, username TEXT NOT NULL,
                first_seen REAL NOT NULL, last_seen REAL NOT NULL, total INTEGER NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS stats_daily (
                user_id INTEGER NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL,
                count INTEGER NOT NULL, PRIMARY KEY(user_id, day, kind)
            )""")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS stats_day ON stats_daily(day)"
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS stats_seen (
                chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL, at REAL NOT NULL,
                PRIMARY KEY(chat_id, message_id)
            )""")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS stats_meta (started REAL NOT NULL)"
            )
            if not connection.execute("SELECT 1 FROM stats_meta").fetchone():
                connection.execute("INSERT INTO stats_meta VALUES (?)", (time(),))

    async def record(
        self,
        user_id: int,
        name: str,
        username: str,
        chat_id: int,
        message_id: int,
        kind: str,
        at: float,
    ) -> None:
        await asyncio.to_thread(
            self._record_sync,
            user_id,
            name[:120],
            username[:64],
            chat_id,
            message_id,
            kind[:32],
            at,
        )

    def _record_sync(
        self,
        user_id: int,
        name: str,
        username: str,
        chat_id: int,
        message_id: int,
        kind: str,
        at: float,
    ) -> None:
        day = datetime.fromtimestamp(at, STATS_ZONE).date()
        with closing(sqlite3.connect(self._path)) as connection, connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO stats_seen VALUES (?, ?, ?)",
                (chat_id, message_id, at),
            ).rowcount
            if not inserted:
                return
            connection.execute(
                "INSERT INTO stats_users VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, username=excluded.username, "
                "first_seen=MIN(first_seen, excluded.first_seen), "
                "last_seen=MAX(last_seen, excluded.last_seen), total=total+1",
                (user_id, name, username, at, at),
            )
            connection.execute(
                "INSERT INTO stats_daily VALUES (?, ?, ?, 1) "
                "ON CONFLICT(user_id,day,kind) DO UPDATE SET count=count+1",
                (user_id, day.isoformat(), kind),
            )
            connection.execute("DELETE FROM stats_seen WHERE at < ?", (at - 7 * 86400,))
            connection.execute(
                "DELETE FROM stats_daily WHERE day < ?",
                ((day - timedelta(days=364)).isoformat(),),
            )

    async def get(self, user_id: int, now: float) -> UserStatistics | None:
        return await asyncio.to_thread(self._get_sync, user_id, now)

    async def delete_user(self, user_id: int) -> None:
        await asyncio.to_thread(self._delete_sync, user_id)

    def _delete_sync(self, user_id: int) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("DELETE FROM stats_users WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM stats_daily WHERE user_id=?", (user_id,))
            connection.execute("DELETE FROM stats_seen WHERE chat_id=?", (user_id,))

    def _get_sync(self, user_id: int, now: float) -> UserStatistics | None:
        start = (
            datetime.fromtimestamp(now, STATS_ZONE).date() - timedelta(days=364)
        ).isoformat()
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                "SELECT user_id,name,username,first_seen,last_seen,total FROM stats_users WHERE user_id=?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            started = connection.execute("SELECT started FROM stats_meta").fetchone()[0]
            daily = connection.execute(
                "SELECT day,SUM(count) FROM stats_daily WHERE user_id=? AND day>=? "
                "GROUP BY day ORDER BY day",
                (user_id, start),
            ).fetchall()
            kinds = connection.execute(
                "SELECT kind,SUM(count) FROM stats_daily WHERE user_id=? AND day>=? "
                "GROUP BY kind ORDER BY SUM(count) DESC",
                (user_id, start),
            ).fetchall()
        return UserStatistics(
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            started,
            tuple(daily),
            tuple(kinds),
        )
