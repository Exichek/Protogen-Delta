"""Ограничивать фоновые дополнения, сохраняя только время и счётчики."""

import asyncio
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time


class ReplySupplementLimits:
    """Попытка анализа тоже расходует окно: молчание не вызывает частые перепроверки."""

    def __init__(self, data_dir: Path, *, clock: Callable[[], float] = time) -> None:
        self._path = data_dir / "llm_usage.db"
        self._clock = clock
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS reply_supplement_limits ("
                "user_id INTEGER PRIMARY KEY,last_attempt REAL NOT NULL,"
                "day TEXT NOT NULL,attempts INTEGER NOT NULL)"
            )

    async def reserve(self, user_id: int) -> bool:
        return await asyncio.to_thread(self._reserve, user_id)

    def _reserve(self, user_id: int) -> bool:
        at = self._clock()
        day = (
            datetime.fromtimestamp(at, timezone(timedelta(hours=3))).date().isoformat()
        )
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT last_attempt,day,attempts FROM reply_supplement_limits WHERE user_id=?",
                (user_id,),
            ).fetchone()
            if row and (at - row[0] < 600 or row[1] == day and row[2] >= 4):
                return False
            count = row[2] + 1 if row and row[1] == day else 1
            db.execute(
                "INSERT INTO reply_supplement_limits VALUES (?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET last_attempt=excluded.last_attempt,"
                "day=excluded.day,attempts=excluded.attempts",
                (user_id, at, day, count),
            )
            return True
