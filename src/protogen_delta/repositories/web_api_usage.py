"""Ограничивать попытки веб-API без сохранения запросов, URL и идентификаторов."""

import asyncio
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time

MSK = timezone(timedelta(hours=3))


class WebApiLimitError(RuntimeError):
    """Отдельный лимит веб-инструментов исчерпан или учёт недоступен."""


class WebApiUsageRepository:
    def __init__(
        self,
        data_dir: Path,
        daily_limit: int = 200,
        brave_monthly_limit: int = 1000,
        *,
        clock: Callable[[], float] = time,
    ) -> None:
        if any(
            type(n) is not int or not 0 < n < 2**31
            for n in (daily_limit, brave_monthly_limit)
        ):
            raise ValueError("Лимиты веб-API должны быть положительными целыми числами")
        self._path = data_dir / "llm_usage.db"
        self._daily, self._monthly, self._clock = (
            daily_limit,
            brave_monthly_limit,
            clock,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS web_api_usage (day TEXT NOT NULL,"
                "provider TEXT NOT NULL,operation TEXT NOT NULL,requests INTEGER "
                "NOT NULL DEFAULT 0,PRIMARY KEY(day,provider,operation))"
            )

    async def reserve(self, provider: str, operation: str) -> None:
        """Атомарно учесть попытку до HTTP; ошибки тоже расходуют локальную квоту."""
        if (provider, operation) not in {
            ("deepseek", "search"),
            ("deepseek", "fetch"),
            ("brave", "search"),
        }:
            raise ValueError("Неизвестный веб-API")
        await asyncio.to_thread(self._reserve, provider, operation)

    def _reserve(self, provider: str, operation: str) -> None:
        day = datetime.fromtimestamp(self._clock(), MSK).date().isoformat()
        try:
            with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                total = db.execute(
                    "SELECT COALESCE(SUM(requests),0) FROM web_api_usage WHERE day=?",
                    (day,),
                ).fetchone()[0]
                if total >= self._daily:
                    raise WebApiLimitError("Суточный лимит веб-API исчерпан")
                if provider == "brave":
                    monthly = db.execute(
                        "SELECT COALESCE(SUM(requests),0) FROM web_api_usage "
                        "WHERE provider='brave' AND substr(day,1,7)=?",
                        (day[:7],),
                    ).fetchone()[0]
                    if monthly >= self._monthly:
                        raise WebApiLimitError(
                            "Месячный лимит резервного поиска исчерпан"
                        )
                db.execute(
                    "INSERT INTO web_api_usage VALUES (?,?,?,1) ON CONFLICT(day,provider,operation) "
                    "DO UPDATE SET requests=requests+1",
                    (day, provider, operation),
                )
        except sqlite3.Error:
            raise WebApiLimitError("Учёт веб-API временно недоступен") from None
