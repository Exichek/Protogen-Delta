"""Атомарный суточный лимит попыток API и фактические токены из usage."""

import asyncio
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time

from protogen_delta.core.llm_budget import (
    LlmDailyLimitError,
    LlmGlobalLimitError,
    LlmUsageUnavailableError,
)

MSK = timezone(timedelta(hours=3), "MSK")


@dataclass(frozen=True, slots=True)
class LlmReservation:
    id: int
    user_id: int
    day: str


class LlmUsageRepository:
    def __init__(
        self,
        data_dir: Path,
        daily_limit: int = 300,
        *,
        clock: Callable[[], float] = time,
        limit_overrides: Mapping[int, int] | None = None,
        global_daily_limit: int = 3000,
    ) -> None:
        if type(daily_limit) is not int or not 0 < daily_limit < 2**63:
            raise ValueError("Суточный лимит должен быть положительным целым числом")
        self._limit_overrides = dict(limit_overrides or {})
        if any(
            type(owner) is not int
            or not 0 < owner < 2**63
            or type(limit) is not int
            or not 0 < limit < 2**63
            for owner, limit in self._limit_overrides.items()
        ):
            raise ValueError(
                "Персональные лимиты требуют положительных ID и целых квот"
            )
        self._path = data_dir / "llm_usage.db"
        self.daily_limit = daily_limit
        if type(global_daily_limit) is not int or not 0 < global_daily_limit < 2**63:
            raise ValueError(
                "Общий суточный лимит должен быть положительным целым числом"
            )
        self.global_daily_limit = global_daily_limit
        self._clock = clock
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS llm_daily (user_id INTEGER NOT NULL,day TEXT NOT "
                "NULL,requests INTEGER NOT NULL DEFAULT 0,prompt_tokens INTEGER NOT NULL "
                "DEFAULT 0,completion_tokens INTEGER NOT NULL DEFAULT 0,usage_missing INTEGER "
                "NOT NULL DEFAULT 0,PRIMARY KEY(user_id,day))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS llm_calls (id INTEGER PRIMARY KEY,user_id INTEGER "
                "NOT NULL,day TEXT NOT NULL,finished INTEGER NOT NULL DEFAULT 0)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS llm_calls_day ON llm_calls(day)")
            db.execute("CREATE INDEX IF NOT EXISTS llm_daily_day ON llm_daily(day)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS llm_global_alerts "
                "(day TEXT NOT NULL, threshold INTEGER NOT NULL, PRIMARY KEY(day,threshold))"
            )

    def limit_for(self, user_id: int) -> int:
        """Выбрать квоту владельца запроса; режим, чат и текст её не меняют."""
        return self._limit_overrides.get(user_id, self.daily_limit)

    def _day(self) -> str:
        return datetime.fromtimestamp(self._clock(), MSK).date().isoformat()

    async def reserve(self, user_id: int) -> LlmReservation:
        return await asyncio.to_thread(self._reserve, user_id)

    def _reserve(self, user_id: int) -> LlmReservation:
        if type(user_id) is not int or not 0 <= user_id < 2**63:
            raise LlmUsageUnavailableError()
        day = self._day()
        try:
            with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                total = db.execute(
                    "SELECT COALESCE(SUM(requests),0) FROM llm_daily WHERE day=?",
                    (day,),
                ).fetchone()[0]
                if total >= self.global_daily_limit:
                    raise LlmGlobalLimitError()
                db.execute(
                    "INSERT OR IGNORE INTO llm_daily(user_id,day) VALUES (?,?)",
                    (user_id, day),
                )
                updated = db.execute(
                    "UPDATE llm_daily SET requests=requests+1,usage_missing=usage_missing+1 WHERE "
                    "user_id=? AND day=? AND requests<?",
                    (
                        user_id,
                        day,
                        self.limit_for(user_id) if user_id else self.global_daily_limit,
                    ),
                ).rowcount
                if not updated:
                    raise LlmDailyLimitError()
                receipt = db.execute(
                    "INSERT INTO llm_calls(user_id,day) VALUES (?,?)", (user_id, day)
                ).lastrowid
                assert receipt is not None
                # Receipts only prevent duplicate accounting; retain 90 days.
                older = (
                    datetime.fromtimestamp(self._clock(), MSK).date()
                    - timedelta(days=90)
                ).isoformat()
                db.execute("DELETE FROM llm_calls WHERE day<?", (older,))
                return LlmReservation(receipt, user_id, day)
        except sqlite3.Error as error:
            raise LlmUsageUnavailableError() from error

    def _global_snapshot(
        self, db: sqlite3.Connection, day: str
    ) -> dict[str, int | str]:
        row = db.execute(
            "SELECT COALESCE(SUM(requests),0),COALESCE(SUM(prompt_tokens),0),"
            "COALESCE(SUM(completion_tokens),0),COALESCE(SUM(usage_missing),0) "
            "FROM llm_daily WHERE day=?",
            (day,),
        ).fetchone()
        return {
            "day": day,
            "timezone": "Europe/Moscow",
            "limit": self.global_daily_limit,
            "requests": row[0],
            "remaining": max(0, self.global_daily_limit - row[0]),
            "prompt_tokens": row[1],
            "completion_tokens": row[2],
            "usage_missing": row[3],
        }

    async def get_global(self) -> dict[str, int | str]:
        def read() -> dict[str, int | str]:
            with closing(sqlite3.connect(self._path, timeout=10)) as db:
                return self._global_snapshot(db, self._day())

        return await asyncio.to_thread(read)

    async def claim_global_alert(self) -> dict[str, int | str] | None:
        """Разрешить одну попытку уведомления на порог и московский день, даже после рестарта."""

        def claim() -> dict[str, int | str] | None:
            day = self._day()
            with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                snapshot = self._global_snapshot(db, day)
                count = int(snapshot["requests"])
                threshold = (
                    100
                    if count >= self.global_daily_limit
                    else 80 if count * 5 >= self.global_daily_limit * 4 else 0
                )
                if not threshold:
                    return None
                changed = db.execute(
                    "INSERT OR IGNORE INTO llm_global_alerts VALUES (?,?)",
                    (day, threshold),
                ).rowcount
                if threshold == 100:
                    db.execute(
                        "INSERT OR IGNORE INTO llm_global_alerts VALUES (?,80)", (day,)
                    )
                return snapshot | {"threshold": threshold} if changed else None

        return await asyncio.to_thread(claim)

    async def record_usage(
        self,
        receipt: LlmReservation,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        await asyncio.to_thread(
            self._record_usage, receipt, prompt_tokens, completion_tokens
        )

    def _record_usage(
        self,
        receipt: LlmReservation,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        known = (
            type(prompt_tokens) is int
            and prompt_tokens >= 0
            and type(completion_tokens) is int
            and completion_tokens >= 0
        )
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            updated = db.execute(
                "UPDATE llm_calls SET finished=1 WHERE id=? AND user_id=? AND day=? AND finished=0",
                (receipt.id, receipt.user_id, receipt.day),
            ).rowcount
            if updated and known:
                db.execute(
                    "UPDATE llm_daily SET "
                    "prompt_tokens=prompt_tokens+?,completion_tokens=completion_tokens+?,usage_missing=usage_missing-1"
                    " WHERE user_id=? AND day=?",
                    (prompt_tokens, completion_tokens, receipt.user_id, receipt.day),
                )

    async def get(self, user_id: int) -> dict[str, int | str]:
        return await asyncio.to_thread(self._get, user_id)

    def _get(self, user_id: int) -> dict[str, int | str]:
        day = self._day()
        with closing(sqlite3.connect(self._path, timeout=10)) as db:
            row = db.execute(
                "SELECT requests,prompt_tokens,completion_tokens,usage_missing FROM llm_daily "
                "WHERE user_id=? AND day=?",
                (user_id, day),
            ).fetchone() or (0, 0, 0, 0)
        limit = self.limit_for(user_id)
        return {
            "day": day,
            "timezone": "Europe/Moscow",
            "limit": limit,
            "requests": row[0],
            "remaining": max(0, limit - row[0]),
            "prompt_tokens": row[1],
            "completion_tokens": row[2],
            "usage_missing": row[3],
        }
