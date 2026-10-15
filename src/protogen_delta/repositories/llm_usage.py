"""Атомарный суточный лимит попыток API и фактические токены из usage."""

import asyncio
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time

from protogen_delta.core.llm_budget import LlmDailyLimitError, LlmUsageUnavailableError

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

    def limit_for(self, user_id: int) -> int:
        """Выбрать квоту владельца запроса; режим, чат и текст её не меняют."""
        return self._limit_overrides.get(user_id, self.daily_limit)

    def _day(self) -> str:
        return datetime.fromtimestamp(self._clock(), MSK).date().isoformat()

    async def reserve(self, user_id: int) -> LlmReservation:
        return await asyncio.to_thread(self._reserve, user_id)

    def _reserve(self, user_id: int) -> LlmReservation:
        day = self._day()
        try:
            with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT OR IGNORE INTO llm_daily(user_id,day) VALUES (?,?)",
                    (user_id, day),
                )
                updated = db.execute(
                    "UPDATE llm_daily SET requests=requests+1,usage_missing=usage_missing+1 WHERE "
                    "user_id=? AND day=? AND requests<?",
                    (user_id, day, self.limit_for(user_id)),
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
