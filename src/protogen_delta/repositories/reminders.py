"""Устойчивая очередь личных напоминаний в существующей memories.db."""

import asyncio
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from protogen_delta.core.async_completion import finish_operation

ReminderStatus = Literal["pending", "sending", "sent", "cancelled", "failed"]


@dataclass(frozen=True, slots=True)
class Reminder:
    id: int
    user_id: int
    chat_id: int
    source_message_id: int
    thread_id: int | None
    text: str
    due_at: float
    status: ReminderStatus
    attempts: int
    lease: str | None


_COLUMNS = (
    "id,user_id,chat_id,source_message_id,thread_id,text,due_at,status,attempts,lease"
)


class RemindersRepository:
    def __init__(
        self, data_dir: Path, *, per_user_limit: int = 20, total_limit: int = 1000
    ) -> None:
        if per_user_limit <= 0 or total_limit <= 0:
            raise ValueError("Лимиты напоминаний должны быть положительными")
        self._path = data_dir / "memories.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._per_user_limit, self._total_limit = per_user_limit, total_limit
        with closing(self._connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
                    source_message_id INTEGER NOT NULL, thread_id INTEGER,
                    text TEXT NOT NULL, due_at REAL NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending','sending','sent','cancelled','failed')),
                    next_attempt_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    lease TEXT, lease_until REAL,
                    UNIQUE(chat_id,source_message_id), CHECK(chat_id=user_id AND user_id>0)
                )
                """)
            db.execute(
                "CREATE INDEX IF NOT EXISTS reminders_due ON reminders(status,next_attempt_at)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS reminders_owner ON reminders(user_id,status)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS reminders_terminal ON reminders(status,updated_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

    async def create(
        self,
        user_id: int,
        chat_id: int,
        message_id: int,
        thread_id: int | None,
        text: str,
        due_at: float,
        now: float,
    ) -> Reminder:
        if (
            any(
                type(value) is not int or not 0 < value < 2**63
                for value in (user_id, chat_id, message_id)
            )
            or chat_id != user_id
        ):
            raise ValueError("Напоминание доступно только в собственном личном чате")
        if thread_id is not None and (
            type(thread_id) is not int or not 0 < thread_id < 2**63
        ):
            raise ValueError("Некорректная тема чата")
        if (
            not text.strip()
            or len(text) > 500
            or not math.isfinite(due_at)
            or not math.isfinite(now)
            or not now < due_at <= now + 365 * 86400
        ):
            raise ValueError("Некорректный текст или срок напоминания")
        return await finish_operation(
            asyncio.to_thread(
                self._create, user_id, chat_id, message_id, thread_id, text, due_at, now
            )
        )

    def _create(
        self,
        user_id: int,
        chat_id: int,
        message_id: int,
        thread_id: int | None,
        text: str,
        due_at: float,
        now: float,
    ) -> Reminder:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                f"SELECT {_COLUMNS} FROM reminders WHERE chat_id=? AND source_message_id=?",
                (chat_id, message_id),
            ).fetchone()
            if existing:
                return Reminder(*existing)
            counts = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(user_id=?),0) FROM reminders WHERE status IN ('pending','sending')",
                (user_id,),
            ).fetchone()
            if counts[1] >= self._per_user_limit:
                raise ValueError(
                    f"У тебя уже {self._per_user_limit} активных напоминаний. Отмени ненужное в /reminders"
                )
            if counts[0] >= self._total_limit:
                raise ValueError("Очередь напоминаний заполнена. Попробуй позже")
            row_id = db.execute(
                "INSERT INTO reminders(user_id,chat_id,source_message_id,thread_id,text,due_at,"
                "created_at,updated_at,next_attempt_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    user_id,
                    chat_id,
                    message_id,
                    thread_id,
                    text,
                    due_at,
                    now,
                    now,
                    due_at,
                ),
            ).lastrowid
            row = db.execute(
                f"SELECT {_COLUMNS} FROM reminders WHERE id=?", (row_id,)
            ).fetchone()
            assert row is not None
            return Reminder(*row)

    async def list_user(self, user_id: int) -> list[Reminder]:
        return await finish_operation(asyncio.to_thread(self._list_user, user_id))

    def _list_user(self, user_id: int) -> list[Reminder]:
        with closing(self._connect()) as db:
            rows = db.execute(
                f"SELECT {_COLUMNS} FROM reminders WHERE user_id=? AND status IN ('pending','sending','failed') "
                "ORDER BY CASE WHEN status='failed' THEN 1 ELSE 0 END,due_at,id LIMIT 30",
                (user_id,),
            ).fetchall()
        return [Reminder(*row) for row in rows]

    async def cancel(self, user_id: int, reminder_id: int, now: float) -> bool:
        return await finish_operation(
            asyncio.to_thread(self._cancel, user_id, reminder_id, now)
        )

    def _cancel(self, user_id: int, reminder_id: int, now: float) -> bool:
        with closing(self._connect()) as db, db:
            return bool(
                db.execute(
                    "UPDATE reminders SET status='cancelled',updated_at=?,lease=NULL,lease_until=NULL "
                    "WHERE id=? AND user_id=? AND status IN ('pending','sending','failed')",
                    (now, reminder_id, user_id),
                ).rowcount
            )

    async def delete_user(self, user_id: int) -> None:
        await finish_operation(asyncio.to_thread(self._delete_user, user_id))

    def _delete_user(self, user_id: int) -> None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM reminders WHERE user_id=?", (user_id,))

    async def claim_due(self, now: float) -> list[Reminder]:
        return await finish_operation(asyncio.to_thread(self._claim_due, now))

    def _claim_due(self, now: float) -> list[Reminder]:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "DELETE FROM reminders WHERE status IN ('sent','cancelled','failed') AND updated_at<?",
                (now - 90 * 86400,),
            )
            db.execute(
                "UPDATE reminders SET status='failed',updated_at=?,lease=NULL,lease_until=NULL WHERE attempts>=5 AND "
                "(status='pending' OR (status='sending' AND lease_until<=?))",
                (now, now),
            )
            rows = db.execute(
                f"WITH due AS (SELECT {_COLUMNS},ROW_NUMBER() OVER "
                "(PARTITION BY user_id ORDER BY due_at,id) AS position FROM reminders WHERE attempts<5 AND "
                "((status='pending' AND next_attempt_at<=?) OR (status='sending' AND lease_until<=?)) "
                f") SELECT {_COLUMNS} FROM due WHERE position=1 ORDER BY due_at,id LIMIT 10",
                (now, now),
            ).fetchall()
            claimed = []
            for row in rows:
                token = uuid4().hex
                db.execute(
                    "UPDATE reminders SET status='sending',lease=?,lease_until=?,"
                    "attempts=attempts+1,updated_at=? WHERE id=?",
                    (token, now + 120, now, row[0]),
                )
                claimed.append(
                    Reminder(
                        id=row[0],
                        user_id=row[1],
                        chat_id=row[2],
                        source_message_id=row[3],
                        thread_id=row[4],
                        text=row[5],
                        due_at=row[6],
                        status="sending",
                        attempts=row[8] + 1,
                        lease=token,
                    )
                )
            return claimed

    async def is_claimed(self, reminder: Reminder) -> bool:
        return await finish_operation(asyncio.to_thread(self._is_claimed, reminder))

    async def renew_claim(self, reminder: Reminder, now: float) -> bool:
        """Обновить аренду перед отправкой, если другой воркер её не перехватил."""
        return await finish_operation(
            asyncio.to_thread(self._renew_claim, reminder, now)
        )

    def _renew_claim(self, reminder: Reminder, now: float) -> bool:
        with closing(self._connect()) as db, db:
            return bool(
                db.execute(
                    "UPDATE reminders SET lease_until=? WHERE id=? AND user_id=? "
                    "AND status='sending' AND lease=?",
                    (now + 120, reminder.id, reminder.user_id, reminder.lease),
                ).rowcount
            )

    def _is_claimed(self, reminder: Reminder) -> bool:
        with closing(self._connect()) as db:
            return bool(
                db.execute(
                    "SELECT 1 FROM reminders WHERE id=? AND user_id=? AND status='sending' AND lease=?",
                    (reminder.id, reminder.user_id, reminder.lease),
                ).fetchone()
            )

    async def finish(
        self,
        reminder: Reminder,
        now: float,
        *,
        retry_at: float | None = None,
        failed: bool = False,
    ) -> None:
        await finish_operation(
            asyncio.to_thread(self._finish, reminder, now, retry_at, failed)
        )

    def _finish(
        self, reminder: Reminder, now: float, retry_at: float | None, failed: bool
    ) -> None:
        status = (
            "failed"
            if failed or retry_at is not None and reminder.attempts >= 5
            else "pending" if retry_at is not None else "sent"
        )
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE reminders SET status=?,updated_at=?,next_attempt_at=?,lease=NULL,lease_until=NULL "
                "WHERE id=? AND user_id=? AND status='sending' AND lease=?",
                (
                    status,
                    now,
                    retry_at or now,
                    reminder.id,
                    reminder.user_id,
                    reminder.lease,
                ),
            )
