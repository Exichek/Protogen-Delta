"""Сохраняемое побуждение к инициативе и пределы самостоятельных отправлений."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.initiative_limits import DAILY_LIMIT, MIN_INTERVAL_SECONDS

MSK = timezone(timedelta(hours=3))


@dataclass(frozen=True, slots=True)
class InitiativeWindow:
    """Результат дешёвой проверки до обращения к модели."""

    ready: bool = False
    last_kind: str = ""
    news_available: bool = False


class InitiativeRepository:
    """Хранить инициативу в памяти бота, независимо от сбрасываемого таймера чата."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "memories.db"
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS initiative_state ("
                "user_id INTEGER PRIMARY KEY REFERENCES engagement(user_id) ON DELETE CASCADE,"
                "urge REAL NOT NULL DEFAULT 0,updated_at REAL NOT NULL,"
                "last_kind TEXT NOT NULL DEFAULT '',last_sent_at REAL NOT NULL DEFAULT 0,"
                "sent_day TEXT NOT NULL DEFAULT '',sent_today INTEGER NOT NULL DEFAULT 0,"
                "deferred_until REAL NOT NULL DEFAULT 0)"
            )
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(initiative_state)")
            }
            if "news_attempted_day" not in columns:
                db.execute(
                    "ALTER TABLE initiative_state ADD COLUMN news_attempted_day TEXT NOT NULL DEFAULT ''"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS initiative_news ("
                "user_id INTEGER NOT NULL REFERENCES engagement(user_id) ON DELETE CASCADE,"
                "url TEXT NOT NULL,sent_at REAL NOT NULL,PRIMARY KEY(user_id,url))"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    async def assess(self, user_id: int, at: float, drive: float) -> InitiativeWindow:
        """Накапливать побуждение по времени, ограничивая скачок после долгого простоя."""
        if (
            not isfinite(at)
            or at <= 0
            or not isfinite(drive)
            or not 0.5 <= drive <= 1.5
        ):
            raise ValueError("Некорректные часы или скорость инициативы")
        return await asyncio.to_thread(self._assess, user_id, at, drive)

    def _assess(self, user_id: int, at: float, drive: float) -> InitiativeWindow:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            activity = db.execute(
                "SELECT MAX(last_user_message_at,last_dialogue_activity_at),unanswered_count,"
                "proactive_enabled,delivery_suspended,COALESCE(last_proactive_at,0) "
                "FROM engagement WHERE user_id=?",
                (user_id,),
            ).fetchone()
            if not activity or activity[1] or not activity[2] or activity[3]:
                return InitiativeWindow()
            db.execute(
                "INSERT OR IGNORE INTO initiative_state(user_id,updated_at,last_sent_at,"
                "sent_day,sent_today) VALUES (?,?,?,?,?)",
                (
                    user_id,
                    max(activity[0], at - 21600),
                    activity[4],
                    (
                        datetime.fromtimestamp(activity[4], MSK).date().isoformat()
                        if activity[4]
                        else ""
                    ),
                    int(bool(activity[4])),
                ),
            )
            row = db.execute(
                "SELECT urge,updated_at,last_kind,last_sent_at,sent_day,sent_today,"
                "deferred_until,news_attempted_day FROM initiative_state WHERE user_id=?",
                (user_id,),
            ).fetchone()
            assert row is not None
            urge, updated, kind, sent, day, count, deferred, news_day = row
            if at < deferred or at - sent < MIN_INTERVAL_SECONDS and sent > 0:
                return InitiativeWindow(last_kind=kind)
            if (
                day == datetime.fromtimestamp(at, MSK).date().isoformat()
                and count >= DAILY_LIMIT
            ):
                return InitiativeWindow(last_kind=kind)
            urge = min(1.0, urge + min(21600, max(0, at - updated)) * drive / 10800)
            next_check = (
                at + max(30, min(900, (0.65 - urge) * 10800 / drive))
                if urge < 0.65
                else 0
            )
            db.execute(
                "UPDATE initiative_state SET urge=?,updated_at=?,deferred_until=? WHERE user_id=?",
                (urge, max(updated, at), next_check, user_id),
            )
            return InitiativeWindow(
                urge >= 0.65,
                kind,
                news_day != datetime.fromtimestamp(at, MSK).date().isoformat(),
            )

    async def claim_news(self, user_id: int, at: float, activity_version: int) -> bool:
        """Резервировать одну попытку за московские сутки, включая ошибку или молчание."""
        return await finish_operation(
            asyncio.to_thread(self._claim_news, user_id, at, activity_version)
        )

    def _claim_news(self, user_id: int, at: float, version: int) -> bool:
        day = datetime.fromtimestamp(at, MSK).date().isoformat()
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "UPDATE initiative_state SET news_attempted_day=? WHERE user_id=? "
                "AND news_attempted_day<>? AND EXISTS (SELECT 1 FROM engagement "
                "WHERE user_id=? AND activity_version=? AND proactive_enabled=1 "
                "AND delivery_suspended=0 AND unanswered_count=0)",
                (day, user_id, day, user_id, version),
            )
            return cursor.rowcount == 1

    async def seen_news(self, user_id: int, at: float) -> tuple[str, ...]:
        """Возвращать только публичные ссылки уже доставленных новостей за неделю."""
        return await asyncio.to_thread(self._seen_news, user_id, at)

    def _seen_news(self, user_id: int, at: float) -> tuple[str, ...]:
        with closing(self._connect()) as db:
            return tuple(
                row[0]
                for row in db.execute(
                    "SELECT url FROM initiative_news WHERE user_id=? AND sent_at>=? ORDER BY sent_at DESC LIMIT 30",
                    (user_id, at - 7 * 86400),
                )
            )

    async def defer(self, user_id: int, until: float, activity_version: int) -> None:
        """После молчания или ошибки не обращаться к модели на каждом тике."""
        await finish_operation(
            asyncio.to_thread(self._defer, user_id, until, activity_version)
        )

    def _defer(self, user_id: int, until: float, version: int) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE initiative_state SET deferred_until=? WHERE user_id=? AND EXISTS "
                "(SELECT 1 FROM engagement WHERE user_id=? AND activity_version=?)",
                (until, user_id, user_id, version),
            )

    async def note_sent(
        self, user_id: int, at: float, kind: str, news_url: str = ""
    ) -> None:
        """Снижать побуждение и учитывать предел только после подтверждения доставки."""
        await finish_operation(
            asyncio.to_thread(self._note_sent, user_id, at, kind, news_url)
        )

    def _note_sent(
        self, user_id: int, at: float, kind: str, news_url: str = ""
    ) -> None:
        day = datetime.fromtimestamp(at, MSK).date().isoformat()
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE initiative_state SET urge=0,updated_at=?,last_sent_at=?,last_kind=?,"
                "sent_today=CASE WHEN sent_day=? THEN sent_today+1 ELSE 1 END,"
                "sent_day=?,deferred_until=0 WHERE user_id=? AND last_sent_at<>?",
                (at, at, kind, day, day, user_id, at),
            )
            if kind == "news" and news_url:
                db.execute(
                    "DELETE FROM initiative_news WHERE user_id=? AND sent_at<?",
                    (user_id, at - 7 * 86400),
                )
                db.execute(
                    "INSERT OR IGNORE INTO initiative_news VALUES (?,?,?)",
                    (user_id, news_url, at),
                )
