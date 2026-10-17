"""Одноразовые поводы спросить о собственном плане пользователя."""

import asyncio
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.initiative_followup import (
    closed_followup_topics,
    closes_followups,
    extract_followup_plan,
    followup_preference,
    visible_plan_text,
)


@dataclass(frozen=True, slots=True)
class DueFollowup:
    id: int
    source: str
    event_day: str


class InitiativeFollowupsRepository:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "memories.db"
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS initiative_followups ("
                "id INTEGER PRIMARY KEY,user_id INTEGER NOT NULL REFERENCES engagement(user_id) ON DELETE CASCADE,"
                "topic TEXT NOT NULL,source TEXT NOT NULL,event_day TEXT NOT NULL,"
                "due_at REAL NOT NULL,expires_at REAL NOT NULL,selected_at REAL NOT NULL DEFAULT 0,"
                "closed_at REAL NOT NULL DEFAULT 0,UNIQUE(user_id,topic,event_day))"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS followups_owner_due ON initiative_followups(user_id,due_at)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS followups_expiry ON initiative_followups(expires_at)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS initiative_followup_preferences ("
                "user_id INTEGER PRIMARY KEY REFERENCES engagement(user_id) ON DELETE CASCADE,"
                "enabled INTEGER NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    async def observe(self, user_id: int, messages: Sequence[str], now: float) -> None:
        await finish_operation(
            asyncio.to_thread(self._observe, user_id, tuple(messages), now)
        )

    def _observe(self, user_id: int, messages: tuple[str, ...], now: float) -> None:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM initiative_followups WHERE expires_at<?", (now,))
            for message in messages[:8]:
                visible = visible_plan_text(message)
                preference = followup_preference(visible)
                if preference is not None:
                    db.execute(
                        "INSERT INTO initiative_followup_preferences VALUES (?,?) "
                        "ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled",
                        (user_id, int(preference)),
                    )
                if closes_followups(visible):
                    db.execute(
                        "UPDATE initiative_followups SET closed_at=? WHERE user_id=? AND closed_at=0",
                        (now, user_id),
                    )
                    continue
                for topic in closed_followup_topics(visible):
                    db.execute(
                        "UPDATE initiative_followups SET closed_at=? WHERE user_id=? AND topic=? AND closed_at=0",
                        (now, user_id, topic),
                    )
                plan = extract_followup_plan(message, now)
                if plan is None:
                    continue
                preference_row = db.execute(
                    "SELECT enabled FROM initiative_followup_preferences WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if preference_row and not preference_row[0]:
                    continue
                db.execute(
                    "UPDATE initiative_followups SET closed_at=? "
                    "WHERE user_id=? AND topic=? AND event_day<>? AND closed_at=0",
                    (now, user_id, plan.topic, plan.event_day),
                )
                active = db.execute(
                    "SELECT COUNT(*) FROM initiative_followups WHERE user_id=? AND closed_at=0 AND selected_at=0",
                    (user_id,),
                ).fetchone()[0]
                if active >= 5:
                    continue
                db.execute(
                    "INSERT OR IGNORE INTO initiative_followups"
                    "(user_id,topic,source,event_day,due_at,expires_at) VALUES (?,?,?,?,?,?)",
                    (
                        user_id,
                        plan.topic,
                        plan.source,
                        plan.event_day,
                        plan.due_at,
                        plan.expires_at,
                    ),
                )

    async def due(self, user_id: int, now: float) -> DueFollowup | None:
        return await asyncio.to_thread(self._due, user_id, now)

    def _due(self, user_id: int, now: float) -> DueFollowup | None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM initiative_followups WHERE expires_at<?", (now,))
            row = db.execute(
                "SELECT id,source,event_day FROM initiative_followups "
                "WHERE user_id=? AND selected_at=0 AND closed_at=0 AND due_at<=? AND expires_at>? "
                "AND COALESCE((SELECT enabled FROM initiative_followup_preferences WHERE user_id=?),1)=1 "
                "ORDER BY due_at,id LIMIT 1",
                (user_id, now, now, user_id),
            ).fetchone()
        return DueFollowup(*row) if row else None

    async def select(self, user_id: int, row_id: int, now: float) -> bool:
        """Одноразовый повод: после выбора не повторять даже при молчании/ошибке модели."""
        return await finish_operation(
            asyncio.to_thread(self._select, user_id, row_id, now)
        )

    def _select(self, user_id: int, row_id: int, now: float) -> bool:
        with closing(self._connect()) as db, db:
            return (
                db.execute(
                    "UPDATE initiative_followups SET selected_at=? WHERE id=? AND user_id=? "
                    "AND selected_at=0 AND closed_at=0 AND due_at<=? AND expires_at>?",
                    (now, row_id, user_id, now, now),
                ).rowcount
                == 1
            )

    async def forget_sources(self, user_id: int, sources: tuple[str, ...]) -> None:
        await finish_operation(
            asyncio.to_thread(self._forget_sources, user_id, sources)
        )

    async def close_matching(
        self, user_id: int, now: float, predicate: Callable[[str], bool]
    ) -> None:
        """Сохранить закрытие темы, даже когда её флаг уйдёт из короткой истории."""
        await finish_operation(
            asyncio.to_thread(self._close_matching, user_id, now, predicate)
        )

    def _close_matching(
        self, user_id: int, now: float, predicate: Callable[[str], bool]
    ) -> None:
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT id,source FROM initiative_followups WHERE user_id=? AND closed_at=0",
                (user_id,),
            ).fetchall()
            for row_id, source in rows:
                if predicate(source):
                    db.execute(
                        "UPDATE initiative_followups SET closed_at=? WHERE id=? AND user_id=?",
                        (now, row_id, user_id),
                    )

    def _forget_sources(self, user_id: int, sources: tuple[str, ...]) -> None:
        with closing(self._connect()) as db, db:
            if not sources:
                db.execute(
                    "DELETE FROM initiative_followups WHERE user_id=?", (user_id,)
                )
                return
            rows = db.execute(
                "SELECT id,source FROM initiative_followups WHERE user_id=?", (user_id,)
            ).fetchall()
            for row_id, text in rows:
                if any(
                    source.strip() and source.casefold() in text.casefold()
                    for source in sources
                ):
                    db.execute(
                        "DELETE FROM initiative_followups WHERE id=? AND user_id=?",
                        (row_id, user_id),
                    )
