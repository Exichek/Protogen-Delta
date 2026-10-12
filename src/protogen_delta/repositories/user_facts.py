"""Постоянные, подтверждённые словами пользователя факты."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FACT_LABELS = {
    "name": "Имя",
    "address_as": "Как обращаться",
    "occupation": "Занятие / работа",
    "education": "Учёба",
    "city": "Город",
    "music": "Музыка",
    "interests": "Интересы",
    "pets": "Питомцы",
    "projects": "Проекты",
    "preferences": "Предпочтения общения",
}
_SINGLE_FIELDS = {"name", "address_as", "occupation", "education", "city"}


@dataclass(frozen=True, slots=True)
class UserFact:
    key: str
    value: str
    source: str
    updated_at: float


@dataclass(frozen=True, slots=True)
class FactChange:
    key: str
    action: Literal["add", "replace", "forget"]
    value: str = ""
    source: str = ""


@dataclass(frozen=True, slots=True)
class FactsUpdate:
    changed: int = 0
    removed_sources: tuple[str, ...] = ()
    attempted: bool = False
    failed: bool = False
    forgotten: bool = False
    removed_values: tuple[str, ...] = ()


class UserFactsRepository:
    """Ограничить профиль 32 фактами, не вытесняя их обычной перепиской."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "user_facts.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS user_facts (
                user_id INTEGER NOT NULL, field TEXT NOT NULL,
                value_key TEXT NOT NULL, value TEXT NOT NULL,
                source TEXT NOT NULL, updated_at REAL NOT NULL,
                PRIMARY KEY(user_id, field, value_key)
            )""")

    async def all(self, user_id: int) -> list[UserFact]:
        return await asyncio.to_thread(self._all_sync, user_id)

    def _all_sync(self, user_id: int) -> list[UserFact]:
        with closing(sqlite3.connect(self._path)) as connection:
            rows = connection.execute(
                "SELECT field, value, source, updated_at FROM user_facts "
                "WHERE user_id = ? ORDER BY field, updated_at, value_key",
                (user_id,),
            ).fetchall()
        return [UserFact(*row) for row in rows]

    async def apply(
        self, user_id: int, changes: list[FactChange], at: float
    ) -> FactsUpdate:
        for change in changes:
            if change.key not in FACT_LABELS or change.action not in {
                "add",
                "replace",
                "forget",
            }:
                raise ValueError("Неизвестное поле или действие профиля")
            if change.action != "forget" and (
                not change.value.strip()
                or len(change.value) > 180
                or not change.source.strip()
                or len(change.source) > 240
            ):
                raise ValueError("Некорректный факт профиля")
        return await asyncio.to_thread(self._apply_sync, user_id, changes, at)

    def _apply_sync(
        self, user_id: int, changes: list[FactChange], at: float
    ) -> FactsUpdate:
        changed = 0
        removed: list[str] = []
        removed_values: list[str] = []
        with closing(sqlite3.connect(self._path)) as connection, connection:
            for change in changes:
                value = change.value.strip()
                rows = connection.execute(
                    "SELECT value_key, source, value FROM user_facts WHERE user_id=? AND field=?",
                    (user_id, change.key),
                ).fetchall()
                identity = value.casefold()
                if change.action != "forget" and any(r[0] == identity for r in rows):
                    if change.action == "add" or len(rows) == 1:
                        continue
                replace = change.action == "replace" or (
                    change.action != "forget" and change.key in _SINGLE_FIELDS
                )
                for old_identity, source, old_value in rows:
                    if replace or (
                        change.action == "forget"
                        and (not value or old_identity == identity)
                    ):
                        connection.execute(
                            "DELETE FROM user_facts WHERE user_id=? AND field=? AND value_key=?",
                            (user_id, change.key, old_identity),
                        )
                        removed.append(source)
                        removed_values.append(old_value)
                        changed += 1
                if change.action == "forget":
                    continue
                count = connection.execute(
                    "SELECT COUNT(*) FROM user_facts WHERE user_id=?", (user_id,)
                ).fetchone()[0]
                if count >= 32:
                    if change.key not in _SINGLE_FIELDS:
                        continue
                    oldest = connection.execute(
                        "SELECT field,value_key,source FROM user_facts WHERE user_id=? "
                        "AND field NOT IN ('name','address_as','occupation','education','city') "
                        "ORDER BY updated_at LIMIT 1",
                        (user_id,),
                    ).fetchone()
                    if oldest is None:
                        continue
                    connection.execute(
                        "DELETE FROM user_facts WHERE user_id=? AND field=? AND value_key=?",
                        (user_id, oldest[0], oldest[1]),
                    )
                    removed.append(oldest[2])
                    changed += 1
                connection.execute(
                    "INSERT OR REPLACE INTO user_facts VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, change.key, identity, value, change.source, at),
                )
                changed += 1
        return FactsUpdate(
            changed,
            tuple(dict.fromkeys(removed)),
            forgotten=any(c.action == "forget" for c in changes),
            removed_values=tuple(dict.fromkeys(removed_values)),
        )

    async def clear(self, user_id: int, key: str | None = None) -> FactsUpdate:
        facts = await self.all(user_id)
        fields = [key] if key is not None else list(dict.fromkeys(f.key for f in facts))
        return await self.apply(
            user_id, [FactChange(field, "forget") for field in fields], 0.0
        )
