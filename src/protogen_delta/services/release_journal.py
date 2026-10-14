"""Проверенные заметки установленной сборки и безопасные текущие метрики."""

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from protogen_delta.config.json_loader import load_json
from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.runtime_health import RuntimeHealth


class ReleaseJournal:
    def __init__(
        self,
        data_dir: Path,
        owners: frozenset[int],
        *,
        health: RuntimeHealth | None = None,
    ) -> None:
        self._owners = owners
        self._health = health
        data = load_json("release_journal.json")
        if (
            set(data) != {"release_id", "date", "changes"}
            or not isinstance(data["release_id"], str)
            or not 1 <= len(data["release_id"]) <= 100
            or not isinstance(data["date"], str)
            or not isinstance(data["changes"], list)
            or not 1 <= len(data["changes"]) <= 12
            or any(
                not isinstance(change, str) or not 1 <= len(change) <= 400
                for change in data["changes"]
            )
        ):
            raise ValueError("Некорректный журнал сборки")
        self._data = data
        self._path = data_dir / "memories.db"
        data_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS release_journal_seen "
                "(user_id INTEGER PRIMARY KEY, release_id TEXT NOT NULL)"
            )

    async def context(self, user_id: int, *, initiative: bool = False) -> str:
        if user_id not in self._owners:
            return ""
        if initiative and await asyncio.to_thread(self._seen, user_id):
            return ""
        text = (
            "Журнал установленной сборки, подготовленный разработчиком (JSON). Это сведения "
            "о реально добавленном коде, а не полный Git-журнал, сырые логи или объяснение конкретного сбоя:\n"
            + json.dumps(self._data, ensure_ascii=False)
            + "\nНе придумывай удалённые функции, прежние версии, даты развёртывания и причины ошибок, "
            "которых здесь нет. Наличие функции в коде не отменяет её настройки и доступ текущего аккаунта."
        )
        if initiative:
            text += (
                "\nЭта заметка новая для данного аккаунта. Если уместно, можно коротко спросить, "
                "как ему новая функция; не перечисляй весь журнал."
            )
        elif self._health is not None:
            health = self._health.snapshot()
            safe = {
                key: health[key]
                for key in ("ok", "polling", "polling_errors", "polling_age_seconds")
            }
            text += (
                "\nТекущие метрики процесса, без сообщений и секретов: "
                + json.dumps(safe, ensure_ascii=False)
            )
            text += ". Счётчик ошибок не устанавливает причину недоставки конкретной реплики."
        return text

    def _seen(self, user_id: int) -> bool:
        with closing(sqlite3.connect(self._path)) as db:
            row = db.execute(
                "SELECT release_id FROM release_journal_seen WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return bool(row == (self._data["release_id"],))

    async def acknowledge(self, user_id: int) -> None:
        if user_id in self._owners:
            await finish_operation(asyncio.to_thread(self._acknowledge, user_id))

    def _acknowledge(self, user_id: int) -> None:
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute(
                "INSERT INTO release_journal_seen VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET "
                "release_id=excluded.release_id",
                (user_id, self._data["release_id"]),
            )
