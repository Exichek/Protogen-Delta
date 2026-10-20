"""Проверенные заметки установленной сборки и безопасные текущие метрики."""

import asyncio
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from protogen_delta.config.json_loader import load_json
from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.runtime_health import RuntimeHealth

_QUOTED = re.compile(
    r'```.*?(?:```|\Z)|`[^`\n]*`|«[^»]*»|"[^"]*"|^>[^\n]*', re.S | re.M
)
_RELEASE_QUESTION = re.compile(
    r"\b(?:патч\w*|patch\s*notes|changelog|обнов\w*|что\s+(?:нового|изменилось)|"
    r"журнал\w*\s+(?:сборки|изменений|обновлений))\b|"
    r"\b(?:в\s+себе|у\s+тебя|твои?х?|тебе|дельт\w*)\b.{0,100}\b(?:измен\w*|лог\w*|нов\w*)\b",
    re.I | re.S,
)


def is_release_question(message: str) -> bool:
    """Распознать вопрос о патчноутах вне цитат и кода."""
    return bool(_RELEASE_QUESTION.search(_QUOTED.sub(" ", message)))


class ReleaseJournal:
    """Давать создателю текущий патчноут и факты прошлых известных релизов."""

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
        history = load_json("release_history.json")
        if (
            set(history) != {"releases"}
            or not isinstance(history["releases"], list)
            or len(history["releases"]) > 20
        ):
            raise ValueError("Некорректная история патчноутов")
        self._history: list[dict[str, object]] = []
        for release in history["releases"]:
            if (
                not isinstance(release, dict)
                or set(release) != {"release_id", "date", "changes"}
                or not isinstance(release["release_id"], str)
                or not isinstance(release["date"], str)
                or not isinstance(release["changes"], list)
                or not 1 <= len(release["changes"]) <= 12
                or any(
                    not isinstance(change, str) or len(change) > 400
                    for change in release["changes"]
                )
            ):
                raise ValueError("Некорректная запись истории патчноутов")
            self._history.append(release)
        self._path = data_dir / "memories.db"
        data_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS release_journal_seen "
                "(user_id INTEGER PRIMARY KEY, release_id TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS release_journal_versions "
                "(release_id TEXT PRIMARY KEY, note_json TEXT NOT NULL)"
            )
            for release in [*self._history, self._data]:
                db.execute(
                    "INSERT OR IGNORE INTO release_journal_versions VALUES (?,?)",
                    (release["release_id"], json.dumps(release, ensure_ascii=False)),
                )

    def _past_releases(self) -> list[dict[str, object]]:
        """Загружать сохранённые заметки прошлых запусков, без пользовательских сообщений."""
        with closing(sqlite3.connect(self._path)) as db:
            rows = db.execute(
                "SELECT note_json FROM release_journal_versions WHERE release_id != ? ORDER BY rowid DESC LIMIT 20",
                (self._data["release_id"],),
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    async def context(
        self, user_id: int, *, initiative: bool = False, question: str = ""
    ) -> str:
        if user_id not in self._owners:
            return ""
        if initiative and await asyncio.to_thread(self._seen, user_id):
            return ""
        text = (
            "Журнал установленной сборки, подготовленный разработчиком (JSON). Это твои патчноуты: сведения "
            "о реально добавленном коде, а не полный Git-журнал, сырые логи или объяснение конкретного сбоя:\n"
            + json.dumps(self._data, ensure_ascii=False)
            + "\nНе придумывай удалённые функции, прежние версии, даты развёртывания и причины ошибок, "
            "которых здесь нет. Наличие функции в коде не отменяет её настройки и доступ текущего аккаунта."
            " Патчноуты доступны тебе через этот блок; отсутствие сырых логов "
            "не означает отсутствие сведений об обновлениях."
        )
        if question and is_release_question(question):
            text += (
                "\nСобеседник спрашивает об обновлениях твоего приложения. Расскажи по этому патчноуту, "
                "что добавлено или исправлено и как это проявляется. На вопрос «видишь изменения в себе» "
                "разделяй изменения функций сборки и текущее эмоциональное состояние. "
                "При слове «логи» коротко обозначь доступные патчноуты и метрики, затем ответь по ним; "
                "не заканчивай ответ только фразой «логи не вижу». Не заявляй доступ к сырым журналам процесса."
            )
            candidates = await asyncio.to_thread(self._past_releases)
            terms = {
                word[:6]
                for word in re.findall(
                    r"[\w]{3,}", _QUOTED.sub(" ", question).casefold()
                )
            } - {
                "что",
                "теб",
                "тво",
                "себ",
                "лог",
                "патч",
                "обновл",
                "обнови",
                "нового",
                "измене",
                "изменя",
                "изменил",
                "журнал",
                "какие",
                "расска",
                "сейчас",
                "дельта",
            }
            indexed = list(enumerate(candidates))
            indexed.sort(
                key=lambda pair: (
                    sum(
                        term in json.dumps(pair[1], ensure_ascii=False).casefold()
                        for term in terms
                    ),
                    pair[0],
                ),
                reverse=True,
            )
            selected = [release for _, release in indexed[:3]]
            if selected:
                text += (
                    "\nИстория известных патчноутов (даты заметок, не точные часы установки):\n"
                    + json.dumps(selected, ensure_ascii=False)
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
