"""Полный текст личной переписки, полнотекстовый поиск и переносимые снимки."""

import asyncio
import json
import math
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.memory_search import memory_terms


@dataclass(frozen=True, slots=True)
class DialogueMessage:
    """Одна реплика; источник позволяет отличать Telegram от импортированной истории."""

    source_key: str
    role: Literal["user", "assistant"]
    text: str
    created_at: float
    kind: str = "text"


class DialogueArchiveRepository:
    """Хранить сообщения без ограничения числом ходов и без TTL контекста модели."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "dialogue_archive.db"
        data_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            index_exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='dialogue_fts'"
            ).fetchone()
            db.executescript("""
                CREATE TABLE IF NOT EXISTS dialogue_messages (
                    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
                    source_key TEXT NOT NULL, role TEXT NOT NULL,
                    text TEXT NOT NULL, created_at REAL NOT NULL, kind TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    UNIQUE(user_id, source_key)
                );
                CREATE INDEX IF NOT EXISTS dialogue_owner_time
                    ON dialogue_messages(user_id, created_at, id);
                CREATE VIRTUAL TABLE IF NOT EXISTS dialogue_fts USING fts5(
                    search_text, content='dialogue_messages', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS dialogue_insert AFTER INSERT ON dialogue_messages
                BEGIN INSERT INTO dialogue_fts(rowid,search_text)
                    VALUES(new.id,new.search_text); END;
                CREATE TRIGGER IF NOT EXISTS dialogue_delete AFTER DELETE ON dialogue_messages
                BEGIN INSERT INTO dialogue_fts(dialogue_fts,rowid,search_text)
                    VALUES('delete',old.id,old.search_text); END;
                CREATE TRIGGER IF NOT EXISTS dialogue_update AFTER UPDATE ON dialogue_messages
                BEGIN
                    INSERT INTO dialogue_fts(dialogue_fts,rowid,search_text)
                        VALUES('delete',old.id,old.search_text);
                    INSERT INTO dialogue_fts(rowid,search_text) VALUES(new.id,new.search_text);
                END;
                """)
            if not index_exists:
                db.execute("INSERT INTO dialogue_fts(dialogue_fts) VALUES('rebuild')")
        self._path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def _owner(user_id: int) -> None:
        if type(user_id) is not int or not 0 < user_id < 2**63:
            raise ValueError("Нужен положительный Telegram ID")

    @staticmethod
    def _validate(message: DialogueMessage) -> None:
        if (
            not isinstance(message.source_key, str)
            or not 1 <= len(message.source_key) <= 200
            or message.role not in {"user", "assistant"}
            or not isinstance(message.text, str)
            or not 1 <= len(message.text) <= 65536
            or type(message.created_at) not in {int, float}
            or not math.isfinite(message.created_at)
            or message.created_at < 0
            or message.created_at > 253402300799
            or not isinstance(message.kind, str)
            or not 1 <= len(message.kind) <= 32
        ):
            raise ValueError("Некорректная реплика архива")

    @staticmethod
    def _index(text: str) -> str:
        # Индексировать также конец длинной реплики, а не только первые 8000 знаков.
        terms: list[str] = []
        for offset in range(0, len(text), 4000):
            end = offset + 4064
            terms.extend(memory_terms(text[offset:end], limit=2048))
        return " ".join(terms)

    async def save(self, user_id: int, message: DialogueMessage) -> None:
        self._owner(user_id)
        self._validate(message)
        await finish_operation(asyncio.to_thread(self._save_sync, user_id, message))

    async def save_many(self, user_id: int, messages: list[DialogueMessage]) -> int:
        """Валидировать и импортировать выгрузку одной транзакцией."""
        self._owner(user_id)
        for message in messages:
            self._validate(message)
        return await finish_operation(
            asyncio.to_thread(self._save_many_sync, user_id, messages)
        )

    def _save_many_sync(self, user_id: int, messages: list[DialogueMessage]) -> int:
        with closing(self._connect()) as db, db:
            for message in messages:
                self._put(db, user_id, message)
        return len(messages)

    def _save_sync(self, user_id: int, message: DialogueMessage) -> None:
        with closing(self._connect()) as db, db:
            self._put(db, user_id, message)

    def _put(
        self, db: sqlite3.Connection, user_id: int, message: DialogueMessage
    ) -> None:
        db.execute(
            "INSERT INTO dialogue_messages(user_id,source_key,role,text,created_at,kind,search_text) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id,source_key) DO UPDATE SET "
            "role=excluded.role,text=excluded.text,kind=excluded.kind,search_text=excluded.search_text",
            (
                user_id,
                message.source_key,
                message.role,
                message.text,
                message.created_at,
                message.kind,
                self._index(message.text),
            ),
        )

    async def search(
        self,
        user_id: int,
        query: str,
        *,
        limit: int = 4,
        exclude_key: str | None = None,
    ) -> list[DialogueMessage]:
        self._owner(user_id)
        terms = memory_terms(query, limit=16)
        if not terms or limit <= 0:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        return await asyncio.to_thread(
            self._search_sync, user_id, expression, min(limit, 6), exclude_key
        )

    def _search_sync(
        self, user_id: int, expression: str, limit: int, exclude_key: str | None
    ) -> list[DialogueMessage]:
        with closing(self._connect()) as db:
            hits = db.execute(
                "SELECT m.id FROM dialogue_fts JOIN dialogue_messages m ON m.id=dialogue_fts.rowid "
                "WHERE dialogue_fts MATCH ? AND m.user_id=? AND (? IS NULL OR m.source_key<>?) "
                "ORDER BY bm25(dialogue_fts),m.created_at DESC,m.id DESC LIMIT ?",
                (expression, user_id, exclude_key, exclude_key, limit),
            ).fetchall()
            result: dict[int, DialogueMessage] = {}
            for (row_id,) in hits:
                row = db.execute(
                    "SELECT source_key,role,text,created_at,kind FROM dialogue_messages "
                    "WHERE id=? AND user_id=?",
                    (row_id, user_id),
                ).fetchone()
                if row is None:
                    continue
                result[row_id] = DialogueMessage(*row)
                # Одна соседняя реплика объясняет ответ/вопрос; другой владелец исключён.
                direction = ">" if row[1] == "user" else "<"
                order = "ASC" if direction == ">" else "DESC"
                adjacent = db.execute(
                    "SELECT id,source_key,role,text,created_at,kind FROM dialogue_messages "
                    f"WHERE user_id=? AND (? IS NULL OR source_key<>?) AND (created_at,id) {direction} (?,?) "
                    f"ORDER BY created_at {order},id {order} LIMIT 1",
                    (user_id, exclude_key, exclude_key, row[3], row_id),
                ).fetchone()
                if adjacent and abs(adjacent[4] - row[3]) <= 600:
                    result[adjacent[0]] = DialogueMessage(*adjacent[1:])
            return [
                item
                for _, item in sorted(
                    result.items(), key=lambda pair: (pair[1].created_at, pair[0])
                )
            ]

    async def delete_user(self, user_id: int) -> None:
        self._owner(user_id)
        await finish_operation(asyncio.to_thread(self._delete_sync, user_id, ()))

    async def forget_sources(self, user_id: int, sources: tuple[str, ...]) -> None:
        self._owner(user_id)
        if sources:
            await finish_operation(
                asyncio.to_thread(self._delete_sync, user_id, sources)
            )

    def _delete_sync(self, user_id: int, sources: tuple[str, ...]) -> None:
        with closing(self._connect()) as db, db:
            if not sources:
                db.execute("DELETE FROM dialogue_messages WHERE user_id=?", (user_id,))
                return
            needles = tuple(s.casefold() for s in sources if s)
            term_sets = tuple(set(memory_terms(s)) for s in sources if s)
            for row_id, text in db.execute(
                "SELECT id,text FROM dialogue_messages WHERE user_id=?", (user_id,)
            ):
                words = set(memory_terms(text, limit=2048))
                if any(s in text.casefold() for s in needles) or any(
                    terms and terms <= words for terms in term_sets
                ):
                    db.execute(
                        "DELETE FROM dialogue_messages WHERE id=? AND user_id=?",
                        (row_id, user_id),
                    )

    async def export(self, user_id: int, destination: Path) -> int:
        self._owner(user_id)
        return await finish_operation(
            asyncio.to_thread(self._export_sync, user_id, destination)
        )

    def _export_sync(self, user_id: int, destination: Path) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        count = 0
        try:
            with (
                closing(self._connect()) as db,
                db,
                temporary.open("w", encoding="utf-8") as output,
            ):
                temporary.chmod(0o600)
                output.write(json.dumps({"schema": 1, "user_id": user_id}) + "\n")
                for row in db.execute(
                    "SELECT source_key,role,text,created_at,kind FROM dialogue_messages "
                    "WHERE user_id=? ORDER BY created_at,id",
                    (user_id,),
                ):
                    output.write(
                        json.dumps(asdict(DialogueMessage(*row)), ensure_ascii=False)
                        + "\n"
                    )
                    count += 1
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return count

    async def import_snapshot(self, user_id: int, source: Path) -> int:
        self._owner(user_id)
        return await finish_operation(
            asyncio.to_thread(self._import_sync, user_id, source)
        )

    def _import_sync(self, user_id: int, source: Path) -> int:
        count = 0
        with (
            source.open(encoding="utf-8") as input_file,
            closing(self._connect()) as db,
            db,
        ):
            header = json.loads(input_file.readline(1024))
            if (
                header != {"schema": 1, "user_id": user_id}
                or type(header.get("user_id")) is not int
            ):
                raise ValueError("Снимок принадлежит другому пользователю или версии")
            while line := input_file.readline(500000):
                value = json.loads(line)
                message = DialogueMessage(**value)
                self._validate(message)
                self._put(db, user_id, message)
                count += 1
        return count
