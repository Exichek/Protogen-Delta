"""Ограниченный журнал сообщений, доставленных вне обычного ответа движка."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutgoingMessage:
    id: int
    text: str
    kind: str
    delivered_at: float


class OutgoingMessagesRepository:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "memories.db"
        data_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS outgoing_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                text TEXT NOT NULL, kind TEXT NOT NULL, delivered_at REAL NOT NULL,
                UNIQUE(chat_id, message_id))""")
            db.execute(
                "CREATE INDEX IF NOT EXISTS outgoing_messages_chat "
                "ON outgoing_messages(chat_id,id)"
            )

    async def record(
        self, chat_id: int, message_id: int, text: str, kind: str, at: float
    ) -> None:
        await asyncio.to_thread(self._record, chat_id, message_id, text, kind, at)

    def _record(
        self, chat_id: int, message_id: int, text: str, kind: str, at: float
    ) -> None:
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO outgoing_messages "
                "(chat_id,message_id,text,kind,delivered_at) VALUES (?,?,?,?,?)",
                (chat_id, message_id, text[:4096], kind, at),
            )
            db.execute(
                "DELETE FROM outgoing_messages WHERE chat_id=? AND "
                "(delivered_at<? OR id NOT IN (SELECT id FROM outgoing_messages "
                "WHERE chat_id=? ORDER BY id DESC LIMIT 32))",
                (chat_id, at - 7 * 86400, chat_id),
            )

    async def pending(
        self, chat_id: int, after_id: int, now: float
    ) -> tuple[OutgoingMessage, ...]:
        return await asyncio.to_thread(self._pending, chat_id, after_id, now)

    def _pending(
        self, chat_id: int, after_id: int, now: float
    ) -> tuple[OutgoingMessage, ...]:
        with closing(sqlite3.connect(self._path)) as db:
            rows = db.execute(
                "SELECT id,text,kind,delivered_at FROM outgoing_messages "
                "WHERE chat_id=? AND id>? AND delivered_at>=? "
                "AND delivered_at<=? ORDER BY id DESC LIMIT 8",
                (
                    chat_id,
                    after_id,
                    now - 7 * 86400,
                    now,
                ),
            ).fetchall()
        return tuple(OutgoingMessage(*row) for row in reversed(rows))

    async def delete_user(self, chat_id: int) -> None:
        await asyncio.to_thread(self._delete, chat_id)

    def _delete(self, chat_id: int) -> None:
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute("DELETE FROM outgoing_messages WHERE chat_id=?", (chat_id,))
