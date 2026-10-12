"""Ограниченный журнал действительно доставленных сообщений создателя."""

import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CreatorMessageKind = Literal["message", "broadcast"]


@dataclass(frozen=True, slots=True)
class CreatorMessage:
    message_id: int
    text: str
    kind: CreatorMessageKind
    delivered_at: float


class CreatorMessagesRepository:
    """Держать до 20 сообщений на чат в существующей базе памяти."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "memories.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS creator_messages (
                chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                text TEXT NOT NULL, kind TEXT NOT NULL,
                delivered_at REAL NOT NULL, PRIMARY KEY(chat_id,message_id)
            )""")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS creator_messages_recent "
                "ON creator_messages(chat_id,delivered_at DESC,message_id DESC)"
            )

    async def record(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        kind: CreatorMessageKind,
        delivered_at: float,
    ) -> None:
        await asyncio.to_thread(
            self._record, chat_id, message_id, text[:4096], kind, delivered_at
        )

    def _record(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        kind: CreatorMessageKind,
        delivered_at: float,
    ) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO creator_messages VALUES(?,?,?,?,?)",
                (chat_id, message_id, text, kind, delivered_at),
            )
            connection.execute(
                "DELETE FROM creator_messages WHERE chat_id=? AND message_id NOT IN "
                "(SELECT message_id FROM creator_messages WHERE chat_id=? "
                "ORDER BY delivered_at DESC,message_id DESC LIMIT 20)",
                (chat_id, chat_id),
            )

    async def recent(self, chat_id: int) -> list[CreatorMessage]:
        return await asyncio.to_thread(self._recent, chat_id)

    def _recent(self, chat_id: int) -> list[CreatorMessage]:
        with closing(sqlite3.connect(self._path)) as connection:
            rows = connection.execute(
                "SELECT message_id,text,kind,delivered_at FROM creator_messages "
                "WHERE chat_id=? ORDER BY delivered_at DESC,message_id DESC LIMIT 20",
                (chat_id,),
            ).fetchall()
        return [CreatorMessage(*row) for row in rows]

    async def delete_user(self, user_id: int) -> None:
        await asyncio.to_thread(self._delete, user_id)

    def _delete(self, chat_id: int) -> None:
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                "DELETE FROM creator_messages WHERE chat_id=?", (chat_id,)
            )
