"""Устойчивые идентификаторы обработки без сохранения текста переписки."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from time import time

from protogen_delta.core.async_completion import finish_operation


class DeliveryLedgerRepository:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "delivery_ledger.db"
        data_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS delivery_ledger (
                    source_key TEXT PRIMARY KEY, owner INTEGER NOT NULL,
                    state TEXT NOT NULL, received_at REAL NOT NULL,
                    updated_at REAL NOT NULL, confirmed_chunks INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS delivery_owner ON delivery_ledger(owner, received_at);
                UPDATE delivery_ledger SET state='interrupted'
                    WHERE state IN ('received','processing');
            """)
        self._path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=10)
        db.execute("PRAGMA synchronous=FULL")
        return db

    async def claim(self, owner: int, key: str) -> bool:
        if (
            type(owner) is not int
            or not 0 < owner < 2**63
            or not key.startswith("telegram:")
        ):
            raise ValueError("Некорректный источник запроса")

        def insert() -> bool:
            now = time()
            with closing(self._connect()) as db, db:
                # Служебные ID без текста: месяц дедупликации, затем удаление.
                db.execute(
                    "DELETE FROM delivery_ledger WHERE received_at<?",
                    (now - 30 * 86400,),
                )
                return bool(
                    db.execute(
                        "INSERT OR IGNORE INTO delivery_ledger(source_key,owner,state,received_at,updated_at) "
                        "VALUES(?,?,'received',?,?)",
                        (key, owner, now, now),
                    ).rowcount
                )

        return await finish_operation(asyncio.to_thread(insert))

    async def mark(self, key: str, state: str, confirmed: int = 0) -> None:
        if state not in {
            "processing",
            "answered",
            "handled",
            "deferred",
            "failed",
            "partial",
            "delivery_unknown",
            "interrupted",
            "forgotten",
        }:
            raise ValueError("Неизвестное состояние доставки")

        def update() -> None:
            with closing(self._connect()) as db, db:
                db.execute(
                    "UPDATE delivery_ledger SET state=?,updated_at=?,confirmed_chunks=? "
                    "WHERE source_key=? AND state<>'forgotten'",
                    (state, time(), confirmed, key),
                )

        await finish_operation(asyncio.to_thread(update))

    async def forget(self, owners: frozenset[int]) -> None:
        def update() -> None:
            with closing(self._connect()) as db, db:
                for owner in owners:
                    db.execute(
                        "UPDATE delivery_ledger SET state='forgotten',confirmed_chunks=0 "
                        "WHERE owner=?",
                        (owner,),
                    )

        await finish_operation(asyncio.to_thread(update))

    async def pending_keys(
        self, owner: int, exclude: str | None
    ) -> list[tuple[str, str]]:
        def read() -> list[tuple[str, str]]:
            with closing(self._connect()) as db:
                return db.execute(
                    "SELECT source_key,state FROM delivery_ledger WHERE owner=? "
                    "AND state IN ('deferred','failed','partial','delivery_unknown','interrupted') "
                    "AND (? IS NULL OR source_key<>?) ORDER BY received_at DESC LIMIT 3",
                    (owner, exclude, exclude),
                ).fetchall()

        return await asyncio.to_thread(read)
