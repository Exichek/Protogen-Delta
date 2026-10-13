"""Зарегистрированные Telegram ID в SQLite с однократным импортом JSON."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Lock


class UsersRepository:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "users.db"
        self._legacy_path = data_dir / "users.json"
        self._lock = Lock()
        self._initialized = False

    def _initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS registered_users (user_id INTEGER PRIMARY KEY, "
                    "position INTEGER NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS users_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                if not db.execute(
                    "SELECT 1 FROM users_meta WHERE key='legacy_imported'"
                ).fetchone():
                    data = (
                        json.loads(self._legacy_path.read_text("utf-8"))
                        if self._legacy_path.exists()
                        else {"USERS": []}
                    )
                    if not isinstance(data, dict):
                        raise TypeError(
                            "Корневой элемент users.json должен быть объектом"
                        )
                    users = data.get("USERS", [])
                    if not isinstance(users, list) or not all(
                        type(user_id) is int and 0 < user_id < 2**63
                        for user_id in users
                    ):
                        raise TypeError(
                            "Поле USERS должно содержать положительные Telegram ID"
                        )
                    db.executemany(
                        "INSERT OR IGNORE INTO registered_users VALUES (?, ?)",
                        ((user_id, position) for position, user_id in enumerate(users)),
                    )
                    db.execute("INSERT INTO users_meta VALUES ('legacy_imported', '1')")
            self._initialized = True

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._initialize()
        with closing(sqlite3.connect(self._path, timeout=10)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            yield db

    def get_all(self) -> list[int]:
        with self._connection() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT user_id FROM registered_users ORDER BY position,user_id"
                )
            ]

    def add(self, user_id: int) -> bool:
        if type(user_id) is not int or not 0 < user_id < 2**63:
            raise ValueError("Telegram ID должен быть положительным целым числом")
        with self._connection() as db:
            return bool(
                db.execute(
                    "INSERT OR IGNORE INTO registered_users(user_id,position) VALUES (?,(SELECT "
                    "COALESCE(MAX(position),-1)+1 FROM registered_users))",
                    (user_id,),
                ).rowcount
            )

    def remove(self, user_id: int) -> bool:
        with self._connection() as db:
            return bool(
                db.execute(
                    "DELETE FROM registered_users WHERE user_id=?", (user_id,)
                ).rowcount
            )

    def count(self) -> int:
        with self._connection() as db:
            return int(
                db.execute("SELECT COUNT(*) FROM registered_users").fetchone()[0]
            )
