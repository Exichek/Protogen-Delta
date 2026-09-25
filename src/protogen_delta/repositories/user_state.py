"""SQLite-хранилище долгоживущего состояния пользователей."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from protogen_delta.core.user_state import (
    EmotionalState,
    PersistentUserState,
    RelationshipState,
    UserStatePersistenceError,
)


class UserStateRepository:
    """Сохранять долгоживущие эмоции, отношения и RP-состояние пользователей."""

    def __init__(self, data_dir: Path) -> None:
        """Создать SQLite-базу и подготовить таблицу состояний."""
        self._path = data_dir / "user_states.db"

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    async def load(
        self,
        user_id: int,
    ) -> PersistentUserState | None:
        """Загрузить состояние без блокировки event loop."""
        return await asyncio.to_thread(
            self._load_sync,
            user_id,
        )

    async def save(
        self,
        user_id: int,
        *,
        emotions: EmotionalState,
        relationship: RelationshipState,
        emotions_updated_at: float,
        roleplay_active: bool,
        roleplay_configuration: str = "male",
        roleplay_character: str = "",
    ) -> None:
        """Сохранить состояние без блокировки event loop."""
        await asyncio.to_thread(
            self._save_sync,
            user_id,
            emotions=emotions,
            relationship=relationship,
            emotions_updated_at=emotions_updated_at,
            roleplay_active=roleplay_active,
            roleplay_configuration=roleplay_configuration,
            roleplay_character=roleplay_character,
        )

    async def delete(
        self,
        user_id: int,
    ) -> None:
        """Удалить долгоживущее состояние без блокировки event loop."""
        await asyncio.to_thread(
            self._delete_sync,
            user_id,
        )

    def _load_sync(
        self,
        user_id: int,
    ) -> PersistentUserState | None:
        """Синхронно загрузить состояние из SQLite."""
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                row = connection.execute(
                    """
                    SELECT
                        warmth,
                        irritation,
                        playfulness,
                        arousal,
                        familiarity,
                        trust,
                        affection,
                        resentment,
                        emotions_updated_at,
                        roleplay_active,
                        roleplay_configuration,
                        roleplay_character
                    FROM user_states
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise UserStatePersistenceError(
                f"Не удалось загрузить состояние пользователя {user_id}"
            ) from error

        if row is None:
            return None

        (
            warmth,
            irritation,
            playfulness,
            arousal,
            familiarity,
            trust,
            affection,
            resentment,
            emotions_updated_at,
            roleplay_active,
            roleplay_configuration,
            roleplay_character,
        ) = row

        return PersistentUserState(
            emotions=EmotionalState(
                warmth=warmth,
                irritation=irritation,
                playfulness=playfulness,
                arousal=arousal,
            ),
            relationship=RelationshipState(
                familiarity=familiarity,
                trust=trust,
                affection=affection,
                resentment=resentment,
            ),
            emotions_updated_at=emotions_updated_at,
            roleplay_active=bool(roleplay_active),
            roleplay_configuration=roleplay_configuration,
            roleplay_character=roleplay_character,
        )

    def _save_sync(
        self,
        user_id: int,
        *,
        emotions: EmotionalState,
        relationship: RelationshipState,
        emotions_updated_at: float,
        roleplay_active: bool,
        roleplay_configuration: str = "male",
        roleplay_character: str = "",
    ) -> None:
        """Синхронно сохранить состояние в SQLite."""
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO user_states (
                        user_id,
                        warmth,
                        irritation,
                        playfulness,
                        arousal,
                        familiarity,
                        trust,
                        affection,
                        resentment,
                        emotions_updated_at,
                        roleplay_active,
                        roleplay_configuration,
                        roleplay_character
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        warmth = excluded.warmth,
                        irritation = excluded.irritation,
                        playfulness = excluded.playfulness,
                        arousal = excluded.arousal,
                        familiarity = excluded.familiarity,
                        trust = excluded.trust,
                        affection = excluded.affection,
                        resentment = excluded.resentment,
                        emotions_updated_at = excluded.emotions_updated_at,
                        roleplay_active = excluded.roleplay_active,
                        roleplay_configuration = excluded.roleplay_configuration,
                        roleplay_character = excluded.roleplay_character
                    """,
                    (
                        user_id,
                        emotions.warmth,
                        emotions.irritation,
                        emotions.playfulness,
                        emotions.arousal,
                        relationship.familiarity,
                        relationship.trust,
                        relationship.affection,
                        relationship.resentment,
                        emotions_updated_at,
                        int(roleplay_active),
                        roleplay_configuration,
                        roleplay_character,
                    ),
                )
        except sqlite3.Error as error:
            raise UserStatePersistenceError(
                f"Не удалось сохранить состояние пользователя {user_id}"
            ) from error

    def _delete_sync(
        self,
        user_id: int,
    ) -> None:
        """Синхронно удалить долгоживущее состояние пользователя."""
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                connection.execute(
                    """
                    DELETE FROM user_states
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
        except sqlite3.Error as error:
            raise UserStatePersistenceError(
                f"Не удалось удалить состояние пользователя {user_id}"
            ) from error

    def _initialize(self) -> None:
        """Создать таблицу и применить совместимые изменения схемы."""
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS user_states (
                    user_id INTEGER PRIMARY KEY,
                    warmth REAL NOT NULL,
                    irritation REAL NOT NULL,
                    playfulness REAL NOT NULL,
                    arousal REAL NOT NULL,
                    familiarity REAL NOT NULL,
                    trust REAL NOT NULL,
                    affection REAL NOT NULL,
                    resentment REAL NOT NULL,
                    emotions_updated_at REAL NOT NULL,
                    roleplay_active INTEGER NOT NULL DEFAULT 0
                )
                """)

            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(user_states)")
            }

            if "emotions_updated_at" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN emotions_updated_at REAL NOT NULL DEFAULT 0.0
                    """)

            if "roleplay_active" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN roleplay_active INTEGER NOT NULL DEFAULT 0
                    """)

            if "roleplay_configuration" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN roleplay_configuration TEXT NOT NULL DEFAULT 'male'
                    """)

            if "roleplay_character" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN roleplay_character TEXT NOT NULL DEFAULT ''
                    """)
