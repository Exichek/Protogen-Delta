"""SQLite-хранилище долгоживущего состояния пользователей."""

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import cast

from protogen_delta.core.user_state import (
    ContentMode,
    EmotionalState,
    PersistentUserState,
    RelationshipState,
    StateKey,
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
        user_id: StateKey,
    ) -> PersistentUserState | None:
        """Загрузить состояние без блокировки event loop."""
        return await asyncio.to_thread(
            self._load_sync,
            user_id,
        )

    async def save(
        self,
        user_id: StateKey,
        *,
        emotions: EmotionalState,
        relationship: RelationshipState,
        emotions_updated_at: float,
        roleplay_active: bool,
        roleplay_configuration: str = "male",
        roleplay_character: str = "",
        roleplay_fetishes: tuple[str, ...] = (),
        roleplay_preferences: str = "",
        roleplay_boundaries: str = "",
        delta_appearance: str = "",
        content_mode: ContentMode = "unselected",
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
            roleplay_fetishes=roleplay_fetishes,
            roleplay_preferences=roleplay_preferences,
            roleplay_boundaries=roleplay_boundaries,
            delta_appearance=delta_appearance,
            content_mode=content_mode,
        )

    async def delete(
        self,
        user_id: StateKey,
    ) -> None:
        """Удалить долгоживущее состояние без блокировки event loop."""
        await asyncio.to_thread(
            self._delete_sync,
            user_id,
        )

    @staticmethod
    def _scope(key: StateKey) -> tuple[str, str, tuple[int, ...]]:
        """Выбирать только фиксированные SQL-идентификаторы, ключи параметризованы."""
        if isinstance(key, tuple):
            return "conversation_states", "chat_id = ? AND user_id = ?", key
        return "user_states", "user_id = ?", (key,)

    def _load_sync(
        self,
        user_id: StateKey,
    ) -> PersistentUserState | None:
        """Синхронно загрузить состояние из SQLite."""
        table, where, key = self._scope(user_id)
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                row = connection.execute(
                    f"""
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
                        roleplay_character,
                        roleplay_fetishes,
                        roleplay_preferences,
                        roleplay_boundaries,
                        delta_appearance,
                        content_mode
                    FROM {table}
                    WHERE {where}
                    """,
                    key,
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
            roleplay_fetishes,
            roleplay_preferences,
            roleplay_boundaries,
            delta_appearance,
            content_mode,
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
            roleplay_fetishes=self._decode_fetishes(roleplay_fetishes),
            roleplay_preferences=roleplay_preferences,
            roleplay_boundaries=roleplay_boundaries,
            delta_appearance=delta_appearance,
            content_mode=self._decode_content_mode(content_mode),
        )

    def _save_sync(
        self,
        user_id: StateKey,
        *,
        emotions: EmotionalState,
        relationship: RelationshipState,
        emotions_updated_at: float,
        roleplay_active: bool,
        roleplay_configuration: str = "male",
        roleplay_character: str = "",
        roleplay_fetishes: tuple[str, ...] = (),
        roleplay_preferences: str = "",
        roleplay_boundaries: str = "",
        delta_appearance: str = "",
        content_mode: ContentMode = "unselected",
    ) -> None:
        """Синхронно сохранить состояние в SQLite."""
        table, _, key = self._scope(user_id)
        group = isinstance(user_id, tuple)
        extra_column = "chat_id," if group else ""
        extra_value = "?," if group else ""
        conflict = "chat_id, user_id" if group else "user_id"
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                connection.execute(
                    f"""
                    INSERT INTO {table} (
                        {extra_column}
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
                        roleplay_character,
                        roleplay_fetishes,
                        roleplay_preferences,
                        roleplay_boundaries,
                        delta_appearance,
                        content_mode
                    )
                    VALUES ({extra_value} ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT({conflict}) DO UPDATE SET
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
                        roleplay_character = excluded.roleplay_character,
                        roleplay_fetishes = excluded.roleplay_fetishes,
                        roleplay_preferences = excluded.roleplay_preferences,
                        roleplay_boundaries = excluded.roleplay_boundaries,
                        delta_appearance = excluded.delta_appearance,
                        content_mode = excluded.content_mode
                    """,
                    (
                        *key,
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
                        json.dumps(roleplay_fetishes, ensure_ascii=False),
                        roleplay_preferences,
                        roleplay_boundaries,
                        delta_appearance,
                        content_mode,
                    ),
                )
        except sqlite3.Error as error:
            raise UserStatePersistenceError(
                f"Не удалось сохранить состояние пользователя {user_id}"
            ) from error

    def _delete_sync(
        self,
        user_id: StateKey,
    ) -> None:
        """Синхронно удалить долгоживущее состояние пользователя."""
        table, where, key = self._scope(user_id)
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                connection.execute(
                    f"DELETE FROM {table} WHERE {where}",
                    key,
                )
                if isinstance(user_id, int):
                    connection.execute(
                        "DELETE FROM conversation_states WHERE user_id = ?",
                        (user_id,),
                    )
        except sqlite3.Error as error:
            raise UserStatePersistenceError(
                f"Не удалось удалить состояние пользователя {user_id}"
            ) from error

    def _initialize(self) -> None:
        """Создать таблицу и применить совместимые изменения схемы."""
        with closing(sqlite3.connect(self._path)) as connection, connection:
            schema = """
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
                    roleplay_active INTEGER NOT NULL DEFAULT 0,
                    roleplay_configuration TEXT NOT NULL DEFAULT 'male',
                    roleplay_character TEXT NOT NULL DEFAULT '',
                    roleplay_fetishes TEXT NOT NULL DEFAULT '[]',
                    roleplay_preferences TEXT NOT NULL DEFAULT '',
                    roleplay_boundaries TEXT NOT NULL DEFAULT '',
                    delta_appearance TEXT NOT NULL DEFAULT '',
                    content_mode TEXT NOT NULL DEFAULT 'unselected'
                )
                """
            connection.execute(schema)
            connection.execute(
                schema.replace("user_states", "conversation_states")
                .replace(
                    "user_id INTEGER PRIMARY KEY,",
                    "chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL,",
                )
                .replace(
                    "content_mode TEXT NOT NULL DEFAULT 'unselected'",
                    "content_mode TEXT NOT NULL DEFAULT 'unselected', PRIMARY KEY(chat_id, user_id)",
                )
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS conversation_states_user ON conversation_states(user_id)"
            )

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

            if "roleplay_fetishes" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN roleplay_fetishes TEXT NOT NULL DEFAULT '[]'
                    """)

            if "content_mode" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN content_mode TEXT NOT NULL DEFAULT 'unselected'
                    """)

            if "roleplay_preferences" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN roleplay_preferences TEXT NOT NULL DEFAULT ''
                    """)

            if "roleplay_boundaries" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN roleplay_boundaries TEXT NOT NULL DEFAULT ''
                    """)

            if "delta_appearance" not in columns:
                connection.execute("""
                    ALTER TABLE user_states
                    ADD COLUMN delta_appearance TEXT NOT NULL DEFAULT ''
                    """)

    @staticmethod
    def _decode_fetishes(value: str) -> tuple[str, ...]:
        """Безопасно прочитать мотивы текущей RP-сцены из JSON."""
        try:
            decoded = json.loads(value)
        except TypeError, json.JSONDecodeError:
            return ()

        if not isinstance(decoded, list) or not all(
            isinstance(item, str) for item in decoded
        ):
            return ()

        return tuple(dict.fromkeys(decoded))

    @staticmethod
    def _decode_content_mode(value: str) -> ContentMode:
        """Безопасно прочитать выбранный режим содержимого."""
        if value in {"unselected", "soft", "adult"}:
            return cast(ContentMode, value)
        return "unselected"
