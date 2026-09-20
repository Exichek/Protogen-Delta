"""SQLite-хранилище долгоживущего состояния пользователей."""

import sqlite3
from pathlib import Path

from protogen_delta.core.user_state import (
    EmotionalState,
    PersistentUserState,
    RelationshipState,
)


class UserStateRepository:
    """Сохранять долгоживущие эмоции и отношения пользователей в SQLite."""

    def __init__(self, data_dir: Path) -> None:
        """Создать SQLite-базу и подготовить таблицу состояний."""
        self._path = data_dir / "user_states.db"

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    def load(
        self,
        user_id: int,
    ) -> PersistentUserState | None:
        """Загрузить сохранённое состояние пользователя."""
        with sqlite3.connect(self._path) as connection:
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
                    resentment
                FROM user_states
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

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
        )

    def save(
        self,
        user_id: int,
        *,
        emotions: EmotionalState,
        relationship: RelationshipState,
    ) -> None:
        """Создать или обновить долгоживущее состояние пользователя."""
        with sqlite3.connect(self._path) as connection:
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
                    resentment
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    warmth = excluded.warmth,
                    irritation = excluded.irritation,
                    playfulness = excluded.playfulness,
                    arousal = excluded.arousal,
                    familiarity = excluded.familiarity,
                    trust = excluded.trust,
                    affection = excluded.affection,
                    resentment = excluded.resentment
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
                ),
            )

    def _initialize(self) -> None:
        """Создать таблицу состояний при первом запуске."""
        with sqlite3.connect(self._path) as connection:
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
                    resentment REAL NOT NULL
                )
                """)
