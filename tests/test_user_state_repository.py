"""Тесты SQLite-хранилища долгоживущего состояния пользователей."""

import asyncio
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from protogen_delta.core.user_state import (
    EmotionalState,
    RelationshipState,
    UserState,
    UserStatePersistenceError,
    UserStateStore,
)
from protogen_delta.repositories.user_state import UserStateRepository

INITIAL_TIMESTAMP = 1000.0
LATER_TIMESTAMP = 19000.0


def test_user_state_repository_creates_database(
    tmp_path: Path,
) -> None:
    """Репозиторий должен создать SQLite-базу при инициализации."""
    UserStateRepository(tmp_path)

    assert (tmp_path / "user_states.db").exists()


def test_user_state_repository_returns_none_for_unknown_user(
    tmp_path: Path,
) -> None:
    """Для неизвестного пользователя сохранённого состояния ещё нет."""
    repository = UserStateRepository(tmp_path)

    state = asyncio.run(
        repository.load(123),
    )

    assert state is None


def test_user_state_repository_saves_and_loads_state(
    tmp_path: Path,
) -> None:
    """Сохранённое состояние пользователя должно полностью восстанавливаться."""
    repository = UserStateRepository(tmp_path)

    async def run_test() -> None:
        await repository.save(
            123,
            emotions=EmotionalState(
                warmth=0.30,
                irritation=0.20,
                playfulness=0.40,
                arousal=0.10,
            ),
            relationship=RelationshipState(
                familiarity=0.50,
                trust=0.60,
                affection=0.70,
                resentment=0.20,
            ),
            emotions_updated_at=INITIAL_TIMESTAMP,
            roleplay_active=True,
        )

        state = await repository.load(123)

        assert state is not None

        assert state.emotions.warmth == pytest.approx(0.30)
        assert state.emotions.irritation == pytest.approx(0.20)
        assert state.emotions.playfulness == pytest.approx(0.40)
        assert state.emotions.arousal == pytest.approx(0.10)

        assert state.relationship.familiarity == pytest.approx(0.50)
        assert state.relationship.trust == pytest.approx(0.60)
        assert state.relationship.affection == pytest.approx(0.70)
        assert state.relationship.resentment == pytest.approx(0.20)

        assert state.emotions_updated_at == pytest.approx(
            INITIAL_TIMESTAMP,
        )
        assert state.roleplay_active is True

    asyncio.run(run_test())


def test_user_state_repository_updates_existing_user(
    tmp_path: Path,
) -> None:
    """Повторное сохранение должно обновлять всё долгоживущее состояние."""
    repository = UserStateRepository(tmp_path)

    async def run_test() -> None:
        await repository.save(
            123,
            emotions=EmotionalState(
                warmth=0.10,
            ),
            relationship=RelationshipState(
                familiarity=0.20,
            ),
            emotions_updated_at=INITIAL_TIMESTAMP,
            roleplay_active=False,
        )

        await repository.save(
            123,
            emotions=EmotionalState(
                warmth=0.80,
                irritation=0.30,
            ),
            relationship=RelationshipState(
                familiarity=0.90,
                affection=0.60,
            ),
            emotions_updated_at=LATER_TIMESTAMP,
            roleplay_active=True,
        )

        state = await repository.load(123)

        assert state is not None

        assert state.emotions.warmth == pytest.approx(0.80)
        assert state.emotions.irritation == pytest.approx(0.30)

        assert state.relationship.familiarity == pytest.approx(0.90)
        assert state.relationship.affection == pytest.approx(0.60)

        assert state.emotions_updated_at == pytest.approx(
            LATER_TIMESTAMP,
        )
        assert state.roleplay_active is True

    asyncio.run(run_test())


def test_user_state_repository_keeps_users_separate(
    tmp_path: Path,
) -> None:
    """Состояния разных пользователей не должны пересекаться."""
    repository = UserStateRepository(tmp_path)

    async def run_test() -> None:
        await repository.save(
            111,
            emotions=EmotionalState(
                warmth=0.70,
            ),
            relationship=RelationshipState(
                affection=0.60,
            ),
            emotions_updated_at=INITIAL_TIMESTAMP,
            roleplay_active=True,
        )

        await repository.save(
            222,
            emotions=EmotionalState(
                irritation=0.80,
            ),
            relationship=RelationshipState(
                resentment=0.50,
            ),
            emotions_updated_at=LATER_TIMESTAMP,
            roleplay_active=False,
        )

        first = await repository.load(111)
        second = await repository.load(222)

        assert first is not None
        assert second is not None

        assert first.emotions.warmth == pytest.approx(0.70)
        assert first.emotions.irritation == 0.0
        assert first.relationship.affection == pytest.approx(0.60)
        assert first.relationship.resentment == 0.0
        assert first.emotions_updated_at == pytest.approx(
            INITIAL_TIMESTAMP,
        )
        assert first.roleplay_active is True

        assert second.emotions.warmth == 0.0
        assert second.emotions.irritation == pytest.approx(0.80)
        assert second.relationship.affection == 0.0
        assert second.relationship.resentment == pytest.approx(0.50)
        assert second.emotions_updated_at == pytest.approx(
            LATER_TIMESTAMP,
        )
        assert second.roleplay_active is False

    asyncio.run(run_test())


def test_user_state_repository_survives_new_instance(
    tmp_path: Path,
) -> None:
    """Состояние должно переживать создание нового экземпляра репозитория."""
    first_repository = UserStateRepository(tmp_path)

    async def save_state() -> None:
        await first_repository.save(
            123,
            emotions=EmotionalState(
                warmth=0.40,
                playfulness=0.30,
            ),
            relationship=RelationshipState(
                familiarity=0.80,
                trust=0.50,
                affection=0.60,
                resentment=0.10,
            ),
            emotions_updated_at=INITIAL_TIMESTAMP,
            roleplay_active=True,
        )

    asyncio.run(save_state())

    second_repository = UserStateRepository(tmp_path)

    state = asyncio.run(
        second_repository.load(123),
    )

    assert state is not None

    assert state.emotions.warmth == pytest.approx(0.40)
    assert state.emotions.playfulness == pytest.approx(0.30)

    assert state.relationship.familiarity == pytest.approx(0.80)
    assert state.relationship.trust == pytest.approx(0.50)
    assert state.relationship.affection == pytest.approx(0.60)
    assert state.relationship.resentment == pytest.approx(0.10)

    assert state.emotions_updated_at == pytest.approx(
        INITIAL_TIMESTAMP,
    )
    assert state.roleplay_active is True


def test_user_state_store_restores_state_after_restart(
    tmp_path: Path,
) -> None:
    """Рестарт без прошедшего времени должен восстановить persistent-состояние."""
    first_repository = UserStateRepository(tmp_path)
    first_store = UserStateStore(
        wall_clock=lambda: INITIAL_TIMESTAMP,
        persistence=first_repository,
    )

    async def save_state() -> None:
        async with first_store.use(123) as state:
            state.emotions.adjust(
                warmth=0.45,
                irritation=0.20,
            )
            state.relationship.adjust(
                familiarity=0.70,
                trust=0.40,
                affection=0.55,
                resentment=0.15,
            )
            state.roleplay_active = True

    asyncio.run(save_state())

    second_repository = UserStateRepository(tmp_path)
    second_store = UserStateStore(
        wall_clock=lambda: INITIAL_TIMESTAMP,
        persistence=second_repository,
    )

    async def restore_state() -> UserState:
        async with second_store.use(123) as state:
            return state

    restored = asyncio.run(restore_state())

    assert restored.emotions.warmth == pytest.approx(0.45)
    assert restored.emotions.irritation == pytest.approx(0.20)

    assert restored.relationship.familiarity == pytest.approx(0.70)
    assert restored.relationship.trust == pytest.approx(0.40)
    assert restored.relationship.affection == pytest.approx(0.55)
    assert restored.relationship.resentment == pytest.approx(0.15)

    assert restored.emotions_updated_at == pytest.approx(
        INITIAL_TIMESTAMP,
    )
    assert restored.roleplay_active is True

    assert restored.mood == "neutral"
    assert restored.reply_count == 0
    assert list(restored.history) == []


def test_user_state_store_decays_emotions_after_restart(
    tmp_path: Path,
) -> None:
    """После рестарта эмоции должны затухнуть согласно прошедшему времени."""
    first_repository = UserStateRepository(tmp_path)
    first_store = UserStateStore(
        wall_clock=lambda: INITIAL_TIMESTAMP,
        persistence=first_repository,
    )

    async def save_state() -> None:
        async with first_store.use(123) as state:
            state.emotions.adjust(
                warmth=0.20,
                irritation=0.20,
                playfulness=0.20,
                arousal=0.20,
            )
            state.relationship.adjust(
                familiarity=0.80,
                trust=0.70,
                affection=0.60,
                resentment=0.40,
            )
            state.roleplay_active = True

    asyncio.run(save_state())

    second_repository = UserStateRepository(tmp_path)
    second_store = UserStateStore(
        wall_clock=lambda: LATER_TIMESTAMP,
        persistence=second_repository,
    )

    async def restore_state() -> UserState:
        async with second_store.use(123) as state:
            return state

    restored = asyncio.run(restore_state())

    assert restored.emotions.warmth == pytest.approx(0.15)
    assert restored.emotions.irritation == pytest.approx(0.10)
    assert restored.emotions.playfulness == pytest.approx(0.10)
    assert restored.emotions.arousal == pytest.approx(0.10)

    assert restored.relationship.familiarity == pytest.approx(0.80)
    assert restored.relationship.trust == pytest.approx(0.70)
    assert restored.relationship.affection == pytest.approx(0.60)
    assert restored.relationship.resentment == pytest.approx(0.40)

    assert restored.emotions_updated_at == pytest.approx(
        LATER_TIMESTAMP,
    )
    assert restored.roleplay_active is True

    persisted = asyncio.run(
        second_repository.load(123),
    )

    assert persisted is not None
    assert persisted.emotions.irritation == pytest.approx(0.10)
    assert persisted.relationship.trust == pytest.approx(0.70)
    assert persisted.emotions_updated_at == pytest.approx(
        LATER_TIMESTAMP,
    )
    assert persisted.roleplay_active is True


def test_user_state_repository_migrates_legacy_database(
    tmp_path: Path,
) -> None:
    """Старая таблица должна получить новые поля без потери состояния."""
    database_path = tmp_path / "user_states.db"

    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("""
            CREATE TABLE user_states (
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
            """,
            (
                123,
                0.40,
                0.30,
                0.20,
                0.10,
                0.80,
                0.70,
                0.60,
                0.50,
            ),
        )

    repository = UserStateRepository(tmp_path)

    state = asyncio.run(
        repository.load(123),
    )

    assert state is not None

    assert state.emotions.warmth == pytest.approx(0.40)
    assert state.emotions.irritation == pytest.approx(0.30)
    assert state.emotions.playfulness == pytest.approx(0.20)
    assert state.emotions.arousal == pytest.approx(0.10)

    assert state.relationship.familiarity == pytest.approx(0.80)
    assert state.relationship.trust == pytest.approx(0.70)
    assert state.relationship.affection == pytest.approx(0.60)
    assert state.relationship.resentment == pytest.approx(0.50)

    assert state.emotions_updated_at == 0.0
    assert state.roleplay_active is False
    assert state.roleplay_configuration == "male"
    assert state.roleplay_character == ""

    with closing(sqlite3.connect(database_path)) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(user_states)")
        }

    assert "emotions_updated_at" in columns
    assert "roleplay_active" in columns


def test_user_state_repository_closes_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Каждая SQLite-операция должна явно закрывать соединение."""
    connections: list[MagicMock] = []

    def connect(_path: Path) -> MagicMock:
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value.fetchone.return_value = None

        connections.append(connection)

        return connection

    monkeypatch.setattr(
        "protogen_delta.repositories.user_state.sqlite3.connect",
        connect,
    )

    repository = UserStateRepository(tmp_path)

    async def run_test() -> None:
        await repository.load(123)

        await repository.save(
            123,
            emotions=EmotionalState(),
            relationship=RelationshipState(),
            emotions_updated_at=INITIAL_TIMESTAMP,
            roleplay_active=False,
        )

    asyncio.run(run_test())

    assert len(connections) == 3

    for connection in connections:
        connection.__enter__.assert_called_once_with()
        connection.__exit__.assert_called_once()
        connection.close.assert_called_once_with()


def test_user_state_repository_wraps_save_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка SQLite при сохранении должна превращаться в persistence-ошибку."""
    repository = UserStateRepository(tmp_path)

    def failing_connect(_path: Path) -> None:
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(
        "protogen_delta.repositories.user_state.sqlite3.connect",
        failing_connect,
    )

    with pytest.raises(
        UserStatePersistenceError,
        match="Не удалось сохранить состояние пользователя 123",
    ):
        asyncio.run(
            repository.save(
                123,
                emotions=EmotionalState(),
                relationship=RelationshipState(),
                emotions_updated_at=INITIAL_TIMESTAMP,
                roleplay_active=False,
            )
        )


def test_user_state_repository_wraps_load_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка SQLite при загрузке должна превращаться в persistence-ошибку."""
    repository = UserStateRepository(tmp_path)

    def failing_connect(_path: Path) -> None:
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(
        "protogen_delta.repositories.user_state.sqlite3.connect",
        failing_connect,
    )

    with pytest.raises(
        UserStatePersistenceError,
        match="Не удалось загрузить состояние пользователя 123",
    ):
        asyncio.run(
            repository.load(123),
        )


def test_user_state_repository_runs_sqlite_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite load/save должны выполняться в worker thread."""
    repository = UserStateRepository(tmp_path)

    event_loop_thread_id = threading.get_ident()
    sqlite_thread_ids: list[int] = []

    original_connect = sqlite3.connect

    def tracked_connect(
        database: Path,
    ) -> sqlite3.Connection:
        sqlite_thread_ids.append(
            threading.get_ident(),
        )

        return original_connect(database)

    monkeypatch.setattr(
        "protogen_delta.repositories.user_state.sqlite3.connect",
        tracked_connect,
    )

    async def run_test() -> None:
        await repository.load(123)

        await repository.save(
            123,
            emotions=EmotionalState(),
            relationship=RelationshipState(),
            emotions_updated_at=INITIAL_TIMESTAMP,
            roleplay_active=False,
        )

    asyncio.run(run_test())

    assert len(sqlite_thread_ids) == 2
    assert all(thread_id != event_loop_thread_id for thread_id in sqlite_thread_ids)


def test_user_state_repository_deletes_only_requested_user(
    tmp_path: Path,
) -> None:
    """Удаление должно стирать только выбранное persistent-состояние."""
    repository = UserStateRepository(tmp_path)

    async def run_test() -> None:
        await repository.save(
            111,
            emotions=EmotionalState(
                warmth=0.70,
            ),
            relationship=RelationshipState(
                trust=0.80,
            ),
            emotions_updated_at=INITIAL_TIMESTAMP,
            roleplay_active=True,
        )

        await repository.save(
            222,
            emotions=EmotionalState(
                irritation=0.60,
            ),
            relationship=RelationshipState(
                resentment=0.50,
            ),
            emotions_updated_at=LATER_TIMESTAMP,
            roleplay_active=True,
        )

        await repository.delete(111)

        first = await repository.load(111)
        second = await repository.load(222)

        assert first is None

        assert second is not None
        assert second.emotions.irritation == pytest.approx(0.60)
        assert second.relationship.resentment == pytest.approx(0.50)
        assert second.roleplay_active is True

    asyncio.run(run_test())


def test_user_state_repository_wraps_delete_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка SQLite при удалении должна превращаться в persistence-ошибку."""
    repository = UserStateRepository(tmp_path)

    def failing_connect(_path: Path) -> None:
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(
        "protogen_delta.repositories.user_state.sqlite3.connect",
        failing_connect,
    )

    with pytest.raises(
        UserStatePersistenceError,
        match="Не удалось удалить состояние пользователя 123",
    ):
        asyncio.run(
            repository.delete(123),
        )


def test_user_state_repository_runs_delete_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite delete должен выполняться в worker thread."""
    repository = UserStateRepository(tmp_path)

    event_loop_thread_id = threading.get_ident()
    sqlite_thread_ids: list[int] = []

    original_connect = sqlite3.connect

    def tracked_connect(
        database: Path,
    ) -> sqlite3.Connection:
        sqlite_thread_ids.append(
            threading.get_ident(),
        )

        return original_connect(database)

    monkeypatch.setattr(
        "protogen_delta.repositories.user_state.sqlite3.connect",
        tracked_connect,
    )

    asyncio.run(
        repository.delete(123),
    )

    assert len(sqlite_thread_ids) == 1
    assert sqlite_thread_ids[0] != event_loop_thread_id


def test_scene_details_survive_restart_and_reset(tmp_path: Path) -> None:
    async def run() -> None:
        first = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with first.use(123) as state:
            state.roleplay_active = True
            state.roleplay_configuration = "female"
            state.roleplay_character = "человек в пальто"
            state.relationship.trust = 0.7
        second = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with second.use(123) as restored:
            assert restored.roleplay_active is True
            assert restored.roleplay_configuration == "female"
            assert restored.roleplay_character == "человек в пальто"
            assert restored.relationship.trust == pytest.approx(0.7)
            assert not restored.history
        await second.reset_user(123)
        third = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with third.use(123) as cleared:
            assert cleared.roleplay_configuration == "male"
            assert cleared.roleplay_character == ""
            assert cleared.relationship.trust == 0.0

    asyncio.run(run())
