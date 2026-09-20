"""Тесты SQLite-хранилища долгоживущего состояния пользователей."""

import asyncio
import sqlite3
import threading
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
    """Сохранённые эмоции и отношения должны восстанавливаться."""
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

    asyncio.run(run_test())


def test_user_state_repository_updates_existing_user(
    tmp_path: Path,
) -> None:
    """Повторное сохранение должно обновлять существующую запись."""
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
        )

        state = await repository.load(123)

        assert state is not None

        assert state.emotions.warmth == pytest.approx(0.80)
        assert state.emotions.irritation == pytest.approx(0.30)

        assert state.relationship.familiarity == pytest.approx(0.90)
        assert state.relationship.affection == pytest.approx(0.60)

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
        )

        await repository.save(
            222,
            emotions=EmotionalState(
                irritation=0.80,
            ),
            relationship=RelationshipState(
                resentment=0.50,
            ),
        )

        first = await repository.load(111)
        second = await repository.load(222)

        assert first is not None
        assert second is not None

        assert first.emotions.warmth == pytest.approx(0.70)
        assert first.emotions.irritation == 0.0
        assert first.relationship.affection == pytest.approx(0.60)
        assert first.relationship.resentment == 0.0

        assert second.emotions.warmth == 0.0
        assert second.emotions.irritation == pytest.approx(0.80)
        assert second.relationship.affection == 0.0
        assert second.relationship.resentment == pytest.approx(0.50)

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


def test_user_state_store_restores_state_after_restart(
    tmp_path: Path,
) -> None:
    """Новый store должен восстановить отношения из существующей SQLite-базы."""
    first_repository = UserStateRepository(tmp_path)
    first_store = UserStateStore(
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

    asyncio.run(save_state())

    second_repository = UserStateRepository(tmp_path)
    second_store = UserStateStore(
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

    assert restored.mood == "neutral"
    assert restored.reply_count == 0
    assert list(restored.history) == []


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
        )

    asyncio.run(run_test())

    assert len(sqlite_thread_ids) == 2
    assert all(thread_id != event_loop_thread_id for thread_id in sqlite_thread_ids)
