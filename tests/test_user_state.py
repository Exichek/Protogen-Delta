"""Тесты пользовательского runtime-состояния."""

import asyncio
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from protogen_delta.core.user_state import (
    ConversationTurn,
    EmotionalState,
    PersistentUserState,
    RelationshipState,
    UserState,
    UserStatePersistenceError,
    UserStateStore,
)


def test_user_state_registers_reply() -> None:
    """Счётчик ответов пользователя должен увеличиваться."""
    state = UserState()

    state.register_reply()
    state.register_reply()

    assert state.reply_count == 2


def test_emotional_state_adjusts_values() -> None:
    """Эмоциональное состояние должно накапливать изменения."""
    state = EmotionalState()

    state.adjust(
        warmth=0.3,
        irritation=0.2,
        playfulness=0.4,
        arousal=0.1,
    )

    assert state.warmth == pytest.approx(0.3)
    assert state.irritation == pytest.approx(0.2)
    assert state.playfulness == pytest.approx(0.4)
    assert state.arousal == pytest.approx(0.1)


def test_emotional_state_clamps_values() -> None:
    """Эмоциональные показатели не должны выходить за диапазон."""
    state = EmotionalState(
        warmth=0.9,
        irritation=0.1,
    )

    state.adjust(
        warmth=0.5,
        irritation=-0.5,
    )

    assert state.warmth == 1.0
    assert state.irritation == 0.0


def test_relationship_state_adjusts_values() -> None:
    """Отношение к пользователю должно накапливать изменения."""
    state = RelationshipState()

    state.adjust(
        familiarity=0.2,
        trust=0.3,
        affection=0.4,
        resentment=0.1,
    )

    assert state.familiarity == pytest.approx(0.2)
    assert state.trust == pytest.approx(0.3)
    assert state.affection == pytest.approx(0.4)
    assert state.resentment == pytest.approx(0.1)


def test_relationship_state_clamps_values() -> None:
    """Показатели отношений не должны выходить за диапазон."""
    state = RelationshipState(
        trust=0.9,
        resentment=0.1,
    )

    state.adjust(
        trust=0.5,
        resentment=-0.5,
    )

    assert state.trust == 1.0
    assert state.resentment == 0.0


def test_user_states_have_isolated_emotions_and_relationships() -> None:
    """Разные пользователи не должны делить эмоциональное состояние."""
    first = UserState()
    second = UserState()

    first.emotions.adjust(
        warmth=0.5,
    )
    first.relationship.adjust(
        affection=0.4,
    )

    assert first.emotions.warmth == pytest.approx(0.5)
    assert first.relationship.affection == pytest.approx(0.4)

    assert second.emotions.warmth == 0.0
    assert second.relationship.affection == 0.0


def test_user_state_reset_preserves_persistent_state() -> None:
    """Сброс диалога не должен стирать накопленные эмоции и отношения."""
    state = UserState(
        mood="angry",
        reply_count=5,
    )

    state.emotions.adjust(
        irritation=0.6,
    )
    state.relationship.adjust(
        familiarity=0.7,
        resentment=0.4,
    )

    state.history.append(
        ConversationTurn(
            user_message="Сообщение",
            assistant_message="Ответ",
        )
    )

    state.reset_context()

    assert state.mood == "neutral"
    assert state.reply_count == 0
    assert list(state.history) == []

    assert state.emotions.irritation == pytest.approx(0.6)
    assert state.relationship.familiarity == pytest.approx(0.7)
    assert state.relationship.resentment == pytest.approx(0.4)


def test_user_state_store_returns_same_state() -> None:
    """Один активный пользователь должен получать тот же объект состояния."""
    store = UserStateStore()

    first = store.get(123)
    second = store.get(123)

    assert first is second
    assert store.tracked_users_count == 1


def test_user_state_store_separates_users() -> None:
    """Состояния разных пользователей не должны пересекаться."""
    store = UserStateStore()

    first = store.get(111)
    second = store.get(222)

    first.mood = "sweet"
    first.register_reply()

    assert first.mood == "sweet"
    assert first.reply_count == 1

    assert second.mood == "neutral"
    assert second.reply_count == 0

    assert store.tracked_users_count == 2


def test_user_state_store_removes_user() -> None:
    """Удаление должно очищать состояние конкретного пользователя."""
    store = UserStateStore()

    store.get(123)

    assert store.remove(123) is True
    assert store.remove(123) is False
    assert store.tracked_users_count == 0


def test_user_state_store_does_not_remove_locked_user() -> None:
    """Используемое состояние нельзя удалять из хранилища."""

    async def run_test() -> None:
        store = UserStateStore()
        state = store.get(123)

        await state.lock.acquire()

        try:
            assert store.remove(123) is False
            assert store.tracked_users_count == 1
        finally:
            state.lock.release()

        assert store.remove(123) is True
        assert store.tracked_users_count == 0

    asyncio.run(run_test())


def test_user_state_store_rejects_invalid_history_limit() -> None:
    """Лимит истории должен быть положительным."""
    with pytest.raises(
        ValueError,
        match="history_limit должен быть больше нуля",
    ):
        UserStateStore(history_limit=0)


@pytest.mark.parametrize(
    "retention_seconds",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_user_state_store_rejects_invalid_retention(
    retention_seconds: float,
) -> None:
    """Время хранения состояния должно быть конечным и положительным."""
    with pytest.raises(ValueError):
        UserStateStore(
            retention_seconds=retention_seconds,
        )


def test_user_state_stores_conversation_history() -> None:
    """Пользователь должен хранить завершённые ходы диалога."""
    store = UserStateStore(history_limit=3)
    state = store.get(123)

    state.history.append(
        ConversationTurn(
            user_message="Привет",
            assistant_message="Привет",
        )
    )

    assert list(state.history) == [
        ConversationTurn(
            user_message="Привет",
            assistant_message="Привет",
        )
    ]


def test_user_state_history_is_bounded() -> None:
    """История должна удалять самые старые ходы сверх лимита."""
    store = UserStateStore(history_limit=2)
    state = store.get(123)

    state.history.append(
        ConversationTurn(
            user_message="Первое",
            assistant_message="Ответ 1",
        )
    )
    state.history.append(
        ConversationTurn(
            user_message="Второе",
            assistant_message="Ответ 2",
        )
    )
    state.history.append(
        ConversationTurn(
            user_message="Третье",
            assistant_message="Ответ 3",
        )
    )

    assert list(state.history) == [
        ConversationTurn(
            user_message="Второе",
            assistant_message="Ответ 2",
        ),
        ConversationTurn(
            user_message="Третье",
            assistant_message="Ответ 3",
        ),
    ]


def test_user_state_history_is_isolated_between_users() -> None:
    """История разных пользователей не должна пересекаться."""
    store = UserStateStore(history_limit=2)

    first = store.get(111)
    second = store.get(222)

    first.history.append(
        ConversationTurn(
            user_message="Сообщение первого",
            assistant_message="Ответ первому",
        )
    )

    assert len(first.history) == 1
    assert len(second.history) == 0


def test_user_states_have_separate_locks() -> None:
    """Разные пользователи должны иметь независимые блокировки."""
    store = UserStateStore()

    first = store.get(111)
    second = store.get(222)

    assert first.lock is not second.lock


def test_user_state_resets_context_without_replacing_lock() -> None:
    """Сброс должен очищать контекст, сохраняя блокировку пользователя."""
    state = UserState(
        mood="sweet",
        reply_count=3,
    )

    state.history.append(
        ConversationTurn(
            user_message="Сообщение",
            assistant_message="Ответ",
        )
    )

    original_lock = state.lock

    state.reset_context()

    assert state.mood == "neutral"
    assert state.reply_count == 0
    assert list(state.history) == []
    assert state.lock is original_lock


def test_user_state_store_removes_expired_state() -> None:
    """Неактивное состояние должно удаляться после retention."""
    times = iter(
        [
            0.0,
            11.0,
        ]
    )

    store = UserStateStore(
        retention_seconds=10.0,
        clock=lambda: next(times),
    )

    first = store.get(111)

    first.mood = "sweet"
    first.register_reply()
    first.history.append(
        ConversationTurn(
            user_message="Старое сообщение",
            assistant_message="Старый ответ",
        )
    )

    store.get(222)

    assert store.tracked_users_count == 1
    assert store.remove(111) is False


def test_user_state_store_recreates_expired_user() -> None:
    """Вернувшийся после retention пользователь должен получить чистое состояние."""
    times = iter(
        [
            0.0,
            11.0,
        ]
    )

    store = UserStateStore(
        retention_seconds=10.0,
        clock=lambda: next(times),
    )

    old_state = store.get(123)

    old_state.mood = "sweet"
    old_state.register_reply()
    old_state.history.append(
        ConversationTurn(
            user_message="Старое сообщение",
            assistant_message="Старый ответ",
        )
    )

    new_state = store.get(123)

    assert new_state is not old_state
    assert new_state.mood == "neutral"
    assert new_state.reply_count == 0
    assert list(new_state.history) == []


def test_user_state_store_refreshes_active_user() -> None:
    """Повторное обращение должно продлевать жизнь состояния."""
    times = iter(
        [
            0.0,
            5.0,
            12.0,
        ]
    )

    store = UserStateStore(
        retention_seconds=10.0,
        clock=lambda: next(times),
    )

    first = store.get(111)
    refreshed = store.get(111)

    store.get(222)

    assert refreshed is first
    assert store.tracked_users_count == 2


def test_user_state_store_keeps_locked_expired_state() -> None:
    """Просроченное, но используемое состояние нельзя удалять."""

    async def run_test() -> None:
        times = iter(
            [
                0.0,
                11.0,
            ]
        )

        store = UserStateStore(
            retention_seconds=10.0,
            clock=lambda: next(times),
        )

        state = store.get(111)

        await state.lock.acquire()

        try:
            store.get(222)

            assert store.tracked_users_count == 2
            assert state.lock.locked()
        finally:
            state.lock.release()

    asyncio.run(run_test())


def test_user_state_store_keeps_state_with_waiting_operation() -> None:
    """Состояние нельзя удалить, пока операция ожидает пользовательский lock."""

    async def run_test() -> None:
        now = 0.0

        def clock() -> float:
            return now

        store = UserStateStore(
            retention_seconds=10.0,
            clock=clock,
        )

        state = store.get(111)

        await state.lock.acquire()

        operation_started = asyncio.Event()

        async def waiting_operation() -> None:
            async with store.use(111):
                operation_started.set()

        task = asyncio.create_task(
            waiting_operation(),
        )

        await asyncio.sleep(0)

        assert state.active_operations == 1

        now = 20.0

        state.lock.release()

        store.get(222)

        await task

        assert store.get(111) is state

    asyncio.run(run_test())


def test_user_state_store_loads_persistent_state() -> None:
    """Первое использование должно восстановить сохранённые отношения."""
    persistence = Mock()

    persistence.load = AsyncMock(
        return_value=PersistentUserState(
            emotions=EmotionalState(
                warmth=0.4,
                irritation=0.2,
            ),
            relationship=RelationshipState(
                familiarity=0.7,
                affection=0.5,
                resentment=0.3,
            ),
        )
    )
    persistence.save = AsyncMock()

    store = UserStateStore(
        persistence=persistence,
    )

    async def load_state() -> UserState:
        async with store.use(123) as state:
            return state

    state = asyncio.run(load_state())

    persistence.load.assert_awaited_once_with(123)

    assert state.emotions.warmth == pytest.approx(0.4)
    assert state.emotions.irritation == pytest.approx(0.2)
    assert state.relationship.familiarity == pytest.approx(0.7)
    assert state.relationship.affection == pytest.approx(0.5)
    assert state.relationship.resentment == pytest.approx(0.3)


def test_user_state_store_loads_persistent_state_only_once() -> None:
    """Активное состояние не должно перечитываться из базы при каждом use."""
    persistence = Mock()
    persistence.load = AsyncMock(return_value=None)
    persistence.save = AsyncMock()

    store = UserStateStore(
        persistence=persistence,
    )

    async def use_state_twice() -> tuple[UserState, UserState]:
        async with store.use(123) as first:
            first_state = first

        async with store.use(123) as second:
            second_state = second

        return first_state, second_state

    first, second = asyncio.run(use_state_twice())

    assert first is second
    persistence.load.assert_awaited_once_with(123)


def test_user_state_store_saves_persistent_state_after_use() -> None:
    """После операции долгоживущее состояние должно сохраняться."""
    persistence = Mock()
    persistence.load = AsyncMock(return_value=None)
    persistence.save = AsyncMock()

    store = UserStateStore(
        persistence=persistence,
    )

    async def update_state() -> UserState:
        async with store.use(123) as state:
            state.emotions.adjust(
                warmth=0.4,
            )
            state.relationship.adjust(
                affection=0.3,
            )

            return state

    state = asyncio.run(update_state())

    persistence.save.assert_awaited_once_with(
        123,
        emotions=state.emotions,
        relationship=state.relationship,
    )


def test_user_state_store_reloads_state_after_runtime_eviction() -> None:
    """После удаления из памяти состояние должно снова загрузиться из persistence."""
    persistence = Mock()

    persistence.load = AsyncMock(
        side_effect=[
            None,
            PersistentUserState(
                emotions=EmotionalState(
                    warmth=0.6,
                ),
                relationship=RelationshipState(
                    familiarity=0.8,
                ),
            ),
        ]
    )
    persistence.save = AsyncMock()

    store = UserStateStore(
        persistence=persistence,
    )

    async def first_use() -> None:
        async with store.use(123):
            pass

    asyncio.run(first_use())

    assert store.remove(123) is True

    async def second_use() -> UserState:
        async with store.use(123) as state:
            return state

    state = asyncio.run(second_use())

    assert persistence.load.await_count == 2
    assert state.emotions.warmth == pytest.approx(0.6)
    assert state.relationship.familiarity == pytest.approx(0.8)


def test_user_state_store_does_not_fail_when_persistence_save_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ошибка сохранения не должна ломать завершённую пользовательскую операцию."""
    persistence = Mock()
    persistence.load = AsyncMock(return_value=None)
    persistence.save = AsyncMock(
        side_effect=UserStatePersistenceError(
            "SQLite недоступен",
        )
    )

    store = UserStateStore(
        persistence=persistence,
    )

    async def update_state() -> None:
        async with store.use(123) as state:
            state.emotions.adjust(
                warmth=0.4,
            )
            state.relationship.adjust(
                affection=0.3,
            )

    with caplog.at_level(
        logging.ERROR,
        logger="protogen_delta.core.user_state",
    ):
        asyncio.run(update_state())

    state = store.get(123)

    assert state.emotions.warmth == pytest.approx(0.4)
    assert state.relationship.affection == pytest.approx(0.3)
    assert state.active_operations == 0
    assert state.lock.locked() is False
    assert "Не удалось сохранить состояние пользователя 123" in caplog.text


def test_user_state_store_does_not_save_defaults_after_load_failure() -> None:
    """Сбой загрузки не должен приводить к сохранению пустого состояния."""
    persistence = Mock()

    persistence.load = AsyncMock(
        side_effect=[
            UserStatePersistenceError(
                "SQLite недоступен",
            ),
            PersistentUserState(
                emotions=EmotionalState(
                    warmth=0.6,
                ),
                relationship=RelationshipState(
                    trust=0.7,
                ),
            ),
        ]
    )
    persistence.save = AsyncMock()

    store = UserStateStore(
        persistence=persistence,
    )

    async def use_state() -> UserState:
        async with store.use(123) as state:
            return state

    with pytest.raises(
        UserStatePersistenceError,
        match="SQLite недоступен",
    ):
        asyncio.run(use_state())

    persistence.save.assert_not_awaited()

    state = asyncio.run(use_state())

    assert persistence.load.await_count == 2
    assert state.emotions.warmth == pytest.approx(0.6)
    assert state.relationship.trust == pytest.approx(0.7)

    persistence.save.assert_awaited_once()


def test_user_state_store_serializes_initial_persistence_load() -> None:
    """Два первых запроса одного пользователя должны выполнить только один load."""
    persistence = Mock()
    persistence.load = AsyncMock()
    persistence.save = AsyncMock()

    store = UserStateStore(
        persistence=persistence,
    )

    async def run_test() -> list[UserState]:
        load_started = asyncio.Event()
        release_load = asyncio.Event()

        async def load_state(
            user_id: int,
        ) -> PersistentUserState:
            assert user_id == 123

            load_started.set()
            await release_load.wait()

            return PersistentUserState(
                emotions=EmotionalState(
                    warmth=0.5,
                ),
                relationship=RelationshipState(
                    familiarity=0.6,
                ),
            )

        persistence.load.side_effect = load_state

        states: list[UserState] = []

        async def use_state() -> None:
            async with store.use(123) as state:
                states.append(state)

        first_task = asyncio.create_task(
            use_state(),
        )

        await load_started.wait()

        second_task = asyncio.create_task(
            use_state(),
        )

        await asyncio.sleep(0)

        assert persistence.load.await_count == 1

        release_load.set()

        await asyncio.gather(
            first_task,
            second_task,
        )

        return states

    states = asyncio.run(run_test())

    persistence.load.assert_awaited_once_with(123)

    assert len(states) == 2
    assert states[0] is states[1]
    assert states[0].emotions.warmth == pytest.approx(0.5)
    assert states[0].relationship.familiarity == pytest.approx(0.6)
