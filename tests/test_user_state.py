"""Тесты пользовательского runtime-состояния."""

import asyncio

import pytest

from protogen_delta.core.user_state import (
    ConversationTurn,
    UserState,
    UserStateStore,
)


def test_user_state_registers_reply() -> None:
    """Счётчик ответов пользователя должен увеличиваться."""
    state = UserState()

    state.register_reply()
    state.register_reply()

    assert state.reply_count == 2


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
