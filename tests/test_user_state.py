"""Тесты пользовательского runtime-состояния."""

from protogen_delta.core.user_state import (
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
    """Один пользователь должен получать тот же объект состояния."""
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

    assert second.mood == "playful"
    assert second.reply_count == 0

    assert store.tracked_users_count == 2


def test_user_state_store_removes_user() -> None:
    """Удаление должно очищать состояние конкретного пользователя."""
    store = UserStateStore()

    store.get(123)

    assert store.remove(123) is True
    assert store.remove(123) is False
    assert store.tracked_users_count == 0
