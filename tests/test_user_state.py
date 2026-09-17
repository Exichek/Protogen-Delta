"""Тесты пользовательского runtime-состояния."""

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


def test_user_state_store_rejects_invalid_history_limit() -> None:
    """Лимит истории должен быть положительным."""
    try:
        UserStateStore(history_limit=0)
    except ValueError as error:
        assert str(error) == "history_limit должен быть больше нуля"
    else:
        raise AssertionError("Ожидался ValueError")


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
