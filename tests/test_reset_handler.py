"""Тесты безопасного полного сброса памяти пользователя."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Router
from aiogram.types import CallbackQuery, Message

import protogen_delta.handlers.reset as reset_module
from protogen_delta.handlers.reset import (
    RESET_CANCELLED_TEXT,
    RESET_CONFIRMATION_TEXT,
    RESET_EXPIRED_TEXT,
    RESET_FOREIGN_CALLBACK_TEXT,
    RESET_SUCCESS_TEXT,
    create_reset_router,
)
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.response_engine import ResponseEngine

TEST_USER_ID = 123456
TEST_TIMESTAMP = 1000


def _create_message_mock(
    user_id: int | None = TEST_USER_ID,
) -> tuple[Message, AsyncMock]:
    """Создать Message с асинхронным методом answer."""
    message_mock = Mock(spec=Message)

    if user_id is None:
        message_mock.from_user = None
    else:
        user_mock = Mock()
        user_mock.id = user_id
        message_mock.from_user = user_mock

    answer_mock = AsyncMock()
    message_mock.answer = answer_mock

    return (
        cast(Message, message_mock),
        answer_mock,
    )


def _create_callback_mock(
    data: str,
    *,
    user_id: int = TEST_USER_ID,
) -> tuple[CallbackQuery, AsyncMock, AsyncMock]:
    """Создать CallbackQuery с сообщением и методом answer."""
    callback_mock = Mock(spec=CallbackQuery)
    callback_mock.data = data

    user_mock = Mock()
    user_mock.id = user_id
    callback_mock.from_user = user_mock

    message_mock = Mock(spec=Message)
    edit_text_mock = AsyncMock()
    message_mock.edit_text = edit_text_mock

    callback_mock.message = message_mock

    answer_mock = AsyncMock()
    callback_mock.answer = answer_mock

    return (
        cast(CallbackQuery, callback_mock),
        edit_text_mock,
        answer_mock,
    )


async def _call_message_handler(
    router: Router,
    message: Message,
) -> None:
    """Вызвать обработчик команды /reset."""
    handler = router.message.handlers[0]
    await handler.callback(message)


async def _call_callback_handler(
    router: Router,
    callback: CallbackQuery,
) -> None:
    """Вызвать обработчик callback подтверждения."""
    handler = router.callback_query.handlers[0]
    await handler.callback(callback)


def _create_router() -> tuple[
    Router,
    AsyncMock,
    Mock,
]:
    """Создать reset-router с тестовыми зависимостями."""
    engine_mock = AsyncMock(spec=ResponseEngine)
    users_repository_mock = Mock(spec=UsersRepository)

    router = create_reset_router(
        cast(ResponseEngine, engine_mock),
        cast(UsersRepository, users_repository_mock),
    )

    return (
        router,
        engine_mock,
        users_repository_mock,
    )


def test_reset_command_only_requests_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сам /reset не должен ничего удалять без подтверждения."""
    monkeypatch.setattr(
        reset_module,
        "time",
        lambda: float(TEST_TIMESTAMP),
    )

    router, engine_mock, users_repository_mock = _create_router()

    message, answer_mock = _create_message_mock()

    asyncio.run(
        _call_message_handler(
            router,
            message,
        )
    )

    engine_mock.reset_user.assert_not_awaited()
    users_repository_mock.remove.assert_not_called()

    answer_mock.assert_awaited_once()

    call = answer_mock.await_args

    assert call is not None
    assert call.args == (RESET_CONFIRMATION_TEXT,)

    keyboard = call.kwargs["reply_markup"]

    confirm_button = keyboard.inline_keyboard[0][0]
    cancel_button = keyboard.inline_keyboard[0][1]

    assert confirm_button.text == "✅ Да, забыть всё"
    assert confirm_button.style == "success"
    assert confirm_button.callback_data == (
        f"reset:confirm:{TEST_USER_ID}:{TEST_TIMESTAMP}"
    )

    assert cancel_button.text == "❌ Нет"
    assert cancel_button.style == "danger"
    assert cancel_button.callback_data == (
        f"reset:cancel:{TEST_USER_ID}:{TEST_TIMESTAMP}"
    )


def test_reset_command_ignores_message_without_user() -> None:
    """Команда без Telegram-пользователя не должна создавать подтверждение."""
    router, engine_mock, users_repository_mock = _create_router()

    message, answer_mock = _create_message_mock(
        user_id=None,
    )

    asyncio.run(
        _call_message_handler(
            router,
            message,
        )
    )

    engine_mock.reset_user.assert_not_awaited()
    users_repository_mock.remove.assert_not_called()
    answer_mock.assert_not_awaited()


def test_reset_confirmation_performs_full_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Подтверждение должно полностью забыть пользователя."""
    monkeypatch.setattr(
        reset_module,
        "time",
        lambda: 1100.0,
    )

    router, engine_mock, users_repository_mock = _create_router()

    callback, edit_text_mock, answer_mock = _create_callback_mock(
        f"reset:confirm:{TEST_USER_ID}:{TEST_TIMESTAMP}",
    )

    asyncio.run(
        _call_callback_handler(
            router,
            callback,
        )
    )

    engine_mock.reset_user.assert_awaited_once_with(
        TEST_USER_ID,
    )

    users_repository_mock.remove.assert_called_once_with(
        TEST_USER_ID,
    )

    edit_text_mock.assert_awaited_once_with(
        RESET_SUCCESS_TEXT,
        reply_markup=None,
    )

    answer_mock.assert_awaited_once_with()


def test_reset_cancellation_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отмена не должна изменять пользовательское состояние."""
    monkeypatch.setattr(
        reset_module,
        "time",
        lambda: 1100.0,
    )

    router, engine_mock, users_repository_mock = _create_router()

    callback, edit_text_mock, answer_mock = _create_callback_mock(
        f"reset:cancel:{TEST_USER_ID}:{TEST_TIMESTAMP}",
    )

    asyncio.run(
        _call_callback_handler(
            router,
            callback,
        )
    )

    engine_mock.reset_user.assert_not_awaited()
    users_repository_mock.remove.assert_not_called()

    edit_text_mock.assert_awaited_once_with(
        RESET_CANCELLED_TEXT,
        reply_markup=None,
    )

    answer_mock.assert_awaited_once_with()


def test_reset_confirmation_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Старое подтверждение не должно выполнять сброс."""
    monkeypatch.setattr(
        reset_module,
        "time",
        lambda: 1301.0,
    )

    router, engine_mock, users_repository_mock = _create_router()

    callback, edit_text_mock, answer_mock = _create_callback_mock(
        f"reset:confirm:{TEST_USER_ID}:{TEST_TIMESTAMP}",
    )

    asyncio.run(
        _call_callback_handler(
            router,
            callback,
        )
    )

    engine_mock.reset_user.assert_not_awaited()
    users_repository_mock.remove.assert_not_called()

    edit_text_mock.assert_awaited_once_with(
        RESET_EXPIRED_TEXT,
        reply_markup=None,
    )

    answer_mock.assert_awaited_once_with()


def test_reset_confirmation_rejects_other_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Чужой пользователь не должен подтверждать сброс."""
    monkeypatch.setattr(
        reset_module,
        "time",
        lambda: 1100.0,
    )

    router, engine_mock, users_repository_mock = _create_router()

    callback, edit_text_mock, answer_mock = _create_callback_mock(
        f"reset:confirm:{TEST_USER_ID}:{TEST_TIMESTAMP}",
        user_id=999999,
    )

    asyncio.run(
        _call_callback_handler(
            router,
            callback,
        )
    )

    engine_mock.reset_user.assert_not_awaited()
    users_repository_mock.remove.assert_not_called()
    edit_text_mock.assert_not_awaited()

    answer_mock.assert_awaited_once_with(
        RESET_FOREIGN_CALLBACK_TEXT,
        show_alert=True,
    )


def test_reset_callback_ignores_malformed_data() -> None:
    """Некорректный callback не должен выполнять сброс."""
    router, engine_mock, users_repository_mock = _create_router()

    callback, edit_text_mock, answer_mock = _create_callback_mock(
        "reset:broken",
    )

    asyncio.run(
        _call_callback_handler(
            router,
            callback,
        )
    )

    engine_mock.reset_user.assert_not_awaited()
    users_repository_mock.remove.assert_not_called()
    edit_text_mock.assert_not_awaited()
    answer_mock.assert_awaited_once_with()
