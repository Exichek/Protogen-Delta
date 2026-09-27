"""Тесты выбора возрастного режима общения."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, Mock

from aiogram import Router
from aiogram.types import CallbackQuery, Message

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.adult import (
    ADULT_ENABLED_TEXT,
    AGE_PROMPT_TEXT,
    FOREIGN_CALLBACK_TEXT,
    SOFT_ENABLED_TEXT,
    create_adult_router,
)

TEST_USER_ID = 123456


def _create_message(user_id: int | None = TEST_USER_ID) -> tuple[Message, AsyncMock]:
    """Создать сообщение с асинхронным методом ответа."""
    message = Mock(spec=Message)
    message.from_user = None if user_id is None else Mock(id=user_id)
    message.answer = AsyncMock()
    return cast(Message, message), message.answer


def _create_callback(
    data: str,
    *,
    user_id: int = TEST_USER_ID,
) -> tuple[CallbackQuery, AsyncMock, AsyncMock]:
    """Создать callback с редактируемым сообщением."""
    callback = Mock(spec=CallbackQuery)
    callback.data = data
    callback.from_user = Mock(id=user_id)
    callback.answer = AsyncMock()
    callback.message = Mock(spec=Message)
    callback.message.edit_text = AsyncMock()
    return (
        cast(CallbackQuery, callback),
        callback.message.edit_text,
        callback.answer,
    )


async def _call_message_handler(router: Router, message: Message) -> None:
    """Вызвать обработчик команды роутера."""
    await router.message.handlers[0].callback(message)


async def _call_callback_handler(router: Router, callback: CallbackQuery) -> None:
    """Вызвать обработчик кнопок роутера."""
    await router.callback_query.handlers[0].callback(callback)


def test_adult_command_shows_current_mode_and_keyboard() -> None:
    """Команда должна объяснить режим и показать обе кнопки выбора."""
    states = UserStateStore()
    router = create_adult_router(states)
    message, answer = _create_message()

    asyncio.run(_call_message_handler(router, message))

    call = answer.await_args
    assert call is not None
    assert call.args == (f"Текущий режим: ещё не выбран.\n\n{AGE_PROMPT_TEXT}",)
    buttons = call.kwargs["reply_markup"].inline_keyboard[0]
    assert [button.text for button in buttons] == [
        "🔞 Мне есть 18",
        "🍓 Мне нет 18",
    ]
    assert [button.callback_data for button in buttons] == [
        f"adult:adult:{TEST_USER_ID}",
        f"adult:soft:{TEST_USER_ID}",
    ]


def test_adult_command_ignores_message_without_user() -> None:
    """Команда без Telegram-пользователя не должна отправлять настройку."""
    router = create_adult_router(UserStateStore())
    message, answer = _create_message(user_id=None)

    asyncio.run(_call_message_handler(router, message))

    answer.assert_not_awaited()


def test_adult_callback_enables_and_persists_adult_mode() -> None:
    """Подтверждение совершеннолетия должно сохраниться в состоянии."""
    states = UserStateStore()
    router = create_adult_router(states)
    callback, edit_text, answer = _create_callback(f"adult:adult:{TEST_USER_ID}")

    asyncio.run(_call_callback_handler(router, callback))

    assert states.get(TEST_USER_ID).content_mode == "adult"
    edit_text.assert_awaited_once_with(ADULT_ENABLED_TEXT, reply_markup=None)
    answer.assert_awaited_once_with()


def test_adult_callback_enables_soft_mode() -> None:
    """Мягкий выбор должен отключить откровенный взрослый режим."""
    states = UserStateStore()
    states.get(TEST_USER_ID).content_mode = "adult"
    router = create_adult_router(states)
    callback, edit_text, answer = _create_callback(f"adult:soft:{TEST_USER_ID}")

    asyncio.run(_call_callback_handler(router, callback))

    assert states.get(TEST_USER_ID).content_mode == "soft"
    edit_text.assert_awaited_once_with(SOFT_ENABLED_TEXT, reply_markup=None)
    answer.assert_awaited_once_with()


def test_adult_callback_rejects_other_user() -> None:
    """Чужая кнопка не должна менять возрастной режим."""
    states = UserStateStore()
    router = create_adult_router(states)
    callback, edit_text, answer = _create_callback(
        f"adult:adult:{TEST_USER_ID}",
        user_id=999999,
    )

    asyncio.run(_call_callback_handler(router, callback))

    assert states.get(TEST_USER_ID).content_mode == "unselected"
    edit_text.assert_not_awaited()
    answer.assert_awaited_once_with(FOREIGN_CALLBACK_TEXT, show_alert=True)


def test_adult_callback_ignores_malformed_data() -> None:
    """Некорректная callback-data не должна менять состояние."""
    states = UserStateStore()
    router = create_adult_router(states)
    callback, edit_text, answer = _create_callback("adult:broken")

    asyncio.run(_call_callback_handler(router, callback))

    assert states.get(TEST_USER_ID).content_mode == "unselected"
    edit_text.assert_not_awaited()
    answer.assert_awaited_once_with()
