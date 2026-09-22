"""Тесты обработчика управления RP-режимом."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, Mock

from aiogram import Router
from aiogram.types import Message

from protogen_delta.handlers.rp import (
    RP_ALREADY_DISABLED_REPLY,
    RP_DISABLED_REPLY,
    RP_USAGE_REPLY,
    create_rp_router,
    is_roleplay_stop_message,
)
from protogen_delta.services.response_engine import ResponseEngine

TEST_USER_ID = 123456


def _create_message_mock(
    text: str | None,
    user_id: int | None = TEST_USER_ID,
) -> tuple[Message, AsyncMock]:
    """Создать mock Telegram-сообщения."""
    message_mock = Mock(spec=Message)
    message_mock.text = text

    if user_id is None:
        message_mock.from_user = None
    else:
        user_mock = Mock()
        user_mock.id = user_id
        message_mock.from_user = user_mock

    answer_mock = AsyncMock()
    message_mock.answer = answer_mock

    return cast(Message, message_mock), answer_mock


async def _call_first_handler(
    router: Router,
    message: Message,
) -> None:
    """Вызвать первый message-handler роутера."""
    handler = router.message.handlers[0]

    await handler.callback(message)


def test_rp_off_disables_active_roleplay() -> None:
    """Команда /rp off должна завершать активный RP."""
    engine_mock = AsyncMock(spec=ResponseEngine)
    engine_mock.disable_roleplay.return_value = True

    router = create_rp_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock = _create_message_mock("/rp off")

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.disable_roleplay.assert_awaited_once_with(
        TEST_USER_ID,
    )
    answer_mock.assert_awaited_once_with(
        RP_DISABLED_REPLY,
    )


def test_rp_off_reports_already_disabled_roleplay() -> None:
    """Команда должна сообщать, если RP уже выключен."""
    engine_mock = AsyncMock(spec=ResponseEngine)
    engine_mock.disable_roleplay.return_value = False

    router = create_rp_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock = _create_message_mock("/rp off")

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.disable_roleplay.assert_awaited_once_with(
        TEST_USER_ID,
    )
    answer_mock.assert_awaited_once_with(
        RP_ALREADY_DISABLED_REPLY,
    )


def test_rp_without_off_shows_usage() -> None:
    """Неизвестный аргумент /rp не должен менять состояние."""
    engine_mock = AsyncMock(spec=ResponseEngine)

    router = create_rp_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock = _create_message_mock("/rp")

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.disable_roleplay.assert_not_awaited()
    answer_mock.assert_awaited_once_with(
        RP_USAGE_REPLY,
    )


def test_rp_with_unknown_argument_shows_usage() -> None:
    """Неизвестный аргумент команды должен показывать подсказку."""
    engine_mock = AsyncMock(spec=ResponseEngine)

    router = create_rp_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock = _create_message_mock("/rp something")

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.disable_roleplay.assert_not_awaited()
    answer_mock.assert_awaited_once_with(
        RP_USAGE_REPLY,
    )


def test_rp_handler_ignores_message_without_user() -> None:
    """Команда без Telegram-пользователя не должна обрабатываться."""
    engine_mock = AsyncMock(spec=ResponseEngine)

    router = create_rp_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock = _create_message_mock(
        "/rp off",
        user_id=None,
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.disable_roleplay.assert_not_awaited()
    answer_mock.assert_not_awaited()


def test_roleplay_stop_message_recognizes_obvious_phrases() -> None:
    """Очевидные естественные фразы должны распознаваться как выход из RP."""
    assert is_roleplay_stop_message("стоп rp") is True
    assert is_roleplay_stop_message("Стоп РП") is True
    assert is_roleplay_stop_message("  хватит   rp  ") is True
    assert is_roleplay_stop_message("выйди из рп") is True
    assert is_roleplay_stop_message("закончим RP") is True


def test_roleplay_stop_message_ignores_unrelated_text() -> None:
    """Обычный текст не должен случайно завершать RP."""
    assert is_roleplay_stop_message("а что такое rp?") is False
    assert is_roleplay_stop_message("не останавливай rp") is False
    assert is_roleplay_stop_message("продолжаем") is False
