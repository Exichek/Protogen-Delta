"""Тесты служебного статуса набора ответа."""

import asyncio
from collections.abc import Coroutine
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

import protogen_delta.handlers.delivery as delivery_module
from protogen_delta.handlers.delivery import create_reply_delivery, show_typing
from protogen_delta.services.stickers import ContextualStickerService


def _message() -> Message:
    """Создать сообщение с идентификатором чата."""
    message = Mock(spec=Message)
    message.chat = Mock(id=321)
    message.text = None
    message.caption = None
    return cast(Message, message)


def test_show_typing_without_bot_is_noop() -> None:
    """Тестовые вызовы без Bot должны работать без фоновой задачи."""

    async def scenario() -> None:
        async with show_typing(_message(), None):
            pass

    asyncio.run(scenario())


def test_show_typing_rejects_negative_initial_delay() -> None:
    """Отрицательная пауза перед статусом набора должна отклоняться."""

    async def scenario() -> None:
        with pytest.raises(ValueError, match="не может быть отрицательной"):
            async with show_typing(
                _message(),
                None,
                initial_delay_seconds=-0.1,
            ):
                pass

    asyncio.run(scenario())


def test_show_typing_sends_and_stops_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Долгая обработка должна показать typing и завершить worker вместе с ней."""
    calls = 0

    async def fast_wait(
        awaitable: Coroutine[Any, Any, bool],
        *,
        timeout: float,
    ) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            awaitable.close()
            raise TimeoutError
        return await awaitable

    monkeypatch.setattr(delivery_module, "wait_for", fast_wait)
    bot = AsyncMock(spec=Bot)

    async def scenario() -> None:
        async with show_typing(_message(), cast(Bot, bot)):
            await asyncio.sleep(0)

    asyncio.run(scenario())

    bot.send_chat_action.assert_awaited_once_with(
        chat_id=321,
        action=ChatAction.TYPING,
    )


def test_show_typing_ignores_telegram_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Недоступный chat action не должен ломать основной ответ."""

    async def immediate_timeout(
        awaitable: Coroutine[Any, Any, bool],
        *,
        timeout: float,
    ) -> bool:
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(delivery_module, "wait_for", immediate_timeout)
    bot = AsyncMock(spec=Bot)
    bot.send_chat_action.side_effect = TelegramAPIError(Mock(), "failed")

    async def scenario() -> None:
        async with show_typing(_message(), cast(Bot, bot)):
            await asyncio.sleep(0)

    asyncio.run(scenario())

    bot.send_chat_action.assert_awaited_once()


def test_reply_delivery_triggers_optional_contextual_sticker() -> None:
    """После успешного текста можно добавить редкую реакцию без участия LLM."""
    message = _message()
    answer = AsyncMock()
    cast(Any, message).answer = answer
    sticker_service = AsyncMock(spec=ContextualStickerService)
    deliver = create_reply_delivery(
        message,
        None,
        cast(ContextualStickerService, sticker_service),
        user_id=42,
        context_tags=("media",),
    )

    asyncio.run(deliver("Готово."))

    answer.assert_awaited_once_with("Готово.")
    sticker_service.maybe_send.assert_awaited_once_with(
        chat_id=321,
        user_id=42,
        context_tags=("media",),
        context_text="",
    )
