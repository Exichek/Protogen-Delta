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
from protogen_delta.handlers.delivery import show_typing


def _message() -> Message:
    """Создать сообщение с идентификатором чата."""
    message = Mock(spec=Message)
    message.chat = Mock(id=321)
    return cast(Message, message)


def test_show_typing_without_bot_is_noop() -> None:
    """Тестовые вызовы без Bot должны работать без фоновой задачи."""

    async def scenario() -> None:
        async with show_typing(_message(), None):
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
