"""Тесты панели управления и Telegram-утилит."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Chat, Message, User

from protogen_delta.handlers.menu import create_menu_router
from protogen_delta.handlers.utilities import create_utilities_router


async def _call_message(router: Router, message: Message) -> None:
    await router.message.handlers[0].callback(message)


def _message(text: str = "/menu") -> tuple[Message, AsyncMock]:
    message = Mock(spec=Message)
    message.text = text
    message.answer = AsyncMock()
    message.reply_to_message = None
    message.from_user = User(id=42, is_bot=False, first_name="Тест", username="tester")
    message.chat = Chat(id=-100777, type="supergroup", title="Лаборатория")
    message.message_thread_id = None
    return cast(Message, message), message.answer


def test_menu_shows_sections_and_optional_mini_app() -> None:
    """Панель должна показывать разделы и настроенную Mini App."""
    router = create_menu_router("https://delta.example/app")
    message, answer = _message()

    asyncio.run(_call_message(router, message))

    call = answer.await_args
    assert call is not None
    assert "ИИ-ассистент Протоген Дельта" in call.args[0]
    buttons = [
        button for row in call.kwargs["reply_markup"].inline_keyboard for button in row
    ]
    assert {button.text for button in buttons} >= {
        "💬 Общение и RP",
        "🎨 Арты",
        "🚀 Открыть Mini App",
    }
    mini_button = next(button for button in buttons if button.web_app is not None)
    assert mini_button.web_app.url == "https://delta.example/app"


def test_menu_callback_edits_existing_message() -> None:
    """Переход по разделам не должен создавать цепочку новых сообщений."""
    router = create_menu_router()
    callback = Mock(spec=CallbackQuery)
    callback.data = "delta-menu:tools"
    callback.answer = AsyncMock()
    callback.message = Mock(spec=Message)
    callback.message.edit_text = AsyncMock()

    asyncio.run(router.callback_query.handlers[0].callback(callback))

    call = callback.message.edit_text.await_args
    assert call is not None
    assert "/id @username" in call.args[0]
    assert call.kwargs["reply_markup"].inline_keyboard[0][0].text == "⬅️ Назад"
    callback.answer.assert_awaited_once_with()


def test_menu_rejects_stale_callback() -> None:
    """Неизвестная старая кнопка должна получить понятный alert."""
    router = create_menu_router()
    callback = Mock(spec=CallbackQuery)
    callback.data = "delta-menu:removed"
    callback.answer = AsyncMock()
    callback.message = None

    asyncio.run(router.callback_query.handlers[0].callback(callback))

    callback.answer.assert_awaited_once_with(
        "Этот раздел больше недоступен.", show_alert=True
    )


def test_id_command_reports_current_user_chat_and_topic() -> None:
    """Без аргумента команда должна показать основные ID текущего контекста."""
    bot = AsyncMock(spec=Bot)
    router = create_utilities_router(cast(Bot, bot))
    message, answer = _message("/id")
    message.message_thread_id = 15

    asyncio.run(_call_message(router, message))

    call = answer.await_args
    assert call is not None
    assert "ID: 42" in call.args[0]
    assert "ID: -100777" in call.args[0]
    assert "ID темы: 15" in call.args[0]
    bot.get_chat.assert_not_awaited()


def test_id_command_resolves_public_username() -> None:
    """Аргумент @username должен разрешаться через Telegram getChat."""
    bot = AsyncMock(spec=Bot)
    bot.get_chat.return_value = Chat(
        id=-100999,
        type="channel",
        title="Новости",
        username="delta_news",
    )
    router = create_utilities_router(cast(Bot, bot))
    message, answer = _message("/id @delta_news")

    asyncio.run(_call_message(router, message))

    bot.get_chat.assert_awaited_once_with("@delta_news")
    call = answer.await_args
    assert call is not None
    assert "Тип: канал" in call.args[0]
    assert "ID: -100999" in call.args[0]


def test_id_command_explains_unavailable_public_object() -> None:
    """Ошибка Telegram getChat должна превращаться в полезную подсказку."""
    bot = AsyncMock(spec=Bot)
    bot.get_chat.side_effect = TelegramBadRequest(
        method=Mock(), message="chat not found"
    )
    router = create_utilities_router(cast(Bot, bot))
    message, answer = _message("/id @missing")

    asyncio.run(_call_message(router, message))

    call = answer.await_args
    assert call is not None
    assert "Не смог получить" in call.args[0]


def test_id_command_uses_replied_sender() -> None:
    """Ответ на сообщение должен показывать ID его автора."""
    bot = AsyncMock(spec=Bot)
    router = create_utilities_router(cast(Bot, bot))
    message, answer = _message("/id")
    message.reply_to_message = cast(
        Any,
        SimpleNamespace(
            from_user=User(id=88, is_bot=False, first_name="Автор"),
            sender_chat=None,
        ),
    )

    asyncio.run(_call_message(router, message))

    call = answer.await_args
    assert call is not None
    assert "Автор сообщения" in call.args[0]
    assert "ID: 88" in call.args[0]
