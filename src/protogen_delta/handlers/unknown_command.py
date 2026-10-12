"""Обработчик неизвестных команд Telegram."""

import random

from aiogram import F, Router
from aiogram.types import Message

UNKNOWN_COMMAND_REPLIES = (
    "Бзз... команда не найдена, ты че там удумал?🐾",
    "Такой команды нет в моей прошивке! UwU",
    "Ошибка 4787: команда не существует >w<",
    "Бзз! Ты ввёл что-то странное, попробуй /help 💜",
    "Попробуй снова",
    "Try again?",
    "Oh stop it, you! Unknown command. :)",
)


def create_unknown_command_router() -> Router:
    """Создать роутер для неизвестных Telegram-команд."""
    router = Router(name=__name__)

    @router.message(F.text.startswith("/"))
    async def unknown_command(message: Message) -> None:
        """Ответить на команду, которую не обработали другие роутеры."""
        await message.reply(random.choice(UNKNOWN_COMMAND_REPLIES))

    return router
