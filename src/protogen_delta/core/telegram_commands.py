"""Настройка списка команд Telegram-бота."""

import logging
from collections.abc import Awaitable, Callable

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    MenuButtonCommands,
    MenuButtonWebApp,
    WebAppInfo,
)

from protogen_delta.core.user_state import ContentMode

ModeChange = Callable[[int, ContentMode], Awaitable[None]]
logger = logging.getLogger(__name__)


def commands_for_mode(mode: ContentMode = "unselected") -> list[BotCommand]:
    """Показать общую коллекцию только подтвердившим совершеннолетие."""
    commands = [
        BotCommand(command="start", description="🚀 Познакомиться с Дельтой"),
        BotCommand(command="menu", description="⚙️ Панель возможностей"),
        BotCommand(command="e6", description="🔎 Поиск артов e621 по тегам"),
        BotCommand(command="rp", description="🎭 Управление RP — /rp off"),
        BotCommand(command="adult", description="🔞 Выбрать возрастной режим"),
        BotCommand(command="id", description="🪪 Узнать Telegram ID"),
        BotCommand(command="download", description="📥 Скачать видео по ссылке"),
        BotCommand(command="source", description="🔎 Найти источник арта (ответом)"),
        BotCommand(command="reset", description="🧹 Полностью очистить память"),
        BotCommand(command="help", description="ℹ️ Помощь"),
    ]
    if mode == "adult":
        commands.insert(
            2, BotCommand(command="randomart", description="🎨 Случайный арт")
        )
    return commands


async def set_user_commands(bot: Bot, user_id: int, mode: ContentMode) -> None:
    """Обновить личное меню, не отправляя сообщений и не отменяя сохранение режима."""
    try:
        await bot.set_my_commands(
            commands_for_mode(mode), scope=BotCommandScopeChat(chat_id=user_id)
        )
    except TelegramAPIError:
        logger.warning("Не удалось обновить меню команд пользователя %s", user_id)


async def set_commands(bot: Bot, mini_app_url: str | None = None) -> None:
    """Установить безопасное меню по умолчанию."""
    await bot.set_my_commands(commands_for_mode())
    menu_button = (
        MenuButtonWebApp(
            text="Открыть Дельту",
            web_app=WebAppInfo(url=mini_app_url),
        )
        if mini_app_url is not None
        else MenuButtonCommands()
    )
    await bot.set_chat_menu_button(menu_button=menu_button)
