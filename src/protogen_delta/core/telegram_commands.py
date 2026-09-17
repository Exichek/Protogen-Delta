"""Настройка списка команд Telegram-бота."""

from aiogram import Bot
from aiogram.types import BotCommand


async def set_commands(bot: Bot) -> None:
    """Установить команды, отображаемые в меню Telegram."""
    commands = [
        BotCommand(command="start", description="🚀 Запустить бота"),
        BotCommand(command="randomart", description="🎨 Случайный арт"),
        BotCommand(command="reset", description="🧹 Очистить память диалога"),
        BotCommand(command="help", description="ℹ️ Помощь"),
    ]

    await bot.set_my_commands(commands)
