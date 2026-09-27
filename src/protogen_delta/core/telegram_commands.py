"""Настройка списка команд Telegram-бота."""

from aiogram import Bot
from aiogram.types import BotCommand


async def set_commands(bot: Bot) -> None:
    """Установить команды, отображаемые в меню Telegram."""
    commands = [
        BotCommand(command="start", description="🚀 Запустить бота"),
        BotCommand(command="randomart", description="🎨 Случайный арт"),
        BotCommand(command="e6", description="🔎 Поиск артов e621 по тегам"),
        BotCommand(command="rp", description="🎭 Управление RP — /rp off"),
        BotCommand(command="adult", description="🔞 Выбрать возрастной режим"),
        BotCommand(command="reset", description="🧹 Полностью очистить память"),
        BotCommand(command="help", description="ℹ️ Помощь"),
    ]

    await bot.set_my_commands(commands)
