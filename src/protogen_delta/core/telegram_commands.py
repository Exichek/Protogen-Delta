"""Настройка списка команд Telegram-бота."""

from aiogram import Bot
from aiogram.types import (
    BotCommand,
    MenuButtonCommands,
    MenuButtonWebApp,
    WebAppInfo,
)


async def set_commands(bot: Bot, mini_app_url: str | None = None) -> None:
    """Установить команды, отображаемые в меню Telegram."""
    commands = [
        BotCommand(command="start", description="🚀 Запустить бота"),
        BotCommand(command="menu", description="⚙️ Панель возможностей"),
        BotCommand(command="randomart", description="🎨 Случайный арт"),
        BotCommand(command="e6", description="🔎 Поиск артов e621 по тегам"),
        BotCommand(command="rp", description="🎭 Управление RP — /rp off"),
        BotCommand(command="adult", description="🔞 Выбрать возрастной режим"),
        BotCommand(command="id", description="🪪 Узнать Telegram ID"),
        BotCommand(command="download", description="📥 Скачать видео по ссылке"),
        BotCommand(command="source", description="🔎 Найти источник арта (ответом)"),
        BotCommand(command="reset", description="🧹 Полностью очистить память"),
        BotCommand(command="help", description="ℹ️ Помощь"),
    ]

    await bot.set_my_commands(commands)
    menu_button = (
        MenuButtonWebApp(
            text="Открыть Дельту",
            web_app=WebAppInfo(url=mini_app_url),
        )
        if mini_app_url is not None
        else MenuButtonCommands()
    )
    await bot.set_chat_menu_button(menu_button=menu_button)
