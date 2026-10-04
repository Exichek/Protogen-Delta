"""Обработчик команды /help."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.message_utils import split_message

HELP_TEXT = (
    "📖 ИИ-ассистент Протоген Дельта — помощь и команды:\n\n"
    "/start – 🚀 Познакомиться с Дельтой\n"
    "/menu – ⚙️ Открыть панель возможностей\n"
    "/randomart – 🎨 Случайный арт\n"
    "/e6 <теги> – 🔎 Поиск артов e621/e926 без повторов\n"
    "/rp off – 🎭 Завершить RP-режим\n"
    "/adult – 🔞 Выбрать возрастной режим общения\n"
    "/id [@username] – 🪪 Узнать Telegram ID\n"
    "/download <ссылка> – 📥 Скачать публичное видео\n"
    "/source – 🔎 Найти источник арта (ответом на фото, через SauceNAO)\n"
    "/reset – 🧹 Полностью очистить память о тебе\n"
    "/help – ℹ️ Помощь (это сообщение)\n"
    "\nМожно спросить дату и время, официальный курс валют ЦБ РФ "
    "или погоду — укажи город и страну. Дельта также умеет искать свежую "
    "информацию в интернете, искать арты по тегам (например, "
    "`/e6 dragon order:favcount`), рассматривать изображения и стикеры, читать "
    "поддерживаемые документы и отвечать на голосовые сообщения.\n"
    "\nRP включается сообщением «Давай начнём РП» или действием вроде "
    "*касаюсь твоего визора*. /rp off завершает сцену, сохраняя RP-профиль.\n"
)


def create_help_router() -> Router:
    """Создать роутер команды /help."""
    router = Router(name=__name__)

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        """Отправить пользователю список доступных команд."""
        for chunk in split_message(HELP_TEXT):
            await message.answer(chunk)

    return router
