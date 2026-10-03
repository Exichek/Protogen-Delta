"""Небольшие Telegram-утилиты, доступные прямо из бота."""

from aiogram import Bot, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Chat, Message, User


def _display_name(chat: Chat) -> str | None:
    """Вернуть читаемое имя пользователя, группы или канала."""
    if chat.title:
        return chat.title
    full_name = " ".join(
        part for part in (chat.first_name, chat.last_name) if part
    ).strip()
    return full_name or None


def _chat_lines(chat: Chat, heading: str) -> list[str]:
    """Представить Telegram-чат несколькими безопасными строками."""
    labels = {
        "private": "личный чат",
        "group": "группа",
        "supergroup": "супергруппа",
        "channel": "канал",
    }
    lines = [heading, f"Тип: {labels.get(chat.type, chat.type)}"]
    name = _display_name(chat)
    if name:
        lines.append(f"Название: {name}")
    if chat.username:
        lines.append(f"Username: @{chat.username}")
    lines.append(f"ID: {chat.id}")
    return lines


def _user_lines(user: User, heading: str) -> list[str]:
    """Представить Telegram-пользователя несколькими строками."""
    lines = [heading, f"Имя: {user.full_name}"]
    if user.username:
        lines.append(f"Username: @{user.username}")
    lines.append(f"ID: {user.id}")
    return lines


def _argument(message: Message) -> str:
    """Извлечь аргумент команды без зависимости от CommandObject."""
    text = message.text or ""
    _, separator, tail = text.partition(" ")
    return tail.strip() if separator else ""


def create_utilities_router(bot: Bot) -> Router:
    """Создать роутер служебных команд."""
    router = Router(name=__name__)

    @router.message(Command("id"))
    async def id_command(message: Message) -> None:
        """Показать ID текущего или указанного Telegram-объекта."""
        query = _argument(message)
        if query:
            lookup: str | int = int(query) if query.lstrip("-").isdigit() else query
            try:
                chat = await bot.get_chat(lookup)
            except TelegramAPIError:
                await message.answer(
                    "Не смог получить этот объект. Для публичной группы или канала "
                    "укажи @username; приватный чат должен быть доступен боту."
                )
                return
            await message.answer("\n".join(_chat_lines(chat, "🔎 Найдено")))
            return

        blocks: list[str] = []
        replied = message.reply_to_message
        if replied is not None:
            if replied.from_user is not None:
                blocks.extend(_user_lines(replied.from_user, "👤 Автор сообщения"))
            if replied.sender_chat is not None:
                if blocks:
                    blocks.append("")
                blocks.extend(_chat_lines(replied.sender_chat, "📣 Отправитель-чат"))

        if not blocks and message.from_user is not None:
            blocks.extend(_user_lines(message.from_user, "👤 Ты"))

        if blocks:
            blocks.append("")
        blocks.extend(_chat_lines(message.chat, "💬 Текущий чат"))
        if message.message_thread_id is not None:
            blocks.append(f"ID темы: {message.message_thread_id}")
        blocks.extend(
            [
                "",
                "Для публичной группы или канала: /id @username",
                "Для автора сообщения: ответь на него командой /id",
            ]
        )
        await message.answer("\n".join(blocks))

    return router
