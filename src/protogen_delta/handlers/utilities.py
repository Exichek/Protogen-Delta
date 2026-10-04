"""Небольшие Telegram-утилиты, доступные прямо из бота."""

import re

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import (
    Chat,
    KeyboardButton,
    KeyboardButtonRequestChat,
    KeyboardButtonRequestUsers,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    User,
)

_USER_REQUEST = 6101
_BOT_REQUEST = 6102
_GROUP_REQUEST = 6103
_CHANNEL_REQUEST = 6104
_FORUM_REQUEST = 6105
_CLOSE_ID = "✖ Закрыть выбор ID"


def _id_keyboard() -> ReplyKeyboardMarkup:
    """Запросить ID без ограничений по Premium и без добавления бота в чат."""

    def person(text: str, request_id: int, is_bot: bool) -> KeyboardButton:
        return KeyboardButton(
            text=text,
            request_users=KeyboardButtonRequestUsers(
                request_id=request_id,
                user_is_bot=is_bot,
                max_quantity=1,
                request_name=True,
                request_username=True,
            ),
        )

    def chat(
        text: str, request_id: int, channel: bool, forum: bool | None = None
    ) -> KeyboardButton:
        return KeyboardButton(
            text=text,
            request_chat=KeyboardButtonRequestChat(
                request_id=request_id,
                chat_is_channel=channel,
                chat_is_forum=forum,
                request_title=True,
                request_username=True,
            ),
        )

    return ReplyKeyboardMarkup(
        keyboard=[
            [
                person("👤 Пользователь", _USER_REQUEST, False),
                person("🤖 Бот", _BOT_REQUEST, True),
            ],
            [
                chat("👥 Группа", _GROUP_REQUEST, False),
                chat("📣 Канал", _CHANNEL_REQUEST, True),
            ],
            [chat("💬 Форум", _FORUM_REQUEST, False, True)],
            [KeyboardButton(text=_CLOSE_ID)],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Выбери пользователя или чат для получения ID",
    )


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
            query = query.strip("[] ")
            match = re.fullmatch(
                r"https?://(?:t\.me|telegram\.me)/([\w]+)(?:/)?", query, re.I
            )
            if match:
                query = "@" + match[1]
            elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{3,31}", query):
                query = "@" + query
            lookup: str | int = int(query) if query.lstrip("-").isdigit() else query
            try:
                chat = await bot.get_chat(lookup)
            except TelegramAPIError:
                await message.answer(
                    "Не смог получить этот объект. Для публичной группы или канала "
                    "укажи @username. ID пользователя по его нику Telegram боту "
                    "не выдаёт: отправь /id и нажми «Пользователь»."
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
                "В личном чате можно выбрать пользователя, бота, группу или канал кнопкой ниже.",
            ]
        )
        await message.answer(
            "\n".join(blocks),
            reply_markup=(_id_keyboard() if message.chat.type == "private" else None),
        )

    @router.message(F.users_shared)
    async def shared_users(message: Message) -> None:
        """Показать предоставленные Telegram ID, даже без доступа getChat."""
        shared = message.users_shared
        if shared is None or shared.request_id not in {_USER_REQUEST, _BOT_REQUEST}:
            return
        lines = ["🪪 Выбранные пользователи"]
        for user in shared.users:
            name = " ".join(part for part in (user.first_name, user.last_name) if part)
            if name:
                lines.append(f"Имя: {name}")
            if user.username:
                lines.append(f"Username: @{user.username}")
            lines.append(f"ID: {user.user_id}")
        await message.answer("\n".join(lines), reply_markup=ReplyKeyboardRemove())

    @router.message(F.chat_shared)
    async def shared_chat(message: Message) -> None:
        """Показать выбранный чат без запроса членства или прав администратора."""
        shared = message.chat_shared
        if shared is None or shared.request_id not in {
            _GROUP_REQUEST,
            _CHANNEL_REQUEST,
            _FORUM_REQUEST,
        }:
            return
        lines = ["🪪 Выбранный чат"]
        if shared.title:
            lines.append(f"Название: {shared.title}")
        if shared.username:
            lines.append(f"Username: @{shared.username}")
        lines.append(f"ID: {shared.chat_id}")
        await message.answer("\n".join(lines), reply_markup=ReplyKeyboardRemove())

    @router.message(F.text == _CLOSE_ID)
    async def close_picker(message: Message) -> None:
        await message.answer("Выбор ID закрыт.", reply_markup=ReplyKeyboardRemove())

    return router
