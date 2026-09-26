"""Админские команды Telegram-бота."""

import asyncio
import logging
import time
from html import escape

from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.state import BotState
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.users import UsersRepository

logger = logging.getLogger(__name__)

MAX_LIST_IMAGES = 200
LIST_IMAGES_DELAY_SECONDS = 1.0


def _art_word(count: int) -> str:
    """Вернуть правильную форму слова «арт» для количества."""
    last_two_digits = count % 100

    if 11 <= last_two_digits <= 14:
        return "артов"

    last_digit = count % 10

    if last_digit == 1:
        return "арт"

    if 2 <= last_digit <= 4:
        return "арта"

    return "артов"


def _parse_list_images_count(
    text: str | None,
) -> int | None:
    """Получить количество артов из команды /listimages."""
    parts = (text or "").strip().split(maxsplit=1)

    if len(parts) == 1:
        return 1

    raw_count = parts[1].strip()

    if not raw_count.isdigit():
        return None

    count = int(raw_count)

    if count < 1:
        return None

    return min(count, MAX_LIST_IMAGES)


def _parse_image_ids(
    text: str | None,
) -> list[str]:
    """Получить уникальные file_id из команды /removeimage."""
    parts = (text or "").strip().split(maxsplit=1)

    if len(parts) < 2:
        return []

    file_ids = [file_id.strip() for file_id in parts[1].split(",") if file_id.strip()]

    return list(dict.fromkeys(file_ids))


def create_admin_router(
    images_repository: ImagesRepository,
    users_repository: UsersRepository,
    bot_state: BotState,
    admin_ids: frozenset[int],
) -> Router:
    """Создать роутер административных команд."""
    router = Router(name=__name__)

    def is_admin(message: Message) -> bool:
        """Проверить, принадлежит ли сообщение администратору."""
        return message.from_user is not None and message.from_user.id in admin_ids

    async def deny_access(message: Message) -> None:
        """Сообщить пользователю об отсутствии доступа."""
        await message.answer("⛔ У тебя нет доступа к этой команде.")

    @router.message(Command("listimages"))
    async def list_images(message: Message) -> None:
        """Показать последние сохранённые арты."""
        if not is_admin(message):
            await deny_access(message)
            return

        count = _parse_list_images_count(message.text)

        if count is None:
            await message.answer(
                "⚠️ Укажи положительное целое количество артов.\n"
                "Пример: /listimages 10"
            )
            return

        images = images_repository.get_all()

        if not images:
            await message.answer("📂 База артов пуста.")
            return

        last_images = images[-count:][::-1]

        await message.answer(
            f"📂 Показываю: {len(last_images)} " f"{_art_word(len(last_images))}."
        )

        for index, file_id in enumerate(last_images):
            try:
                send = (
                    message.answer_document
                    if images_repository.get_kind(file_id) == "document"
                    else message.answer_photo
                )
                await send(
                    file_id,
                    caption=f"<code>{escape(file_id)}</code>",
                    parse_mode="HTML",
                )
            except TelegramAPIError:
                logger.warning(
                    "Не удалось отправить арт с file_id=%s",
                    file_id,
                    exc_info=True,
                )

                await message.answer(f"⚠️ Не удалось отправить арт с ID: {file_id}")

            if index < len(last_images) - 1:
                await asyncio.sleep(
                    LIST_IMAGES_DELAY_SECONDS,
                )

    @router.message(Command("removeimage"))
    async def remove_image(message: Message) -> None:
        """Удалить арты по Telegram file_id."""
        if not is_admin(message):
            await deny_access(message)
            return

        ids_to_remove = _parse_image_ids(message.text)

        if not ids_to_remove:
            await message.answer(
                "⚠️ Укажи ID артов через запятую.\n" "Пример: /removeimage id1,id2,id3"
            )
            return

        removed = 0
        not_found = 0

        for file_id in ids_to_remove:
            if images_repository.remove(file_id):
                removed += 1
            else:
                not_found += 1

        reply: list[str] = []

        if removed:
            reply.append(f"✅ Удалено: {removed} {_art_word(removed)}")

        if not_found:
            reply.append(f"⚠️ Не найдено: {not_found} " f"{_art_word(not_found)}")

        await message.answer("\n".join(reply) if reply else "⚠️ Ничего не удалено.")

    @router.message(Command("artcount"))
    async def art_count(message: Message) -> None:
        """Показать количество сохранённых артов."""
        if not is_admin(message):
            await deny_access(message)
            return

        count = images_repository.count()

        if count == 0:
            await message.answer("📂 База артов пуста.")
            return

        await message.answer(f"📂 В базе {count} {_art_word(count)}.")

    @router.message(Command("status"))
    async def status(message: Message) -> None:
        """Показать состояние текущего процесса бота."""
        if not is_admin(message):
            await deny_access(message)
            return

        uptime = int(time.monotonic() - bot_state.start_time)
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60

        reply = (
            "📊 Статус бота:\n"
            f"• Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
            f"• Пользователей: {users_repository.count()}\n"
            f"• Ответов отправлено: {bot_state.reply_count}"
        )

        await message.answer(reply)

    @router.message(Command("ownhelp"))
    async def own_help(message: Message) -> None:
        """Показать список административных команд."""
        if not is_admin(message):
            await deny_access(message)
            return

        help_text = (
            "📖 Админские команды:\n\n"
            "/addimage — добавить фото ответом или по file_id\n"
            "/artchat list|add|remove — группы-источники артов\n"
            "/listimages <N> — показать последние N артов\n"
            "/removeimage <id1,id2,...> — удалить арты по ID\n"
            "/artcount — показать количество артов\n"
            "/status — показать статус бота\n"
            "/message <user_id> <текст> — сообщение от создателя\n"
            "/broadcast <текст> — рассылка с подтверждением (только создатель)\n"
            "/ping — проверить доступность\n"
            "/ownhelp — показать эту справку"
        )

        await message.answer(help_text)

    @router.message(Command("ping"))
    async def ping(message: Message) -> None:
        """Проверить доступность административного роутера."""
        if not is_admin(message):
            await deny_access(message)
            return

        await message.answer("🏓 Pong от админского роутера!")

    return router
