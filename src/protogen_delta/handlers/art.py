"""Обработчики загрузки и выдачи артов."""

import logging
import random
from typing import Literal

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.images import ImagesRepository

logger = logging.getLogger(__name__)


def create_art_router(
    images_repository: ImagesRepository,
    art_chat_id: int,
    admin_ids: frozenset[int] = frozenset(),
    sources: ArtSourcesRepository | None = None,
) -> Router:
    """Создать роутер для работы с артами."""
    router = Router(name=__name__)

    def is_admin(message: Message) -> bool:
        return message.from_user is not None and message.from_user.id in admin_ids

    def allowed(message: Message) -> bool:
        chats = sources.get_all() if sources is not None else [art_chat_id]
        return message.chat.id in chats and is_admin(message)

    @router.message(F.photo)
    async def save_photo(message: Message) -> None:
        """Сохранить арт, присланный в разрешённую группу."""
        if not allowed(message):
            return

        if not message.photo:
            return

        file_id = message.photo[-1].file_id

        if images_repository.add(file_id):
            logger.info("Сохранён новый арт: %s", file_id)

    @router.message(F.document.mime_type.startswith("image/"))
    async def save_document(message: Message) -> None:
        """Сохранить изображение из разрешённой группы."""
        if not allowed(message):
            return

        document = message.document

        if (
            document is None
            or document.mime_type is None
            or not document.mime_type.startswith("image/")
        ):
            return

        if images_repository.add(document.file_id, kind="document"):
            logger.info(
                "Сохранён новый арт-документ: %s",
                document.file_id,
            )
        else:
            logger.info(
                "Арт-документ уже есть в базе: %s",
                document.file_id,
            )

    @router.message(Command("randomart"))
    async def random_art(message: Message) -> None:
        """Отправить случайный арт из локального хранилища."""
        images = images_repository.get_all()

        if not images:
            await message.answer("База пустая 😢 сначала добавь арты.")
            return

        file_id = random.choice(images)

        try:
            send = (
                message.answer_document
                if images_repository.get_kind(file_id) == "document"
                else message.answer_photo
            )
            await send(
                file_id,
                caption="🎨 Лови артик!",
            )
        except TelegramAPIError:
            logger.warning(
                "Не удалось отправить случайный арт: %s",
                file_id,
                exc_info=True,
            )

            await message.answer("Не смог отправить арт 😢 попробуй ещё раз.")
            return

        logger.info(
            "Выдан случайный арт: %s",
            file_id,
        )

    @router.message(Command("artchat"))
    async def art_chat(message: Message) -> None:
        if not is_admin(message):
            await message.answer("⛔ У тебя нет доступа к этой команде.")
            return
        if sources is None:
            await message.answer("Управление группами не настроено.")
            return
        parts = (message.text or "").split()
        if len(parts) == 1 or parts[1] == "list":
            await message.answer(
                "Группы артов: " + ", ".join(map(str, sources.get_all()))
            )
            return
        try:
            if parts[1] not in ("add", "remove") or len(parts) > 3:
                raise ValueError
            chat_id = int(parts[2]) if len(parts) == 3 else message.chat.id
            changed = sources.change(chat_id, add=parts[1] == "add")
        except ValueError:
            await message.answer(
                "Используй /artchat add -100123, /artchat remove -100123 или /artchat list."
            )
            return
        await message.answer(
            "Список групп обновлён." if changed else "Список уже в таком состоянии."
        )

    @router.message(Command("addimage"))
    async def add_image(message: Message) -> None:
        if not is_admin(message):
            await message.answer("⛔ У тебя нет доступа к этой команде.")
            return
        replied = message.reply_to_message
        kind: Literal["photo", "document"] = "photo"
        parts = (message.text or "").split(maxsplit=1)
        if replied is not None and replied.photo:
            ids = [replied.photo[-1].file_id]
        elif (
            replied is not None
            and replied.document is not None
            and (replied.document.mime_type or "").startswith("image/")
        ):
            ids = [replied.document.file_id]
            kind = "document"
        else:
            ids = parts[1].replace(",", " ").split() if len(parts) == 2 else []
        if not ids or len(ids) > 100 or any(len(value) > 512 for value in ids):
            await message.answer(
                "Ответь /addimage на фото или передай до 100 file_id через пробел/запятую."
            )
            return
        added = sum(
            images_repository.add(value, kind=kind) for value in dict.fromkeys(ids)
        )
        await message.answer(
            f"Добавлено артов: {added}. Уже были: {len(set(ids)) - added}."
        )

    return router
