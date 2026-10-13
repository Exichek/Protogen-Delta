"""Архивировать входящие сообщения и подтверждённую Telegram доставку."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.methods import Response, TelegramMethod
from aiogram.methods.base import TelegramType
from aiogram.types import Message, TelegramObject

from protogen_delta.core.input_operations import InputOperations, input_ticket
from protogen_delta.repositories.dialogue_archive import DialogueMessage
from protogen_delta.services.dialogue_archive import (
    DialogueArchiveService,
    archive_generation,
    current_archive_message,
)

logger = logging.getLogger(__name__)


async def archive_edit(message: Message) -> None:
    """Получать edited_message для архивного middleware без повторного ответа бота."""
    return None


def archive_message(message: Message, *, incoming: bool) -> DialogueMessage:
    """Сохранить текст/подпись и вид вложения без токенов и ссылок скачивания."""
    kind = message.content_type
    text = message.text or message.caption or ""
    if message.document and message.document.file_name:
        text = f"[Файл: {message.document.file_name[:512]}]\n{text}"
    if message.forward_origin is not None:
        text = f"[Пересланное сообщение: цитата]\n{text}"
    if kind != "text":
        label = f"[Вложение: {kind}]"
        text = f"{label}\n{text}" if text else label
    return DialogueMessage(
        f"telegram:{message.message_id}",
        "user" if incoming else "assistant",
        text or "[Служебное сообщение]",
        message.date.timestamp(),
        kind,
    )


class ArchiveIncomingMiddleware(BaseMiddleware):
    def __init__(
        self, archive: DialogueArchiveService, operations: InputOperations
    ) -> None:
        self._archive, self._operations = archive, operations

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if (
            not isinstance(event, Message)
            or not event.from_user
            or event.from_user.is_bot
            or event.chat.type != "private"
            or event.chat.id != event.from_user.id
            or not self._archive.enabled_for(event.from_user.id)
        ):
            return await handler(event, data)
        ticket = input_ticket(data)
        if ticket is not None and not self._operations.is_current(ticket):
            return None
        user_id = event.from_user.id
        generation = self._archive.generation(user_id)
        token = archive_generation.set((user_id, generation))
        message_token = current_archive_message.set(f"telegram:{event.message_id}")
        try:
            try:
                await self._archive.record(
                    user_id, archive_message(event, incoming=True), generation
                )
            except Exception:
                logger.error("Не удалось архивировать входящее сообщение")
            return await handler(event, data)
        finally:
            archive_generation.reset(token)
            current_archive_message.reset(message_token)


class ArchiveOutgoingMiddleware(BaseRequestMiddleware):
    def __init__(self, archive: DialogueArchiveService) -> None:
        self._archive = archive

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ) -> Response[TelegramType]:
        chat_id = getattr(method, "chat_id", None)
        owner = (
            chat_id
            if type(chat_id) is int and self._archive.enabled_for(chat_id)
            else None
        )
        scope = archive_generation.get()
        generation = (
            scope[1]
            if scope and scope[0] == owner
            else self._archive.generation(owner or 0)
        )
        response = await make_request(bot, method)
        if owner is None:
            return response
        value = response.result if isinstance(response, Response) else response
        messages = value if isinstance(value, list) else [value]
        for message in messages:
            if (
                not isinstance(message, Message)
                or message.chat.type != "private"
                or message.chat.id != owner
            ):
                continue
            try:
                await self._archive.record(
                    owner, archive_message(message, incoming=False), generation
                )
            except Exception:
                # Ошибка архива не должна повторно отправлять уже доставленный ответ.
                logger.error("Не удалось архивировать доставленный ответ")
        return response
