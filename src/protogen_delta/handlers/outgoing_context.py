"""Учитывать только подтверждённую Telegram отправку в личный чат."""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import time

from aiogram import Bot
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.methods import Response, TelegramMethod
from aiogram.methods.base import TelegramType
from aiogram.types import Message

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.tool_markup import contains_tool_markup
from protogen_delta.repositories.outgoing_messages import OutgoingMessagesRepository

logger = logging.getLogger(__name__)
_delivery_kind: ContextVar[str] = ContextVar("outgoing_context_kind", default="bot")


@contextmanager
def outgoing_kind(kind: str) -> Iterator[None]:
    token = _delivery_kind.set(kind)
    try:
        yield
    finally:
        _delivery_kind.reset(token)


class OutgoingContextMiddleware(BaseRequestMiddleware):
    def __init__(
        self,
        repository: OutgoingMessagesRepository,
        *,
        clock: Callable[[], float] = time,
    ) -> None:
        self._repository, self._clock = repository, clock

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ) -> Response[TelegramType]:
        kind = _delivery_kind.get()
        response = await make_request(bot, method)
        # Редактирование/чтение старого сообщения не является новой отправкой.
        if kind == "dialogue" or not method.__api_method__.startswith("send"):
            return response
        value = response.result if isinstance(response, Response) else response
        for message in value if isinstance(value, list) else [value]:
            if (
                not isinstance(message, Message)
                or message.chat.type != "private"
                or message.chat.id <= 0
            ):
                continue
            text = message.text or message.caption or ""
            if not text and message.sticker:
                text = (
                    "[Отправлен стикер"
                    + (" " + message.sticker.emoji if message.sticker.emoji else "")
                    + "]"
                )
            if not text:
                for attribute, label in (
                    ("animation", "анимация"),
                    ("video", "видео"),
                    ("photo", "фото"),
                    ("voice", "голосовое сообщение"),
                    ("audio", "аудио"),
                    ("document", "документ"),
                    ("video_note", "видеосообщение"),
                ):
                    if getattr(message, attribute, None):
                        text = f"[Отправлено вложение: {label}; содержимое в текстовом контексте отсутствует]"
                        break
            if not text or contains_tool_markup(text):
                continue
            try:
                await finish_operation(
                    self._repository.record(
                        message.chat.id, message.message_id, text, kind, self._clock()
                    )
                )
            except Exception:
                # Отправку не повторять из-за сбоя учёта. Не логировать личный текст.
                logger.warning("Не удалось сохранить контекст доставленного сообщения")
        return response
