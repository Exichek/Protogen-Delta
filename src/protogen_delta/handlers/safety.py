"""Явный возраст в тексте или подписи обрабатывается до загрузки файлов."""

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from protogen_delta.core.conversation_safety import declares_minor
from protogen_delta.services.response_engine import ResponseEngine


class AgeSafetyMiddleware(BaseMiddleware):
    def __init__(self, engine: ResponseEngine) -> None:
        self._engine = engine

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if (
            isinstance(event, Message)
            and event.from_user
            and not event.from_user.is_bot
            and event.forward_origin is None
            and declares_minor(event.text or event.caption or "")
        ):
            await self._engine.restrict_minor(event.from_user.id)
            await event.answer(
                "Спасибо, что сказал. Остаёмся в обычном общении, взрослый режим выключен."
            )
            return None
        return await handler(event, data)
