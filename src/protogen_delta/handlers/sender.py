"""Передавать модели фактический аккаунт отправителя без разбора префиксов."""

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from protogen_delta.core.sender import Sender, current_sender


class SenderMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        sender = None
        if isinstance(event, Message) and event.from_user is not None:
            user = event.from_user
            sender = Sender(
                user.id,
                user.full_name[:160],
                (user.username or "")[:64],
                event.forward_origin is not None,
            )
        token = current_sender.set(sender)
        try:
            return await handler(event, data)
        finally:
            current_sender.reset(token)
