"""Владелец API-расходов задаётся из Telegram, а не из текста пользователя."""

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from protogen_delta.core.llm_budget import LlmBudgetError, llm_user


class LlmUserMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = getattr(event, "from_user", None)
        with llm_user(user.id if user is not None and not user.is_bot else None):
            try:
                return await handler(event, data)
            except LlmBudgetError as error:
                if isinstance(event, Message):
                    await event.answer(str(error))
                elif isinstance(event, CallbackQuery):
                    await event.answer(str(error), show_alert=True)
                return None
