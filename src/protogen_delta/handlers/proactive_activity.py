"""Начинать отсчёт паузы после настоящей активности личного диалога."""

import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from time import time
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
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.memories import MemoriesRepository

logger = logging.getLogger(__name__)
proactive_delivery: ContextVar[bool] = ContextVar("proactive_delivery", default=False)


class ProactiveIncomingMiddleware(BaseMiddleware):
    def __init__(
        self,
        repository: MemoriesRepository,
        states: UserStateStore,
        operations: InputOperations,
        *,
        clock: Callable[[], float] = time,
    ) -> None:
        self._repository, self._states, self._operations, self._clock = (
            repository,
            states,
            operations,
            clock,
        )

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
            and event.chat.type == "private"
            and event.chat.id == event.from_user.id
        ):
            try:
                async with self._states.use_activity(event.from_user.id):
                    ticket = input_ticket(data)
                    if ticket is not None and not self._operations.is_current(ticket):
                        return None
                    await self._repository.note_user_activity(
                        event.from_user.id, self._clock()
                    )
            except Exception:
                logger.error("Не удалось учесть активность личного диалога")
        return await handler(event, data)


class ProactiveOutgoingMiddleware(BaseRequestMiddleware):
    def __init__(
        self, repository: MemoriesRepository, *, clock: Callable[[], float] = time
    ) -> None:
        self._repository, self._clock = repository, clock

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ) -> Response[TelegramType]:
        background = proactive_delivery.get()
        response = await make_request(bot, method)
        if background:
            return response
        value = response.result if isinstance(response, Response) else response
        messages = value if isinstance(value, list) else [value]
        owners = {
            message.chat.id
            for message in messages
            if isinstance(message, Message)
            and message.chat.type == "private"
            and message.chat.id > 0
        }
        for user_id in owners:
            try:
                await self._repository.note_dialogue_activity(user_id, self._clock())
            except Exception:
                logger.error("Не удалось учесть доставку ответа в личном диалоге")
        return response
