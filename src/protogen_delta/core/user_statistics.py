"""Учёт каждого входящего сообщения ровно один раз."""

import logging
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.input_operations import InputOperations, input_ticket
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.user_statistics import UserStatisticsRepository

logger = logging.getLogger(__name__)


class UserStatisticsMiddleware(BaseMiddleware):
    def __init__(
        self,
        repository: UserStatisticsRepository,
        user_states: UserStateStore | None = None,
        input_operations: InputOperations | None = None,
    ) -> None:
        self._repository = repository
        self._user_states = user_states
        self._input_operations = input_operations

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
        ):
            try:
                kind = (
                    "command"
                    if (event.text or "").startswith("/")
                    else event.content_type
                )
                async with (
                    self._user_states.use_activity(event.from_user.id)
                    if self._user_states is not None
                    else nullcontext()
                ):
                    ticket = input_ticket(data)
                    if (
                        ticket is None
                        or self._input_operations is None
                        or self._input_operations.is_current(ticket)
                    ):
                        await finish_operation(
                            self._repository.record(
                                event.from_user.id,
                                event.from_user.full_name,
                                event.from_user.username or "",
                                event.chat.id,
                                event.message_id,
                                kind,
                                event.date.timestamp(),
                            )
                        )
            except Exception as error:
                logger.warning("User statistics unavailable: %s", type(error).__name__)
        return await handler(event, data)
