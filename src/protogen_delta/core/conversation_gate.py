"""Параллельные чаты и исключительные операции над общими настройками."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from protogen_delta.core.async_completion import finish_operation


class ConversationGate:
    """Пускать чаты вместе, отдавая приоритет ожидающим изменениям пользователя."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and not self._waiting_writers
            )
            self._readers += 1
        try:
            yield
        finally:
            await finish_operation(self._release_shared())

    async def _release_shared(self) -> None:
        async with self._condition:
            self._readers -= 1
            self._condition.notify_all()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writer and not self._readers
                )
                self._writer = True
            finally:
                self._waiting_writers -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            await finish_operation(self._release_exclusive())

    async def _release_exclusive(self) -> None:
        async with self._condition:
            self._writer = False
            self._condition.notify_all()
