"""Не позволять старым входящим сообщениям восстановить память после сброса."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from protogen_delta.core.async_completion import finish_operation

_TICKET_KEY = "_delta_input_ticket"


@dataclass(slots=True)
class _UserInputs:
    generation: int = 0
    tasks: dict[asyncio.Task[Any], int] = field(default_factory=dict)
    memory_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    barrier: asyncio.Event | None = None


@dataclass(frozen=True, slots=True)
class InputTicket:
    """Сохранить поколение запроса без текста, файлов и пользовательских фактов."""

    entry: _UserInputs
    generation: int
    task: asyncio.Task[Any]


class InputOperations:
    """Следить только за выполняющимися запросами; команды не отменять."""

    def __init__(self) -> None:
        self._users: dict[int, _UserInputs] = {}

    def _release(self, user_id: int, entry: _UserInputs) -> None:
        if not entry.tasks and entry.barrier is None:
            if self._users.get(user_id) is entry:
                del self._users[user_id]

    @asynccontextmanager
    async def receive(self, user_id: int) -> AsyncIterator[InputTicket]:
        """Пометить момент получения до статистики и выбора обработчика."""
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Входящий запрос должен выполняться в asyncio.Task")
        while True:
            entry = self._users.setdefault(user_id, _UserInputs())
            if entry.barrier is None:
                break
            await entry.barrier.wait()
        entry.tasks[task] = entry.tasks.get(task, 0) + 1
        try:
            yield InputTicket(entry, entry.generation, task)
        finally:
            count = entry.tasks[task] - 1
            if count:
                entry.tasks[task] = count
            else:
                del entry.tasks[task]
                entry.memory_tasks.discard(task)
            self._release(user_id, entry)

    @staticmethod
    def is_current(ticket: InputTicket) -> bool:
        """Проверить, не было ли сброса после получения сообщения."""
        return ticket.generation == ticket.entry.generation

    def start_memory(self, ticket: InputTicket) -> bool:
        """Выделить выбранный диалоговый обработчик, оставив команды в покое."""
        if not self.is_current(ticket) or ticket.entry.barrier is not None:
            return False
        ticket.entry.memory_tasks.add(ticket.task)
        return True

    @asynccontextmanager
    async def resetting(
        self, user_id: int, *, exempt: asyncio.Task[Any] | None = None
    ) -> AsyncIterator[None]:
        """Под общим барьером состояния остановить старые диалоговые запросы."""
        while True:
            entry = self._users.setdefault(user_id, _UserInputs())
            if entry.barrier is None:
                break
            await entry.barrier.wait()
        barrier = asyncio.Event()
        entry.barrier = barrier
        entry.generation += 1
        current = asyncio.current_task()
        tasks = [
            task
            for task in entry.memory_tasks
            if task is not current and task is not exempt
        ]
        try:
            for task in tasks:
                task.cancel()
            if tasks:
                await finish_operation(asyncio.gather(*tasks, return_exceptions=True))
            yield
        finally:
            entry.barrier = None
            barrier.set()
            self._release(user_id, entry)


class ReceiveInputsMiddleware(BaseMiddleware):
    """Регистрировать запрос до любой асинхронной обработки сообщения."""

    def __init__(self, operations: InputOperations) -> None:
        self._operations = operations

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
        ):
            return await handler(event, data)
        async with self._operations.receive(event.from_user.id) as ticket:
            previous = data.get(_TICKET_KEY)
            data[_TICKET_KEY] = ticket
            try:
                return await handler(event, data)
            finally:
                if previous is None:
                    data.pop(_TICKET_KEY, None)
                else:
                    data[_TICKET_KEY] = previous


class MemoryInputsMiddleware(BaseMiddleware):
    """Применять только к тексту и медиа для диалога, после выбора роутера."""

    def __init__(self, operations: InputOperations) -> None:
        self._operations = operations

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        ticket = data.get(_TICKET_KEY)
        if isinstance(ticket, InputTicket) and not self._operations.start_memory(
            ticket
        ):
            return None
        return await handler(event, data)


def input_ticket(data: dict[str, Any]) -> InputTicket | None:
    """Получить билет для согласованной записи статистики."""
    ticket = data.get(_TICKET_KEY)
    return ticket if isinstance(ticket, InputTicket) else None
