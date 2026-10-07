"""Ограничить реально работающие потоки, включая отменённые запросы."""

import asyncio
from collections.abc import Callable
from typing import Any


class BlockingWorkPool:
    def __init__(self, limit: int = 1) -> None:
        if limit < 1:
            raise ValueError("limit должен быть положительным")
        self._slots = asyncio.Semaphore(limit)

    async def run[T](self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        await self._slots.acquire()
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))

        def finished(completed: asyncio.Task[T]) -> None:
            self._slots.release()
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
        # Отмена ожидания не останавливает нативный inference в потоке.
        # Место освободится только после его фактического завершения.
        return await asyncio.shield(task)
