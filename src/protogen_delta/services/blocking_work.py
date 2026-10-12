"""Ограничить реально работающие потоки, включая отменённые запросы."""

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from protogen_delta.core.operation_metrics import OperationMetrics, OperationSnapshot


class WorkRunner(Protocol):
    async def run[T](
        self, function: Callable[..., T], data: bytes, *args: Any, **kwargs: Any
    ) -> T: ...


class BlockingWorkPool:
    def __init__(self, limit: int = 1) -> None:
        if limit < 1:
            raise ValueError("limit должен быть положительным")
        self._slots = asyncio.Semaphore(limit)
        self._metrics = OperationMetrics(capacity=limit)

    def snapshot(self) -> OperationSnapshot:
        return self._metrics.snapshot()

    async def run[T](self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        measurement = self._metrics.measure(queued=True)
        try:
            await self._slots.acquire()
        except BaseException as error:
            measurement.finish(error)
            raise
        measurement.start()

        async def execute() -> T:
            with measurement:
                return await asyncio.to_thread(function, *args, **kwargs)

        task = asyncio.create_task(execute())

        def finished(completed: asyncio.Task[T]) -> None:
            self._slots.release()
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
        # Отмена ожидания не останавливает нативный inference в потоке.
        # Место освободится только после его фактического завершения.
        return await asyncio.shield(task)
