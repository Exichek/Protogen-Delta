"""Создание, отмена и фоновая доставка явно запрошенных напоминаний."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from time import time
from weakref import WeakValueDictionary

from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.types import LinkPreviewOptions

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.operation_metrics import OperationMetrics, OperationSnapshot
from protogen_delta.core.reminder_request import ReminderRequest
from protogen_delta.repositories.reminders import Reminder, RemindersRepository

logger = logging.getLogger(__name__)


class ReminderService:
    def __init__(
        self,
        bot: Bot,
        repository: RemindersRepository,
        *,
        clock: Callable[[], float] = time,
        interval: float = 5,
    ) -> None:
        if not 0 < interval <= 60:
            raise ValueError("Интервал должен быть от 0 до 60 секунд")
        self._bot, self._repository, self._clock, self._interval = (
            bot,
            repository,
            clock,
            interval,
        )
        self._owners: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()
        self._dispatch_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._metrics = OperationMetrics(capacity=1)

    def snapshot(self) -> OperationSnapshot:
        return self._metrics.snapshot()

    @asynccontextmanager
    async def _owner(self, user_id: int) -> AsyncIterator[None]:
        lock = self._owners.setdefault(user_id, asyncio.Lock())
        async with lock:
            yield

    async def create(
        self,
        user_id: int,
        chat_id: int,
        message_id: int,
        thread_id: int | None,
        request: ReminderRequest,
    ) -> Reminder:
        async with self._owner(user_id):
            return await self._repository.create(
                user_id,
                chat_id,
                message_id,
                thread_id,
                request.text,
                request.due_at,
                self._clock(),
            )

    async def list_user(self, user_id: int) -> list[Reminder]:
        return await self._repository.list_user(user_id)

    async def cancel(self, user_id: int, reminder_id: int) -> bool:
        async with self._owner(user_id):
            return await self._repository.cancel(user_id, reminder_id, self._clock())

    async def delete_user(self, user_id: int) -> None:
        async with self._owner(user_id):
            await self._repository.delete_user(user_id)

    async def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.deliver_due()
            except Exception:
                # Не выводить текст напоминания, адреса и исключения Telegram.
                logger.warning("Очередь напоминаний временно недоступна")
            try:
                await asyncio.wait_for(self._stop.wait(), self._interval)
            except TimeoutError:
                pass

    async def deliver_due(self) -> None:
        with self._metrics.measure(queued=True) as measurement:
            async with self._dispatch_lock:
                measurement.start()
                jobs = await self._repository.claim_due(self._clock())
                unfinished = list(jobs)
                delivery_failed = False
                try:
                    for job in jobs:
                        async with self._owner(job.user_id):
                            if await self._repository.renew_claim(job, self._clock()):
                                delivery_failed = (
                                    not await self._deliver(job) or delivery_failed
                                )
                        unfinished.remove(job)
                    if delivery_failed:
                        measurement.finish(RuntimeError("Напоминание не доставлено"))
                finally:
                    # Отмена запуска/остановки не теряет задания, уже взятые из базы.
                    for job in unfinished:
                        await finish_operation(
                            self._repository.finish(
                                job, self._clock(), retry_at=self._clock() + 30
                            )
                        )

    async def _deliver(self, job: Reminder) -> bool:
        try:
            await asyncio.wait_for(
                self._bot.send_message(
                    chat_id=job.chat_id,
                    message_thread_id=job.thread_id,
                    text=f"⏰ Напоминание #{job.id}\n{job.text}",
                    parse_mode=None,
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                ),
                timeout=20,
            )
        except TelegramRetryAfter as error:
            await self._repository.finish(
                job,
                self._clock(),
                retry_at=self._clock() + max(1, min(error.retry_after, 3600)),
            )
            logger.info("Доставка напоминания отложена из-за ограничения Telegram")
        except TelegramForbiddenError, TelegramBadRequest:
            await self._repository.finish(job, self._clock(), failed=True)
            logger.info("Доставка напоминания недоступна")
        except Exception:
            await self._repository.finish(
                job,
                self._clock(),
                retry_at=self._clock() + min(3600, 15 * 2**job.attempts),
            )
            logger.info("Доставка напоминания будет повторена")
        else:
            await self._repository.finish(job, self._clock())
            return True
        return False
