"""Уведомлять владельца о суммарном потолке без вызовов модели и текста переписки."""

import asyncio
import logging
from contextlib import suppress

from aiogram import Bot

from protogen_delta.repositories.llm_usage import LlmUsageRepository

logger = logging.getLogger(__name__)


class LlmBudgetAlerts:
    def __init__(
        self, bot: Bot, repository: LlmUsageRepository, owner: int | None
    ) -> None:
        self._bot, self._repository, self._owner = bot, repository, owner
        self._task: asyncio.Task[None] | None = None

    async def run_once(self) -> None:
        if self._owner is None:
            return
        snapshot = await self._repository.claim_global_alert()
        if snapshot is None:
            return
        # После сетевой ошибки исход доставки может быть неизвестен: не дублируем сигнал.
        await self._bot.send_message(
            self._owner,
            f"Расход API Дельты: {snapshot['requests']} из {snapshot['limit']} попыток "
            f"за {snapshot['day']} (Москва), достигнут порог {snapshot['threshold']}%. "
            f"Осталось: {snapshot['remaining']}. "
            "Общий потолок обновится в 00:00 по Москве; личные квоты действуют отдельно.",
            parse_mode=None,
        )

    async def _run(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.warning(
                    "Не удалось проверить или доставить уведомление общего лимита API"
                )
            await asyncio.sleep(60)

    def start(self) -> None:
        if self._owner is not None and self._task is None:
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
