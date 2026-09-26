"""Планирование коротких проактивных сообщений пользователям."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from time import time

from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError

from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProactiveConfig:
    """Ограничения фоновой отправки."""

    check_interval_seconds: float = 300.0
    idle_seconds: float = 86400.0
    cooldown_seconds: float = 172800.0
    quiet_start_hour: int = 23
    quiet_end_hour: int = 9
    batch_size: int = 10


class ProactiveMessenger:
    """Иногда начинать разговор с неактивными пользователями без спама."""

    def __init__(
        self,
        *,
        bot: Bot,
        deepseek: DeepSeekService,
        repository: MemoriesRepository,
        system_prompt: str,
        config: ProactiveConfig,
        clock: Callable[[], float] = time,
        local_datetime: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._bot = bot
        self._deepseek = deepseek
        self._repository = repository
        self._system_prompt = system_prompt
        self._config = config
        self._clock = clock
        self._local_datetime = local_datetime
        self._stop = asyncio.Event()

    async def run_once(self) -> int:
        """Выполнить один проход планировщика и вернуть число отправлений."""
        if self._is_quiet_hour(self._local_datetime().hour):
            return 0
        now = self._clock()
        candidates = await self._repository.due_candidates(
            now=now,
            idle_seconds=self._config.idle_seconds,
            cooldown_seconds=self._config.cooldown_seconds,
            limit=self._config.batch_size,
        )
        sent = 0
        for candidate in candidates:
            memories = await self._repository.recent(candidate.user_id, limit=5)
            memory_lines = (
                "\n".join(f"- {item.kind}: {item.text!r}" for item in memories)
                or "- значимых эпизодов пока нет"
            )
            request = (
                "Напиши пользователю одно короткое естественное сообщение, чтобы самому "
                "возобновить общение. Можно сказать, что соскучился, или ненавязчиво "
                "вернуться к подходящей прошлой теме. Не выдумывай события, не упоминай "
                "служебную память и не требуй ответа. Один абзац, до 240 символов.\n\n"
                f"Прошлые эпизоды:\n{memory_lines}"
            )
            try:
                text = (
                    await self._deepseek.chat(
                        system_prompt=self._system_prompt,
                        user_message=request,
                    )
                ).strip()
                if not text:
                    continue
                await self._bot.send_message(candidate.user_id, text[:1000])
            except TelegramForbiddenError:
                logger.info("Пользователь %s заблокировал бота", candidate.user_id)
                await self._repository.set_proactive(candidate.user_id, False, now)
                continue
            except DeepSeekError:
                logger.warning(
                    "Не удалось создать проактивное сообщение", exc_info=True
                )
                continue
            except Exception:
                logger.exception(
                    "Не удалось отправить проактивное сообщение пользователю %s",
                    candidate.user_id,
                )
                continue
            await self._repository.note_proactive_sent(candidate.user_id, now)
            sent += 1
        return sent

    async def run_forever(self) -> None:
        """Запускать проходы до штатной остановки приложения."""
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("Ошибка прохода планировщика проактивных сообщений")
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._config.check_interval_seconds
                )
            except TimeoutError:
                pass

    def stop(self) -> None:
        """Запросить остановку фонового цикла."""
        self._stop.set()

    def _is_quiet_hour(self, hour: int) -> bool:
        start = self._config.quiet_start_hour
        end = self._config.quiet_end_hour
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end
