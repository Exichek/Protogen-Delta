"""Состояние polling без содержимого сообщений и личных идентификаторов."""

import math
from collections.abc import Callable
from time import monotonic

from aiogram import Bot
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.methods import GetUpdates, Response, TelegramMethod
from aiogram.methods.base import TelegramType


class RuntimeHealth:
    def __init__(
        self, *, stale_seconds: float = 90, clock: Callable[[], float] = monotonic
    ) -> None:
        if not math.isfinite(stale_seconds) or stale_seconds <= 0:
            raise ValueError("stale_seconds должен быть положительным и конечным")
        self._clock, self._stale_seconds = clock, stale_seconds
        self._last_poll: float | None = None
        self._running = True
        self._errors = 0

    def polling_succeeded(self) -> None:
        self._last_poll = self._clock()

    def polling_failed(self) -> None:
        self._errors += 1

    def stop(self) -> None:
        self._running = False

    def snapshot(self) -> dict[str, bool | float | int | None]:
        age = (
            max(0.0, self._clock() - self._last_poll)
            if self._last_poll is not None
            else None
        )
        ready = self._running and age is not None and age < self._stale_seconds
        return {
            "ok": ready,
            "polling": ready,
            "polling_age_seconds": round(age, 1) if age is not None else None,
            "polling_errors": self._errors,
        }


class PollingHealthMiddleware(BaseRequestMiddleware):
    def __init__(self, health: RuntimeHealth) -> None:
        self._health = health

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ) -> Response[TelegramType]:
        try:
            response = await make_request(bot, method)
        except Exception:
            if isinstance(method, GetUpdates):
                self._health.polling_failed()
            raise
        if isinstance(method, GetUpdates):
            self._health.polling_succeeded()
        return response
