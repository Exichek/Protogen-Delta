"""Состояние polling без содержимого сообщений и личных идентификаторов."""

import math
from collections.abc import Callable
from time import monotonic
from typing import Any

from aiogram import Bot
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.methods import GetUpdates, Response, TelegramMethod
from aiogram.methods.base import TelegramType

from protogen_delta.core.operation_metrics import OperationSnapshot


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
        self._diagnostics: dict[str, Callable[[], OperationSnapshot]] = {}

    def add_diagnostics(
        self, name: str, snapshot: Callable[[], OperationSnapshot]
    ) -> None:
        if name not in {"llm", "audio", "native", "whisper"}:
            raise ValueError("Неизвестная подсистема диагностики")
        self._diagnostics[name] = snapshot

    def polling_succeeded(self) -> None:
        self._last_poll = self._clock()

    def polling_failed(self) -> None:
        self._errors += 1

    def stop(self) -> None:
        self._running = False

    def snapshot(self) -> dict[str, Any]:
        age = (
            max(0.0, self._clock() - self._last_poll)
            if self._last_poll is not None
            else None
        )
        ready = self._running and age is not None and age < self._stale_seconds
        result: dict[str, Any] = {
            "ok": ready,
            "polling": ready,
            "polling_age_seconds": round(age, 1) if age is not None else None,
            "polling_errors": self._errors,
        }
        if self._diagnostics:
            result["diagnostics"] = {
                name: snapshot() for name, snapshot in self._diagnostics.items()
            }
        return result


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
