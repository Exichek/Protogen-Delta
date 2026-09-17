"""Ограничение частоты пользовательских запросов."""

from collections import OrderedDict
from collections.abc import Callable
from math import isfinite
from time import monotonic


class UserRateLimiter:
    """Ограничивать частоту запросов отдельно для каждого пользователя."""

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        retention_seconds: float = 300.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Настроить cooldown и время хранения неактивных пользователей."""
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds не может быть отрицательным")

        if retention_seconds <= 0:
            raise ValueError("retention_seconds должен быть больше нуля")

        if retention_seconds < cooldown_seconds:
            raise ValueError("retention_seconds не может быть меньше cooldown_seconds")

        if not isfinite(cooldown_seconds):
            raise ValueError("cooldown_seconds должен быть конечным числом")

        if not isfinite(retention_seconds):
            raise ValueError("retention_seconds должен быть конечным числом")

        self._cooldown_seconds = cooldown_seconds
        self._retention_seconds = retention_seconds
        self._clock = clock or monotonic
        self._last_request_at: OrderedDict[int, float] = OrderedDict()

    def allow(self, user_id: int) -> bool:
        """Вернуть True, если пользователь может выполнить новый запрос."""
        now = self._clock()

        self._remove_stale_entries(now)

        previous_request_at = self._last_request_at.get(user_id)

        if (
            previous_request_at is not None
            and now - previous_request_at < self._cooldown_seconds
        ):
            return False

        self._last_request_at[user_id] = now
        self._last_request_at.move_to_end(user_id)

        return True

    @property
    def tracked_users_count(self) -> int:
        """Вернуть количество пользователей, хранящихся в limiter."""
        return len(self._last_request_at)

    def _remove_stale_entries(self, now: float) -> None:
        """Удалить пользователей, которые давно не отправляли запросы."""
        while self._last_request_at:
            _, oldest_request_at = next(iter(self._last_request_at.items()))

            if now - oldest_request_at < self._retention_seconds:
                break

            self._last_request_at.popitem(last=False)
