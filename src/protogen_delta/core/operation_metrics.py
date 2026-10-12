"""Ограниченные метрики операций без сообщений, идентификаторов и секретов."""

import asyncio
import math
from collections import deque
from collections.abc import Callable
from time import monotonic
from types import TracebackType

OperationSnapshot = dict[str, int | float | str | None]


class OperationMetrics:
    def __init__(
        self,
        *,
        capacity: int | None = None,
        sample_limit: int = 128,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if sample_limit < 1 or capacity is not None and capacity < 1:
            raise ValueError("Лимиты метрик должны быть положительными")
        self._capacity, self._clock = capacity, clock
        self._samples: deque[tuple[float, float]] = deque(maxlen=sample_limit)
        self._active = self._waiting = self._completed = 0
        self._failed = self._cancelled = 0
        self._last_result = "unknown"
        self._last_finished: float | None = None
        self._last_success: float | None = None

    def measure(self, *, queued: bool = False) -> "Measurement":
        return Measurement(self, queued)

    def snapshot(self) -> OperationSnapshot:
        def percentile(index: int, fraction: float) -> float | None:
            values = sorted(sample[index] for sample in self._samples)
            return (
                round(values[max(0, math.ceil(len(values) * fraction) - 1)] * 1000, 1)
                if values
                else None
            )

        now = self._clock()
        return {
            "capacity": self._capacity,
            "active": self._active,
            "waiting": self._waiting,
            "completed": self._completed,
            "failed": self._failed,
            "cancelled": self._cancelled,
            "recent_samples": len(self._samples),
            "recent_p50_ms": percentile(0, 0.5),
            "recent_p95_ms": percentile(0, 0.95),
            "queue_p95_ms": percentile(1, 0.95),
            "last_result": self._last_result,
            "last_finished_age_seconds": (
                round(max(0, now - self._last_finished), 1)
                if self._last_finished is not None
                else None
            ),
            "last_success_age_seconds": (
                round(max(0, now - self._last_success), 1)
                if self._last_success is not None
                else None
            ),
        }


class Measurement:
    def __init__(self, metrics: OperationMetrics, queued: bool) -> None:
        self._metrics = metrics
        self._created = metrics._clock()
        self._started: float | None = None
        self._finished = False
        if queued:
            metrics._waiting += 1
        else:
            self._started = self._created
            metrics._active += 1

    def start(self) -> None:
        if self._started is None and not self._finished:
            self._metrics._waiting -= 1
            self._metrics._active += 1
            self._started = self._metrics._clock()

    def __enter__(self) -> "Measurement":
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish(error)

    def finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        metrics = self._metrics
        now = metrics._clock()
        if self._started is None:
            metrics._waiting -= 1
        else:
            metrics._active -= 1
        result = "ok"
        if isinstance(error, asyncio.CancelledError):
            metrics._cancelled += 1
            result = "cancelled"
        elif error is not None:
            metrics._failed += 1
            result = "failed"
        else:
            metrics._last_success = now
        metrics._completed += 1
        metrics._last_result = result
        metrics._last_finished = now
        metrics._samples.append(
            (
                max(0, now - self._created),
                max(
                    0,
                    (self._started if self._started is not None else now)
                    - self._created,
                ),
            )
        )
