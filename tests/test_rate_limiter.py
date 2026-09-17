"""Тесты ограничения частоты пользовательских запросов."""

import pytest

from protogen_delta.core.rate_limiter import UserRateLimiter


def test_rate_limiter_allows_first_request() -> None:
    """Первый запрос пользователя должен проходить."""
    limiter = UserRateLimiter(
        clock=lambda: 100.0,
    )

    assert limiter.allow(123) is True


def test_rate_limiter_blocks_request_during_cooldown() -> None:
    """Повторный запрос во время cooldown должен блокироваться."""
    times = iter([100.0, 100.5])

    limiter = UserRateLimiter(
        cooldown_seconds=2.0,
        clock=lambda: next(times),
    )

    assert limiter.allow(123) is True
    assert limiter.allow(123) is False


def test_rate_limiter_allows_request_after_cooldown() -> None:
    """После cooldown пользователь снова может отправить запрос."""
    times = iter([100.0, 102.0])

    limiter = UserRateLimiter(
        cooldown_seconds=2.0,
        clock=lambda: next(times),
    )

    assert limiter.allow(123) is True
    assert limiter.allow(123) is True


def test_rate_limiter_is_independent_for_users() -> None:
    """Cooldown одного пользователя не должен блокировать другого."""
    times = iter([100.0, 100.5])

    limiter = UserRateLimiter(
        cooldown_seconds=2.0,
        clock=lambda: next(times),
    )

    assert limiter.allow(111) is True
    assert limiter.allow(222) is True


def test_rate_limiter_removes_stale_users() -> None:
    """Давно неактивные пользователи должны удаляться из памяти."""
    times = iter([0.0, 10.0, 100.0])

    limiter = UserRateLimiter(
        cooldown_seconds=2.0,
        retention_seconds=30.0,
        clock=lambda: next(times),
    )

    assert limiter.allow(111) is True
    assert limiter.allow(222) is True
    assert limiter.tracked_users_count == 2

    assert limiter.allow(333) is True

    assert limiter.tracked_users_count == 1


@pytest.mark.parametrize(
    ("cooldown", "retention"),
    [
        (-1.0, 300.0),
        (float("nan"), 300.0),
        (float("inf"), 300.0),
        (2.0, 0.0),
        (2.0, float("nan")),
        (2.0, float("inf")),
        (10.0, 5.0),
    ],
)
def test_rate_limiter_rejects_invalid_configuration(
    cooldown: float,
    retention: float,
) -> None:
    """Некорректные настройки limiter должны отклоняться."""
    with pytest.raises(ValueError):
        UserRateLimiter(
            cooldown_seconds=cooldown,
            retention_seconds=retention,
        )
