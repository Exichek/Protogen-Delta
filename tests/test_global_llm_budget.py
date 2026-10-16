"""Суммарный потолок применяется ко всем аккаунтам и API-попыткам."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from aiogram import Bot
from openai import AsyncOpenAI

from protogen_delta.config.settings import load_settings
from protogen_delta.core.llm_budget import LlmBudgetError, LlmDailyLimitError
from protogen_delta.repositories.llm_usage import LlmReservation, LlmUsageRepository
from protogen_delta.services.llm_accounting import accounted_completion
from protogen_delta.services.llm_budget_alerts import LlmBudgetAlerts


def test_global_limit_serializes_many_owners_and_preserves_personal_budget(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        first = LlmUsageRepository(
            tmp_path, 3, global_daily_limit=5, limit_overrides={7: 1000}
        )
        second = LlmUsageRepository(
            tmp_path, 3, global_daily_limit=5, limit_overrides={7: 1000}
        )
        receipts = await asyncio.gather(
            *((first if n % 2 else second).reserve(n % 4 + 7) for n in range(40)),
            return_exceptions=True,
        )
        assert sum(isinstance(r, LlmReservation) for r in receipts) == 5
        assert all(isinstance(r, (LlmReservation, LlmBudgetError)) for r in receipts)
        assert (await first.get_global())["requests"] == 5
        assert (await first.get_global())["remaining"] == 0
        assert first.limit_for(7) == 1000
        for owner in range(7, 11):
            assert int((await first.get(owner))["requests"]) <= first.limit_for(owner)
        with pytest.raises(LlmBudgetError, match="Общий дневной"):
            await LlmUsageRepository(tmp_path, global_daily_limit=5).reserve(77)
        assert (await first.get(77))["requests"] == 0

    asyncio.run(run())


def test_unowned_retries_cannot_escape_global_cap(tmp_path: Path) -> None:
    async def run() -> None:
        attempts = []

        def http(request: httpx.Request) -> httpx.Response:
            attempts.append(request)
            return httpx.Response(
                500, json={"error": {"message": "synthetic", "type": "server_error"}}
            )

        client = AsyncOpenAI(
            api_key="synthetic",
            base_url="https://example.invalid",
            max_retries=2,
            http_client=cast(
                Any, httpx.AsyncClient(transport=httpx.MockTransport(http))
            ),
        )
        repo = LlmUsageRepository(tmp_path, global_daily_limit=1)
        try:
            with pytest.raises(LlmBudgetError, match="Общий дневной"):
                await accounted_completion(
                    client,
                    repo,
                    model="test",
                    messages=[{"role": "user", "content": "synthetic"}],
                )
            assert len(attempts) == 1
            assert (await repo.get_global())["requests"] == 1
            assert (await repo.get(0))["requests"] == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_alert_thresholds_are_durable_and_no_owner_does_not_consume_them(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [datetime(2026, 10, 15, 20, 59, 59, tzinfo=timezone.utc).timestamp()]
        repo = LlmUsageRepository(
            tmp_path, 10, global_daily_limit=5, clock=lambda: now[0]
        )
        bot = AsyncMock(spec=Bot)
        disabled = LlmBudgetAlerts(bot, repo, None)
        alerts = LlmBudgetAlerts(bot, repo, 7)
        await alerts.run_once()
        bot.send_message.assert_not_awaited()
        for _ in range(4):
            await repo.reserve(7)
        await disabled.run_once()
        await asyncio.gather(
            alerts.run_once(), LlmBudgetAlerts(bot, repo, 7).run_once()
        )
        assert bot.send_message.await_count == 1
        assert "80%" in bot.send_message.await_args.args[1]
        await LlmBudgetAlerts(
            bot,
            LlmUsageRepository(
                tmp_path, 10, global_daily_limit=5, clock=lambda: now[0]
            ),
            7,
        ).run_once()
        assert bot.send_message.await_count == 1
        await repo.reserve(7)
        await alerts.run_once()
        assert bot.send_message.await_count == 2
        assert "100%" in bot.send_message.await_args.args[1]
        now[0] += 2
        await alerts.run_once()
        assert bot.send_message.await_count == 2
        for _ in range(4):
            await repo.reserve(7)
        await alerts.run_once()
        assert bot.send_message.await_count == 3

    asyncio.run(run())


def test_unknown_alert_delivery_is_not_retried_or_duplicated(tmp_path: Path) -> None:
    async def run() -> None:
        repo = LlmUsageRepository(tmp_path, 10, global_daily_limit=1)
        await repo.reserve(7)
        bot = AsyncMock(spec=Bot)
        bot.send_message.side_effect = TimeoutError("synthetic")
        alerts = LlmBudgetAlerts(bot, repo, 7)
        with pytest.raises(TimeoutError):
            await alerts.run_once()
        await alerts.run_once()
        bot.send_message.assert_awaited_once()
        assert "100%" in bot.send_message.await_args.args[1]

    asyncio.run(run())


def test_alert_worker_starts_and_closes_without_duplicate_task(tmp_path: Path) -> None:
    async def run() -> None:
        alerts = LlmBudgetAlerts(AsyncMock(spec=Bot), LlmUsageRepository(tmp_path), 7)
        alerts.start()
        first = alerts._task
        alerts.start()
        assert alerts._task is first
        await alerts.close()
        assert alerts._task is None
        await alerts.close()

    asyncio.run(run())


def test_monitor_survives_repository_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        alerts = LlmBudgetAlerts(AsyncMock(spec=Bot), LlmUsageRepository(tmp_path), 7)
        monkeypatch.setattr(
            alerts,
            "run_once",
            AsyncMock(side_effect=sqlite3.OperationalError("synthetic")),
        )
        sleep = AsyncMock(side_effect=asyncio.CancelledError)
        monkeypatch.setattr(asyncio, "sleep", sleep)
        with pytest.raises(asyncio.CancelledError):
            await alerts._run()
        sleep.assert_awaited_once_with(60)

    asyncio.run(run())


@pytest.mark.parametrize(
    "name,value",
    [
        ("LLM_GLOBAL_DAILY_REQUEST_LIMIT", "0"),
        ("LLM_GLOBAL_DAILY_REQUEST_LIMIT", "-1"),
        ("LLM_GLOBAL_DAILY_REQUEST_LIMIT", "bad"),
        ("LLM_BUDGET_ALERT_USER_ID", "bad"),
    ],
)
def test_invalid_global_configuration_fails_before_start(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "synthetic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic")
    monkeypatch.setenv("ART_CHAT_ID", "-100")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_settings()


def test_personal_refusal_does_not_spend_global_quota(tmp_path: Path) -> None:
    async def run() -> None:
        repo = LlmUsageRepository(tmp_path, 1, global_daily_limit=5)
        await repo.reserve(7)
        with pytest.raises(LlmDailyLimitError):
            await repo.reserve(7)
        assert (await repo.get_global())["requests"] == 1
        await repo.reserve(8)
        assert (await repo.get_global())["requests"] == 2

    asyncio.run(run())


def test_existing_usage_counts_towards_new_limit_and_rolls_over_in_moscow(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [datetime(2026, 10, 15, 20, 59, 59, tzinfo=timezone.utc).timestamp()]
        repo = LlmUsageRepository(tmp_path, 10, clock=lambda: now[0])
        receipt = await repo.reserve(7)
        await repo.reserve(8)
        capped = LlmUsageRepository(
            tmp_path, 10, global_daily_limit=2, clock=lambda: now[0]
        )
        with pytest.raises(LlmBudgetError, match="Общий дневной"):
            await capped.reserve(9)
        now[0] += 2
        assert (await capped.get_global())["requests"] == 0
        await capped.record_usage(receipt, 11, 7)
        assert (await capped.get_global())["prompt_tokens"] == 0
        await capped.reserve(9)
        with closing(sqlite3.connect(tmp_path / "llm_usage.db")) as db:
            assert db.execute(
                "SELECT SUM(requests) FROM llm_daily WHERE day=?", ("2026-10-15",)
            ).fetchone() == (2,)

    asyncio.run(run())
