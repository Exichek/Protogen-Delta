"""Персональная квота: сохранение расходов, атомарность и владение в Mini App."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data

import protogen_delta.config.settings as settings_module
from protogen_delta.core.llm_budget import LlmDailyLimitError
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.llm_usage import LlmReservation, LlmUsageRepository


@pytest.mark.parametrize(
    "value",
    [
        "[]",
        "null",
        "true",
        "{",
        '{"0":1000}',
        '{"-42":1000}',
        '{"042":1000}',
        '{"42":true}',
        '{"42":0}',
        '{"42":-1}',
        '{"42":1.5}',
        '{"42":"1000"}',
        '{"42":[]}',
        '{"42":1000,"42":2000}',
        '{"9223372036854775808":1000}',
        '{"42":9223372036854775808}',
        '{"secret":1000}',
        '{"４２":1000}',
    ],
)
def test_invalid_override_configuration_fails_without_disclosing_values(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("TELEGRAM_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ART_CHAT_ID", "-1")
    monkeypatch.setenv("LLM_DAILY_REQUEST_LIMIT_OVERRIDES", value)
    with pytest.raises(ValueError, match="LLM_DAILY_REQUEST_LIMIT_OVERRIDES") as error:
        settings_module.load_settings()
    assert value not in str(error.value)
    assert "secret" not in str(error.value)


def test_override_settings_preserve_default_and_are_hidden_in_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("TELEGRAM_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ART_CHAT_ID", "-1")
    monkeypatch.setenv("LLM_DAILY_REQUEST_LIMIT", "300")
    monkeypatch.setenv("LLM_DAILY_REQUEST_LIMIT_OVERRIDES", '{"987654321":1000}')
    settings = settings_module.load_settings()
    assert settings.llm_daily_request_limit == 300
    assert settings.llm_daily_request_limit_overrides == {987654321: 1000}
    assert "987654321" not in repr(settings)
    monkeypatch.delenv("LLM_DAILY_REQUEST_LIMIT_OVERRIDES")
    assert settings_module.load_settings().llm_daily_request_limit_overrides == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {0: 1000},
        {-42: 1000},
        {True: 1000},
        {42: True},
        {42: 0},
        {42: 2**63},
        {2**63: 1000},
    ],
)
def test_repository_rejects_invalid_override_before_creating_database(
    tmp_path: Path,
    overrides: dict[Any, Any],
) -> None:
    with pytest.raises(ValueError, match="Персональные"):
        LlmUsageRepository(tmp_path, limit_overrides=overrides)
    assert not (tmp_path / "llm_usage.db").exists()


def test_upgrade_to_1000_preserves_existing_daily_expense_and_tokens(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = LlmUsageRepository(tmp_path)
        receipt = await repo.reserve(42)
        await repo.record_usage(receipt, 123, 45)
        with closing(sqlite3.connect(tmp_path / "llm_usage.db")) as db, db:
            db.execute("UPDATE llm_daily SET requests=300 WHERE user_id=42")
        before = await repo.get(42)
        with pytest.raises(LlmDailyLimitError):
            await repo.reserve(42)
        overrides = {42: 1000}
        upgraded = LlmUsageRepository(tmp_path, limit_overrides=overrides)
        overrides[42] = 9999
        after = await upgraded.get(42)
        assert after == before | {"limit": 1000, "remaining": 700}
        assert (await upgraded.get(7))["limit"] == 300
        await upgraded.reserve(42)
        restored = LlmUsageRepository(tmp_path, limit_overrides={42: 1000})
        assert (await restored.get(42))["remaining"] == 699
        assert (await restored.get(42))["prompt_tokens"] == 123
        assert (await restored.get(42))["completion_tokens"] == 45
        # Возврат прежней квоты не сбрасывает расход и не открывает новый слот.
        downgraded = LlmUsageRepository(tmp_path)
        with pytest.raises(LlmDailyLimitError):
            await downgraded.reserve(42)
        assert (await downgraded.get(42))["requests"] == 301

    asyncio.run(run())


def test_concurrent_personal_and_default_quotas_remain_separate(tmp_path: Path) -> None:
    async def run() -> None:
        first = LlmUsageRepository(tmp_path, 3, limit_overrides={42: 6})
        second = LlmUsageRepository(tmp_path, 3, limit_overrides={42: 6})
        results = await asyncio.gather(
            *(
                (first if i % 2 else second).reserve(owner)
                for owner in (42, 7)
                for i in range(16)
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(r, LlmReservation) for r in results) == 9
        assert sum(isinstance(r, LlmDailyLimitError) for r in results) == 23
        assert (await first.get(42))["requests"] == 6
        assert (await second.get(7))["requests"] == 3

    asyncio.run(run())


def test_personal_quota_rolls_over_at_moscow_midnight(tmp_path: Path) -> None:
    async def run() -> None:
        now = [datetime(2026, 10, 15, 20, 59, 59, tzinfo=timezone.utc).timestamp()]
        repo = LlmUsageRepository(
            tmp_path, 1, limit_overrides={42: 2}, clock=lambda: now[0]
        )
        await repo.reserve(42)
        await repo.reserve(42)
        with pytest.raises(LlmDailyLimitError):
            await repo.reserve(42)
        now[0] += 2
        assert (await repo.get(42))["remaining"] == 2
        assert (await repo.get(7))["remaining"] == 1
        await repo.reserve(42)

    asyncio.run(run())


def test_signed_usage_exposes_only_effective_quota_of_authenticated_user(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = LlmUsageRepository(tmp_path, 300, limit_overrides={42: 1000})
        server = MiniAppServer(TOKEN, UserStateStore(), llm_usage=repo)
        async with TestClient(TestServer(server.application())) as client:
            for owner, expected in ((42, 1000), (7, 300)):
                response = await client.get(
                    "/api/usage?user_id=42",
                    headers={"X-Telegram-Init-Data": _signed_init_data(user_id=owner)},
                )
                assert response.status == 200
                assert (await response.json())["limit"] == expected
            assert (
                await client.post(
                    "/api/usage",
                    headers={"X-Telegram-Init-Data": _signed_init_data(user_id=7)},
                    json={"user_id": 7, "limit": 1000},
                )
            ).status == 405

    asyncio.run(run())
