"""Лимит API, токены, retries, Москва и проверенное владение расходами."""

import asyncio
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from aiogram.types import Message
from aiohttp.test_utils import TestClient, TestServer
from openai import AsyncOpenAI
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

import protogen_delta.services.deepseek as deepseek_module
from protogen_delta.config.settings import load_settings
from protogen_delta.core.llm_budget import (
    LlmDailyLimitError,
    LlmUsageUnavailableError,
    current_llm_user,
    llm_user,
)
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.llm_budget import LlmUserMiddleware
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.miniapp.tools import MiniAppTools
from protogen_delta.repositories.llm_usage import LlmReservation, LlmUsageRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.audio_understanding import AudioUnderstandingService
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.llm_accounting import record_response_usage
from protogen_delta.services.translation import TranslationService


def test_concurrent_reservations_and_restart_cannot_exceed_limit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first, second = LlmUsageRepository(tmp_path, 3), LlmUsageRepository(tmp_path, 3)
        results = await asyncio.gather(
            *((first if i % 2 else second).reserve(42) for i in range(20)),
            return_exceptions=True,
        )
        assert sum(isinstance(value, LlmReservation) for value in results) == 3
        assert sum(isinstance(value, LlmDailyLimitError) for value in results) == 17
        restored = LlmUsageRepository(tmp_path, 3)
        assert (await restored.get(42))["requests"] == 3
        with pytest.raises(LlmDailyLimitError):
            await restored.reserve(42)
        await restored.reserve(7)
        assert (await restored.get(7))["requests"] == 1

    asyncio.run(scenario())


def test_midnight_moscow_and_late_usage_are_accounted_for_original_day(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 10, 13, 20, 59, 59, tzinfo=timezone.utc).timestamp()]
        repo = LlmUsageRepository(tmp_path, 1, clock=lambda: now[0])
        receipt = await repo.reserve(42)
        assert (await repo.get(42))["day"] == "2026-10-13"
        now[0] += 2
        assert (await repo.get(42))["remaining"] == 1
        await repo.record_usage(receipt, 11, 7)
        await repo.record_usage(receipt, 11, 7)
        assert (await repo.get(42))["prompt_tokens"] == 0
        with closing(sqlite3.connect(tmp_path / "llm_usage.db")) as db:
            assert db.execute(
                "SELECT requests,prompt_tokens,completion_tokens,usage_missing FROM llm_daily WHERE day='2026-10-13'"
            ).fetchone() == (1, 11, 7, 0)
        await repo.reserve(42)

    asyncio.run(scenario())


def test_reset_and_user_removal_do_not_reset_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = LlmUsageRepository(tmp_path, 1)
        await repo.reserve(42)
        UsersRepository(tmp_path).add(42)
        UsersRepository(tmp_path).remove(42)
        await UserStateStore().reset_user(42)
        with pytest.raises(LlmDailyLimitError):
            await LlmUsageRepository(tmp_path, 1).reserve(42)

    asyncio.run(scenario())


def test_unavailable_usage_storage_fails_closed(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = LlmUsageRepository(tmp_path)
        (tmp_path / "llm_usage.db").unlink()
        with pytest.raises(LlmUsageUnavailableError):
            await repo.reserve(42)

    asyncio.run(scenario())


def _sdk_model(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    retries: int = 0,
) -> DeepSeekService:
    client = AsyncOpenAI(
        api_key="synthetic",
        base_url="https://synthetic.invalid",
        max_retries=retries,
        http_client=cast(
            Any, httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    monkeypatch.setattr(deepseek_module, "AsyncOpenAI", lambda **kwargs: client)
    return DeepSeekService("synthetic", "https://synthetic.invalid", "synthetic-model")


def _response(text: str = "Hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "synthetic",
            "object": "chat.completion",
            "created": 1,
            "model": "synthetic-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    )


@pytest.mark.parametrize("limit,expected_requests", [(1, 1), (3, 2)])
def test_real_sdk_retry_attempts_each_consume_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int, expected_requests: int
) -> None:
    async def scenario() -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return (
                httpx.Response(
                    429,
                    headers={"retry-after": "0.001"},
                    json={"error": {"message": "synthetic", "type": "rate_limit"}},
                )
                if len(calls) == 1
                else _response()
            )

        model = _sdk_model(monkeypatch, handler, retries=1)
        repo = LlmUsageRepository(tmp_path, limit)
        model.set_usage_repository(repo)
        try:
            with llm_user(42):
                if limit == 1:
                    with pytest.raises(LlmDailyLimitError):
                        await model.chat("system", "user")
                else:
                    assert await model.chat("system", "user") == "Hello"
            assert len(calls) == expected_requests
            data = await repo.get(42)
            assert data["requests"] == expected_requests
            assert data["prompt_tokens"] == (11 if limit > 1 else 0)
            assert data["usage_missing"] == 1
        finally:
            await model.close()

    asyncio.run(scenario())


def test_chat_translation_share_quota_and_api_never_trusts_body_user_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return _response(
                json.dumps({"translated_text": "Привет", "detected_language": "en"})
            )

        model = _sdk_model(monkeypatch, handler)
        repo = LlmUsageRepository(tmp_path, 1)
        model.set_usage_repository(repo)
        with llm_user(42):
            await model.chat("system", "user")
        tools = MiniAppTools(
            AsyncMock(), Mock(), translator=TranslationService(model, Mock())
        )
        server = MiniAppServer(TOKEN, UserStateStore(), tools=tools, llm_usage=repo)
        try:
            async with TestClient(TestServer(server.application())) as client:
                assert (await client.get("/api/usage")).status == 401
                headers = {"X-Telegram-Init-Data": _signed_init_data()}
                response = await client.post(
                    "/api/tools/translate",
                    headers=headers,
                    json={
                        "text": "Hello",
                        "source": "en",
                        "target": "ru",
                        "user_id": 7,
                    },
                )
                assert (
                    response.status == 429 and "Дневной лимит" in await response.text()
                )
                usage = await client.get("/api/usage?user_id=7", headers=headers)
                assert (await usage.json())["remaining"] == 0
                assert (await repo.get(7))["requests"] == 0
                assert (
                    await client.patch(
                        "/api/profile", headers=headers, json={"roleplay_active": False}
                    )
                ).status == 200
            assert len(calls) == 1
            assert current_llm_user.get() is None
        finally:
            await model.close()

    asyncio.run(scenario())


def test_engine_returns_limit_reply_and_safety_still_precedes_model() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        model.chat.side_effect = LlmDailyLimitError()
        reply = await engine.respond(42, "Привет")
        assert "Дневной лимит" in reply
        assert current_llm_user.get() is None
        model.chat.reset_mock()
        assert "взрослый режим выключен" in await engine.respond(
            42, "Привет, мне 15 лет"
        )
        model.chat.assert_not_awaited()

    asyncio.run(scenario())


def test_telegram_middleware_scopes_owner_and_allows_unpaid_commands() -> None:
    async def scenario() -> None:
        event = Mock(spec=Message)
        event.from_user = SimpleNamespace(id=42, is_bot=False)
        event.answer = AsyncMock()

        async def paid(event: object, data: object) -> None:
            assert current_llm_user.get() == 42
            raise LlmDailyLimitError()

        await LlmUserMiddleware()(paid, event, {})
        assert "Дневной лимит" in event.answer.await_args.args[0]
        assert current_llm_user.get() is None
        handler = AsyncMock(return_value="ordinary command")
        assert await LlmUserMiddleware()(handler, event, {}) == "ordinary command"

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_invalid_daily_limit_cannot_start_bot(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "synthetic")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic")
    monkeypatch.setenv("ART_CHAT_ID", "-100")
    monkeypatch.setenv("LLM_DAILY_REQUEST_LIMIT", value)
    with pytest.raises(ValueError, match="LLM_DAILY_REQUEST_LIMIT"):
        load_settings()


def test_unknown_usage_and_failed_accounting_keep_attempt_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        repo = LlmUsageRepository(tmp_path)
        first = await repo.reserve(42)
        await record_response_usage(repo, first, SimpleNamespace(usage=None))
        assert (await repo.get(42))["usage_missing"] == 1
        second = await repo.reserve(42)
        monkeypatch.setattr(
            repo,
            "record_usage",
            AsyncMock(side_effect=sqlite3.OperationalError("synthetic")),
        )
        await record_response_usage(
            repo,
            second,
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2)
            ),
        )
        assert (await repo.get(42))["requests"] == 2
        assert (await repo.get(42))["usage_missing"] == 2

    asyncio.run(scenario())


def test_cancellation_restores_owner_context() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        started = asyncio.Event()

        async def chat(**kwargs: object) -> str:
            assert current_llm_user.get() == 42
            started.set()
            await asyncio.Event().wait()
            return "unreachable"

        model.chat.side_effect = chat
        task = asyncio.create_task(engine.respond(42, "Привет"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert current_llm_user.get() is None

    asyncio.run(scenario())


def test_streaming_audio_shares_budget_and_records_final_usage(tmp_path: Path) -> None:
    async def scenario() -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            base = {
                "id": "synthetic",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "synthetic",
            }
            chunks = [
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Описание звука"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **base,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                },
            ]
            data = (
                "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, text=data
            )

        client = AsyncOpenAI(
            api_key="synthetic",
            base_url="https://synthetic.invalid",
            max_retries=0,
            http_client=cast(
                Any, httpx.AsyncClient(transport=httpx.MockTransport(handler))
            ),
        )
        native = Mock()
        native.run = AsyncMock(return_value=b"synthetic-wav")
        audio = AudioUnderstandingService(client, "synthetic", native_work=native)
        repo = LlmUsageRepository(tmp_path, 1)
        audio.set_usage_repository(repo)
        try:
            with llm_user(42):
                assert await audio.analyze(b"synthetic") == "Описание звука"
                with pytest.raises(LlmDailyLimitError):
                    await audio.analyze(b"synthetic")
            usage = await repo.get(42)
            assert (
                usage["requests"] == 1
                and usage["prompt_tokens"] == 11
                and usage["completion_tokens"] == 7
            )
            assert usage["usage_missing"] == 0 and len(calls) == 1
        finally:
            await audio.close()

    asyncio.run(scenario())
