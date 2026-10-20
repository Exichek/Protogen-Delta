"""Обзор не дублирует платные запросы и сохраняет отчёт до доставки."""

import asyncio
import sys
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from test_action_handlers import message
from test_source_access import source

from protogen_delta.core.llm_budget import current_llm_user
from protogen_delta.handlers.code_review import create_code_review_router
from protogen_delta.repositories.code_reviews import CodeReviewsRepository
from protogen_delta.services.code_review import CodeReviewService, answer_review
from protogen_delta.services.deepseek import DeepSeekConnectionError, DeepSeekService
from protogen_delta.services.source_access import current_source_owner


def test_review_receives_runtime_and_parse_facts_and_cache_is_explicit(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Явных дефектов в просмотренном коде нет."
        service = CodeReviewService(
            AsyncMock(spec=Bot),
            model,
            source(tmp_path),
            CodeReviewsRepository(tmp_path),
            clock=lambda: 10000,
        )
        report = await service.review(10, "services/sample.py")
        version = ".".join(str(part) for part in sys.version_info[:3])
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert f"Python {version}" in prompt
        assert "PEP 758" in prompt and "при наличии as" in prompt
        assert "ast.parse" in prompt and "до маскирования" in prompt
        assert "continue/return" in prompt
        assert model.chat.await_args.kwargs["thinking"] is True
        assert report and f"Среда: Python {version}" in report
        cached = await service.review(10, "services/sample.py")
        assert cached and "Сохранённый предыдущий обзор" in cached
        assert "новая проверка не запускалась" in cached and report in cached
        assert model.chat.await_count == 1

    asyncio.run(run())


def test_review_validates_model_diff_before_cache_and_delivery(tmp_path: Path) -> None:
    async def run() -> None:
        access = source(tmp_path)
        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = (
            "Предложение для обсуждения:\n```diff\n"
            "--- a/src/protogen_delta/services/sample.py\n"
            "+++ b/src/protogen_delta/services/sample.py\n"
            "@@ -1 +1 @@\n-value = 42\n+value = 43\n```\n"
        )
        original = (tmp_path / "services/sample.py").read_text("utf-8")
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(
            AsyncMock(spec=Bot),
            model,
            access,
            repo,
            clock=lambda: 10000,
        )
        report = await service.review(10, "services/sample.py")
        assert report and "Контекст строк совпал" in report
        assert "Код не изменялся" in report and "AST предложения прошёл" in report
        assert (await repo.load(10)).report == report  # type: ignore[union-attr]
        assert (tmp_path / "services/sample.py").read_text("utf-8") == original
        assert "администратор решит" in model.chat.await_args.kwargs["system_prompt"]

    asyncio.run(run())


def test_allowed_command_returns_report_and_long_automatic_notice_keeps_full_cache(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        access = source(tmp_path)
        bot, model = AsyncMock(spec=Bot), AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Замечания по коду. " * 400
        now = [10000.0]
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(bot, model, access, repo, clock=lambda: now[0])
        incoming = message("/code_review services/sample.py")
        router = create_code_review_router(service)
        await router.message.handlers[0].callback(incoming)
        assert incoming.answer.await_count > 1
        state = await repo.load(10)
        assert state and state.notified and len(state.report) > 3500
        (tmp_path / "services/sample.py").write_text("value = 999", "utf-8")
        now[0] += 86400
        await service.run_once()
        assert len(bot.send_message.await_args.args[1]) <= 3500
        assert "/code_review" in bot.send_message.await_args.args[1]
        assert model.chat.await_count == 2
        # Запущенный рабочий цикл при неизменном коде не генерирует ещё один отчёт.
        service.start()
        task = service._task
        assert task is not None
        service.start()
        assert service._task is task
        await asyncio.sleep(0.05)
        await service.close()
        assert service._task is None and model.chat.await_count == 2

    asyncio.run(run())


def test_changed_files_are_not_lost_when_daily_batch_is_limited(tmp_path: Path) -> None:
    async def run() -> None:
        access = source(tmp_path)
        bot, model = AsyncMock(spec=Bot), AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Явных проблем в просмотренных фрагментах нет."
        now = [10000.0]
        service = CodeReviewService(
            bot, model, access, CodeReviewsRepository(tmp_path), clock=lambda: now[0]
        )
        await service.run_once()
        for index in range(5):
            (tmp_path / f"services/new{index}.py").write_text(
                f"value = {index}", "utf-8"
            )
        await service.run_once()
        first = model.chat.await_args.kwargs["user_message"]
        assert first.count("Файл services/new") == 3
        now[0] += 86400
        await service.run_once()
        second = model.chat.await_args.kwargs["user_message"]
        assert second.count("Файл services/new") == 2
        now[0] += 86400
        await service.run_once()
        assert model.chat.await_count == 2

    asyncio.run(run())


def test_baseline_changed_code_daily_limit_restart_and_pending_delivery(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        access = source(tmp_path)
        bot, model = AsyncMock(spec=Bot), AsyncMock(spec=DeepSeekService)
        now = [10000.0]
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(
            cast(Bot, bot), model, access, repo, clock=lambda: now[0]
        )

        async def chat(**args: object) -> str:
            assert current_source_owner.get() == 10 and current_llm_user.get() == 10
            assert args["tool_names"] == {"read_source", "search_source"}
            return "Замечание по services/sample.py:1"

        model.chat.side_effect = chat
        await service.run_once()
        await service.run_once()
        model.chat.assert_not_awaited()
        (tmp_path / "services/sample.py").write_text("value = 43", "utf-8")
        bot.send_message.side_effect = RuntimeError("delivery failed")
        await service.run_once()
        assert model.chat.await_count == 1
        saved = await repo.load(10)
        assert saved and saved.report and not saved.notified
        bot.send_message.side_effect = None
        restored = CodeReviewService(
            cast(Bot, bot),
            model,
            access,
            CodeReviewsRepository(tmp_path),
            clock=lambda: now[0],
        )
        await restored.run_once()
        assert model.chat.await_count == 1 and (await repo.load(10)).notified  # type: ignore[union-attr]
        cached = await restored.review(10)
        assert cached and "Замечание" in cached and model.chat.await_count == 1
        (tmp_path / "services/sample.py").write_text("value = 44", "utf-8")
        await restored.run_once()
        assert model.chat.await_count == 1
        now[0] += 86400
        await restored.run_once()
        assert model.chat.await_count == 2
        await restored.run_once()
        assert model.chat.await_count == 2
        assert current_source_owner.get() is None and current_llm_user.get() is None

    asyncio.run(run())


def test_manual_review_rejects_bad_paths_and_reserves_failed_attempt(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        model = AsyncMock(spec=DeepSeekService)
        service = CodeReviewService(
            AsyncMock(spec=Bot),
            model,
            source(tmp_path),
            CodeReviewsRepository(tmp_path),
            clock=lambda: 10000,
        )
        for owner, path in [(11, ""), (10, "../.env")]:
            with pytest.raises(ValueError):
                await service.review(owner, path)
        model.chat.assert_not_awaited()
        model.chat.side_effect = DeepSeekConnectionError("unavailable")
        with pytest.raises(ValueError):
            await service.review(10, "services/sample.py")
        assert "через час" in (await service.review(10) or "")
        assert model.chat.await_count == 1 and current_source_owner.get() is None

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["outsider", "group", "forward", "bot"])
def test_command_uses_telegram_scope_before_reading_anything(
    tmp_path: Path, reason: str
) -> None:
    async def run() -> None:
        service = CodeReviewService(
            AsyncMock(spec=Bot),
            AsyncMock(spec=DeepSeekService),
            source(tmp_path),
            CodeReviewsRepository(tmp_path),
        )
        incoming = message(
            "/code_review services/sample.py", user=11 if reason == "outsider" else 10
        )
        if reason == "group":
            incoming.chat.type = "group"
        if reason == "forward":
            incoming.forward_origin = object()
        if reason == "bot":
            incoming.from_user.is_bot = True
        await answer_review(service, incoming)
        service.model.chat.assert_not_awaited()  # type: ignore[attr-defined]
        assert "личном чате" in incoming.answer.await_args.args[0]

    asyncio.run(run())
