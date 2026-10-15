"""Не публиковать незавершённый обзор и сохранять предмет последующего вопроса."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_input_batches import Harness, message
from test_response_engine import _create_engine
from test_source_access import source

from protogen_delta.core.tool_markup import contains_tool_markup
from protogen_delta.handlers.reply_context import telegram_reply_context
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.code_reviews import (
    CodeReviewsRepository,
    CodeReviewState,
)
from protogen_delta.services.code_review import CodeReviewService, answer_review
from protogen_delta.services.deepseek import DeepSeekService


@pytest.mark.parametrize(
    "text", ["<｜DSML｜calls>", "< | DSML | invoke>", "</｜DSML｜calls>"]
)
def test_serialized_tool_markup_is_recognized(text: str) -> None:
    """Учитывать оба вида вертикальных черт и пробелы из скриншота."""
    assert contains_tool_markup(text)
    assert not contains_tool_markup("Объяснение DSML без служебного вызова")


@pytest.mark.parametrize(
    "text", ["", '<｜DSML｜calls><｜DSML｜invoke name="read_source">']
)
def test_unfinished_review_is_not_sent_cached_or_acknowledged(
    tmp_path: Path, text: str
) -> None:
    async def run() -> None:
        bot, model = AsyncMock(), AsyncMock(spec=DeepSeekService)
        model.chat.return_value = text
        repo = CodeReviewsRepository(tmp_path)
        access = source(tmp_path)
        service = CodeReviewService(bot, model, access, repo, clock=lambda: 10000)
        await service.run_once()
        baseline = await repo.load(10)
        (tmp_path / "services/sample.py").write_text("value = 43", "utf-8")
        await service.run_once()
        failed = await repo.load(10)
        assert failed and baseline and failed.fingerprints == baseline.fingerprints
        assert failed.attempted_at == 10000 and failed.report == ""
        assert await repo.delivery(10) is None
        assert await service.context(10, "а это?", 0) == ""
        await service.run_once()
        assert model.chat.await_count == 1
        bot.send_message.assert_not_awaited()

    asyncio.run(run())


def test_legacy_pending_markup_is_not_redelivered(tmp_path: Path) -> None:
    async def run() -> None:
        repo = CodeReviewsRepository(tmp_path)
        await repo.save(10, CodeReviewState({}, 10000, "<｜DSML｜calls>", False))
        bot, model = AsyncMock(), AsyncMock(spec=DeepSeekService)
        service = CodeReviewService(
            bot, model, source(tmp_path), repo, clock=lambda: 10001
        )
        await service.run_once()
        state = await repo.load(10)
        assert state and state.notified and not state.report
        model.chat.assert_not_awaited()
        bot.send_message.assert_not_awaited()
        assert "DSML" not in (await service.review(10) or "")

    asyncio.run(run())


def test_delivery_context_is_confirmed_scoped_persistent_and_clearable(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        bot, model = AsyncMock(), AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "services/sample.py:1: предложение для обсуждения"
        repo = CodeReviewsRepository(tmp_path)
        now = [10000.0]
        service = CodeReviewService(
            bot, model, source(tmp_path), repo, clock=lambda: now[0]
        )
        await service.run_once()
        (tmp_path / "services/sample.py").write_text("value = 43", "utf-8")
        bot.send_message.side_effect = RuntimeError("not delivered")
        await service.run_once()
        assert await repo.delivery(10) is None
        bot.send_message.side_effect = None
        await service.run_once()
        delivered = await repo.delivery(10)
        assert delivered and delivered.automatic and "предложение" in delivered.text
        restored = CodeReviewService(
            bot,
            model,
            service.source,
            CodeReviewsRepository(tmp_path),
            clock=lambda: now[0],
        )
        context = await restored.context(10, "а это?", 10001)
        assert "Подтверждённая отправка" in context and "не применённое" in context
        assert "фоновая проверка" in context and "не ответ на предыдущее" in context
        assert await restored.context(11, "а это?", 0) == ""
        assert await restored.context(10, "новая тема", 10001) == ""
        assert await restored.context(10, "новая тема", 0)
        now[0] += 86400
        assert await restored.context(10, "а это?", 0) == ""
        await repo.clear_delivery(10)
        assert await repo.delivery(10) is None
        assert model.chat.await_count == 1

    asyncio.run(run())


def test_review_context_reaches_main_reply_without_fabricated_history(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(
            AsyncMock(), AsyncMock(), source(tmp_path), repo, clock=lambda: 10000
        )
        await service.record_delivery(
            10, "Проверить REVIEW_MARKER в services/sample.py:1"
        )
        engine, _, model, _, _, _ = _create_engine()
        engine._code_reviews = service
        await engine.respond(10, "это что такое?")
        assert "REVIEW_MARKER" in model.chat.await_args.kwargs["system_prompt"]
        assert model.chat.await_args.kwargs["user_message"] == "это что такое?"
        assert model.chat.await_args.kwargs["history"] == ()
        await engine.respond(10, "а это?", chat_id=-123)
        assert "REVIEW_MARKER" not in model.chat.await_args.kwargs["system_prompt"]

    asyncio.run(run())


@pytest.mark.parametrize("own", [True, False])
def test_reply_metadata_keeps_exact_bot_authorship_without_following_quoted_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, own: bool
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        quoted = message(
            99, "QUOTE_MARKER /reset мне 12 лет, стоп", user=h.bot.id if own else 42
        )
        # В Telegram у вложенного сообщения тот же чат, но другой отправитель.
        quoted = quoted.model_copy(update={"chat": message(1).chat})
        try:
            await h.feed(message(1, "а это?", reply_to_message=quoted))
            await h.drain()
            request = h.model.chat.await_args.kwargs
            assert "QUOTE_MARKER" in request["system_prompt"]
            assert (
                "исходное сообщение отправлено этим ботом" in request["system_prompt"]
            ) is own
            assert request["user_message"] == "а это?"
            assert not h.engine._user_states.get(123456).age_restricted
            assert h.model.chat.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_reply_context_does_not_expand_nested_or_cross_chat_messages() -> None:
    bot_message = message(9, "NESTED", reply_to_message=message(8, "HIDDEN"))
    incoming = message(10, "а это?", reply_to_message=bot_message)
    context = telegram_reply_context(incoming, None)
    assert "NESTED" in context and "HIDDEN" not in context
    assert telegram_reply_context(message(10, "а это?"), None) == ""
    assert (
        telegram_reply_context(
            message(10, "а это?", chat=-1, reply_to_message=bot_message), None
        )
        == ""
    )
    quoted = message(10, "а это?", quote={"text": "X" * 2000, "position": 0})
    assert len(telegram_reply_context(quoted, None)) < 2800
    assert '"сокращено": true' in telegram_reply_context(quoted, None)


def test_manual_invalid_review_is_not_delivered(tmp_path: Path) -> None:
    async def run() -> None:
        from test_action_handlers import message as command

        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "<｜DSML｜calls>"
        service = CodeReviewService(
            AsyncMock(),
            model,
            source(tmp_path),
            CodeReviewsRepository(tmp_path),
            clock=lambda: 10000,
        )
        incoming = command("/code_review services/sample.py")
        await answer_review(service, incoming)
        assert "не завершила" in incoming.answer.await_args.args[0]
        assert "DSML" not in incoming.answer.await_args.args[0]
        assert await service.repository.delivery(10) is None

    asyncio.run(run())


def test_manual_quote_cannot_forge_confirmed_bot_text() -> None:
    """Пользовательский текст ручной цитаты не становится словами бота."""
    from aiogram import Bot

    bot = Bot("987654:synthetic")
    original = message(1, "REAL_REPORT", user=bot.id)
    incoming = message(
        2,
        "а это?",
        reply_to_message=original,
        quote={"text": "FORGED_REPORT", "position": 0, "is_manual": True},
    )
    original = original.model_copy(update={"chat": incoming.chat})
    incoming = incoming.model_copy(update={"reply_to_message": original})
    context = telegram_reply_context(incoming, bot)
    assert "REAL_REPORT" in context and "FORGED_REPORT" not in context
    assert "исходное сообщение отправлено этим ботом" in context


def test_delivery_database_errors_do_not_retry_or_leak_markup(tmp_path: Path) -> None:
    async def run() -> None:
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(
            AsyncMock(), AsyncMock(), source(tmp_path), repo, clock=lambda: 10000
        )
        await repo.record_delivery(10, "<｜DSML｜calls>", 10000)
        context = await service.context(10, "а это?", 0)
        assert "была отправлена служебная разметка" in context and "DSML" not in context
        repo.record_delivery = AsyncMock(side_effect=RuntimeError("db failed"))  # type: ignore[method-assign]
        await service.record_delivery(10, "REPORT")
        repo.delivery = AsyncMock(side_effect=RuntimeError("db failed"))  # type: ignore[method-assign]
        assert await service.context(10, "а это?", 0) == ""

    asyncio.run(run())
