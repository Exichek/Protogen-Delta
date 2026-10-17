"""Открытие отчёта не запускает модель и не теряет последнюю успешную проверку."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_code_review import message
from test_source_access import source

from protogen_delta.repositories.code_reviews import CodeReviewsRepository
from protogen_delta.services.code_review import CodeReviewService, answer_review
from protogen_delta.services.deepseek import DeepSeekConnectionError, DeepSeekService


def test_open_saved_report_after_cooldown_without_source_access_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        now = [10000.0]
        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Предложение, которое нужно сохранить."
        access = source(tmp_path)
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(
            AsyncMock(), model, access, repo, clock=lambda: now[0]
        )
        original = await service.review(10, "services/sample.py")
        assert original is not None
        now[0] += 7200
        # Даже недоступность исходников не мешает открыть сохранённый результат.
        snapshot = AsyncMock(side_effect=RuntimeError("unavailable"))
        monkeypatch.setattr(access, "snapshot", snapshot)
        restored = CodeReviewService(
            service.bot,
            model,
            access,
            CodeReviewsRepository(tmp_path),
            clock=lambda: now[0],
        )
        incoming = message("/code_review")
        await answer_review(restored, incoming)
        opened_text = incoming.answer.await_args.args[0]
        assert opened_text.endswith(original)
        assert "новая проверка не запускалась" in opened_text
        snapshot.assert_not_awaited()
        assert model.chat.await_count == 1
        assert (await repo.delivery(10)).text == opened_text  # type: ignore[union-attr]

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["empty", "markup", "network"])
def test_failed_explicit_refresh_keeps_report_and_cooldown(
    tmp_path: Path, failure: str
) -> None:
    async def run() -> None:
        now = [10000.0]
        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Успешный отчёт с предложением."
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(
            AsyncMock(), model, source(tmp_path), repo, clock=lambda: now[0]
        )
        original = await service.review(10, "services/sample.py")
        assert original is not None
        now[0] += 7200
        if failure == "network":
            model.chat.side_effect = DeepSeekConnectionError("synthetic")
        else:
            model.chat.return_value = "" if failure == "empty" else "<｜DSML｜calls>"
        incoming = message("/code_review new services/sample.py")
        await answer_review(service, incoming)
        assert "час" in incoming.answer.await_args.args[0]
        state = await repo.load(10)
        assert state and state.report == original and state.attempted_at == now[0]
        opened = message("/code_review")
        await answer_review(service, opened)
        assert opened.answer.await_args.args[0].endswith(original)
        await answer_review(service, message("/code_review new services/sample.py"))
        assert model.chat.await_count == 2

    asyncio.run(run())


def test_full_report_keeps_text_after_old_character_cutoff(tmp_path: Path) -> None:
    async def run() -> None:
        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Пояснение. " * 700 + "КОНЕЦ_ПРЕДЛОЖЕНИЯ"
        service = CodeReviewService(
            AsyncMock(), model, source(tmp_path), CodeReviewsRepository(tmp_path)
        )
        incoming = message("/code_review services/sample.py")
        await answer_review(service, incoming)
        chunks = [call.args[0] for call in incoming.answer.await_args_list]
        assert len(chunks) > 1 and chunks[-1].endswith("КОНЕЦ_ПРЕДЛОЖЕНИЯ")
        state = await service.repository.load(10)
        assert state and state.report.endswith("КОНЕЦ_ПРЕДЛОЖЕНИЯ")

    asyncio.run(run())


def test_explicit_new_without_path_refreshes_changed_files(tmp_path: Path) -> None:
    async def run() -> None:
        now = [10000.0]
        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Первый отчёт."
        service = CodeReviewService(
            AsyncMock(),
            model,
            source(tmp_path),
            CodeReviewsRepository(tmp_path),
            clock=lambda: now[0],
        )
        await service.review(10, "services/sample.py")
        (tmp_path / "services/sample.py").write_text("value = 999", "utf-8")
        now[0] += 7200
        model.chat.return_value = "Новый отчёт."
        incoming = message("/code_review new")
        await answer_review(service, incoming)
        assert "Новый отчёт." in incoming.answer.await_args.args[0]
        assert model.chat.await_count == 2

    asyncio.run(run())


def test_failed_automatic_refresh_keeps_cache_without_resending_it(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [10000.0]
        bot = AsyncMock()
        model = AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Последний успешный отчёт."
        repo = CodeReviewsRepository(tmp_path)
        service = CodeReviewService(
            bot, model, source(tmp_path), repo, clock=lambda: now[0]
        )
        original = await service.review(10, "services/sample.py")
        assert original is not None
        (tmp_path / "services/sample.py").write_text("value = 999", "utf-8")
        now[0] += 86400
        model.chat.side_effect = DeepSeekConnectionError("synthetic")
        await service.run_once()
        await service.run_once()
        state = await repo.load(10)
        assert state and state.report == original and state.notified
        incoming = message("/code_review")
        await answer_review(service, incoming)
        assert incoming.answer.await_args.args[0].endswith(original)
        assert model.chat.await_count == 2
        bot.send_message.assert_not_awaited()

    asyncio.run(run())
