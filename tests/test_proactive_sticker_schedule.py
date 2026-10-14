"""Отдельные часы стикера, общий предел инициативы и сохранение после рестарта."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendSticker

from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.handlers.proactive_activity import proactive_delivery
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.stickers import StickerEntry, StickersRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger
from protogen_delta.services.stickers import ContextualStickerService


def _build(
    root: Path,
    now: list[float],
    value: float = 0.0,
    *,
    chance: float = 1,
    hour: int = 12,
    text_delay: float = 5400,
    sticker_delay: float = 1800,
) -> tuple[
    MemoriesRepository, UserStateStore, AsyncMock, AsyncMock, ProactiveMessenger
]:
    repository = MemoriesRepository(
        root, random_delay=lambda lo, hi: text_delay if lo == 5400 else sticker_delay
    )
    states = UserStateStore(
        persistence=UserStateRepository(root), wall_clock=lambda: now[0]
    )
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Привет! Как там проект?"
    catalogue = StickersRepository(root)
    catalogue.upsert(
        StickerEntry("safe", "safe-id", ("happy", "greeting"), name="Улыбка")
    )
    catalogue.upsert(StickerEntry("adult", "adult-id", ("greeting",), rating="adult"))
    stickers = ContextualStickerService(
        cast(Bot, bot), catalogue, states, chance=chance
    )
    service = ProactiveMessenger(
        bot=cast(Bot, bot),
        deepseek=cast(DeepSeekService, model),
        repository=repository,
        user_states=states,
        system_prompt="core",
        stickers=stickers,
        random_value=lambda: value,
        clock=lambda: now[0],
        local_datetime=lambda: datetime(2026, 10, 14, hour),
        config=ProactiveConfig(
            idle_seconds=5400,
            idle_max_seconds=10800,
            sticker_idle_seconds=1800,
            sticker_idle_max_seconds=7200,
        ),
    )
    return repository, states, bot, model, service


@pytest.mark.parametrize("delay", [1800, 4000, 7200])
def test_sticker_timer_is_persisted_and_does_not_call_model(
    tmp_path: Path, delay: float
) -> None:
    async def run() -> None:
        now = [100.0]
        repo, _, _, _, service = _build(
            tmp_path, now, sticker_delay=delay, text_delay=10800
        )
        await repo.note_user_activity(7, now[0])
        assert await service.run_once() == 0
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute("SELECT next_sticker_at FROM engagement").fetchone() == (
                100 + delay,
            )
        repo, states, bot, model, service = _build(tmp_path, now, sticker_delay=1800)
        now[0] = 100 + delay - 1
        assert await service.run_once() == 0
        now[0] += 1
        assert sum(await asyncio.gather(service.run_once(), service.run_once())) == 1
        bot.send_sticker.assert_awaited_once_with(chat_id=7, sticker="safe")
        bot.send_message.assert_not_awaited()
        model.chat.assert_not_awaited()
        assert states.get(7).history[-1].proactive
        assert states.get(7).history[-1].user_message == ""
        assert "стикер" in states.get(7).history[-1].assistant_message
        assert proactive_delivery.get() is False

    asyncio.run(run())


@pytest.mark.parametrize("answered", [False, True])
def test_after_sticker_next_initiative_is_text_across_restart(
    tmp_path: Path, answered: bool
) -> None:
    async def run() -> None:
        now = [1900.0]
        repo, states, bot, model, service = _build(tmp_path, now)
        await repo.note_user_activity(7, 100)
        assert await service.run_once() == 1
        repo, states, bot, model, service = _build(tmp_path, now)
        if answered:
            now[0] += 100
            await repo.note_user_activity(7, now[0])
            next_at = now[0] + 5400
        else:
            next_at = 1900 + 72000
        now[0] = next_at - 1
        assert await service.run_once() == 0
        now[0] = next_at
        assert await service.run_once() == 1
        bot.send_sticker.assert_not_awaited()
        bot.send_message.assert_awaited_once()
        model.chat.assert_awaited_once()
        if not answered:
            now[0] += 72000 * 5
            assert await service.run_once() == 0
            assert len(states.get(7).history) == 2

    asyncio.run(run())


@pytest.mark.parametrize("value,expected_stickers", [(0.99, 0), (0.35, 1)])
def test_text_and_both_wait_for_text_deadline_and_count_one_initiative(
    tmp_path: Path, value: float, expected_stickers: int
) -> None:
    async def run() -> None:
        now = [1900.0]
        repo, states, bot, model, service = _build(tmp_path, now, value)
        await repo.note_user_activity(7, 100)
        state = states.get(7)
        state.history.append(
            ConversationTurn("Хочу доделать PROJECT_MARKER", "Ок, начнём с меню")
        )
        # Состояние сохранено перед загрузкой планировщиком.
        async with states.use(7):
            pass
        assert await service.run_once() == 0
        model.chat.assert_not_awaited()
        now[0] = 5500
        assert await service.run_once() == 1
        assert bot.send_message.await_count == 1
        assert bot.send_sticker.await_count == expected_stickers
        assert "PROJECT_MARKER" in model.chat.await_args.kwargs["user_message"]
        if expected_stickers:
            assert [
                call[0]
                for call in bot.mock_calls
                if call[0] in {"send_message", "send_sticker"}
            ] == ["send_message", "send_sticker"]
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute("SELECT unanswered_count FROM engagement").fetchone() == (
                1,
            )

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["quiet", "disabled", "no_stickers"])
def test_initiative_obeys_quiet_hours_and_settings(tmp_path: Path, reason: str) -> None:
    async def run() -> None:
        now = [1900.0]
        repo, _, bot, model, service = _build(
            tmp_path,
            now,
            hour=23 if reason == "quiet" else 12,
            chance=0 if reason == "no_stickers" else 1,
        )
        await repo.note_user_activity(7, 100)
        if reason == "disabled":
            await repo.set_proactive(7, False, 100)
        assert await service.run_once() == 0
        bot.send_sticker.assert_not_awaited()
        bot.send_message.assert_not_awaited()
        model.chat.assert_not_awaited()

    asyncio.run(run())


def test_extra_sticker_error_does_not_duplicate_delivered_text(tmp_path: Path) -> None:
    async def run() -> None:
        now = [5500.0]
        repo, _, bot, model, service = _build(tmp_path, now, 0.35)
        bot.send_sticker.side_effect = TelegramBadRequest(
            method=SendSticker(chat_id=7, sticker="safe"), message="bad file"
        )
        await repo.note_user_activity(7, 100)
        assert await service.run_once() == 1
        assert await service.run_once() == 0
        assert bot.send_message.await_count == 1
        assert model.chat.await_count == 1

    asyncio.run(run())


def test_failed_sticker_is_not_recorded_and_falls_back_only_at_text_deadline(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [1900.0]
        repo, states, bot, model, service = _build(tmp_path, now)
        bot.send_sticker.side_effect = TelegramBadRequest(
            method=SendSticker(chat_id=7, sticker="safe"), message="bad file"
        )
        await repo.note_user_activity(7, 100)
        assert await service.run_once() == 0
        assert not states.get(7).history
        now[0] = 5500
        assert await service.run_once() == 1
        bot.send_message.assert_awaited_once()
        model.chat.assert_awaited_once()

    asyncio.run(run())
