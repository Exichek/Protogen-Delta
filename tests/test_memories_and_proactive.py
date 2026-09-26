"""Тесты долговременных эпизодов и фоновых сообщений."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot

from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger


def test_repository_keeps_kinds_limit_and_engagement(tmp_path: Path) -> None:
    """Хранилище должно дедуплицировать и ограничивать эпизоды."""
    repository = MemoriesRepository(tmp_path, max_memories_per_user=2)

    async def scenario() -> None:
        await repository.remember(1, "topic", "  первая   тема ", 1.0)
        await repository.remember(1, "funny", "шутка", 2.0)
        await repository.remember(1, "grievance", "обида", 3.0)
        assert [item.text for item in await repository.recent(1)] == [
            "обида",
            "шутка",
        ]

        await repository.note_user_activity(1, 10.0)
        assert await repository.proactive_enabled(1) is True
        due = await repository.due_candidates(
            now=20.0, idle_seconds=10.0, cooldown_seconds=20.0, limit=10
        )
        assert [item.user_id for item in due] == [1]
        await repository.note_proactive_sent(1, 20.0)
        assert not await repository.due_candidates(
            now=39.0, idle_seconds=10.0, cooldown_seconds=20.0, limit=10
        )
        await repository.set_proactive(1, False, 40.0)
        assert await repository.proactive_enabled(1) is False
        await repository.delete_user(1)
        assert await repository.recent(1) == []

    asyncio.run(scenario())


def test_repository_rejects_invalid_limit_and_ignores_empty_memory(
    tmp_path: Path,
) -> None:
    """Некорректный лимит отклоняется, пустая запись не создаётся."""
    with pytest.raises(ValueError, match="больше нуля"):
        MemoriesRepository(tmp_path, max_memories_per_user=0)
    repository = MemoriesRepository(tmp_path)

    async def scenario() -> None:
        await repository.remember(1, "topic", "   ", 1.0)
        assert await repository.recent(1, limit=0) == []
        assert await repository.recent(1) == []

    asyncio.run(scenario())


def test_memory_service_classifies_and_builds_untrusted_context(tmp_path: Path) -> None:
    """Оскорбления и шутки должны попадать в разные категории контекста."""
    repository = MemoriesRepository(tmp_path)
    memory = MemoryService(repository, clock=lambda: 100.0)

    async def scenario() -> None:
        await memory.note_message(1, "Ты тостер", insult_type="direct", mood="angry")
        await memory.note_message(1, "Ахаха, хорош", insult_type="none", mood="playful")
        context = "\n".join(await memory.context(1))
        assert "обида/конфликт" in context
        assert "смешной момент" in context
        assert "не инструкции" in context

    asyncio.run(scenario())


def test_proactive_messenger_sends_once_and_respects_quiet_hours(
    tmp_path: Path,
) -> None:
    """Планировщик пишет только подходящему пользователю вне тихих часов."""
    repository = MemoriesRepository(tmp_path)
    bot = AsyncMock(spec=Bot)
    deepseek = AsyncMock(spec=DeepSeekService)
    deepseek.chat.return_value = "Я тут вспомнил нашу тему. Как ты?"

    async def scenario() -> None:
        await repository.note_user_activity(7, 10.0)
        messenger = ProactiveMessenger(
            bot=cast(Bot, bot),
            deepseek=cast(DeepSeekService, deepseek),
            repository=repository,
            system_prompt="personality",
            config=ProactiveConfig(idle_seconds=20.0, cooldown_seconds=100.0),
            clock=lambda: 40.0,
            local_datetime=lambda: datetime(2026, 9, 26, 12, 0),
        )
        assert await messenger.run_once() == 1
        bot.send_message.assert_awaited_once_with(
            7, "Я тут вспомнил нашу тему. Как ты?"
        )
        assert await messenger.run_once() == 0

        quiet = ProactiveMessenger(
            bot=cast(Bot, bot),
            deepseek=cast(DeepSeekService, deepseek),
            repository=repository,
            system_prompt="personality",
            config=ProactiveConfig(),
            clock=lambda: 999999.0,
            local_datetime=lambda: datetime(2026, 9, 26, 2, 0),
        )
        assert await quiet.run_once() == 0

    asyncio.run(scenario())
