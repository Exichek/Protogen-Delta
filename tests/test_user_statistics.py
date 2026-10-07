"""Перезапуск, календарные границы, доступ администратора и отказ базы."""

import asyncio
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import Chat, Message, TelegramObject, User
from PIL import Image

from protogen_delta.core.state import BotState
from protogen_delta.core.user_statistics import UserStatisticsMiddleware
from protogen_delta.handlers.admin import create_admin_router
from protogen_delta.repositories.user_statistics import UserStatisticsRepository
from protogen_delta.services.user_statistics import activity_chart

NOW = datetime(2026, 10, 7, 22, tzinfo=timezone.utc).timestamp()


def test_counts_persist_dedupe_and_use_moscow_days(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = UserStatisticsRepository(tmp_path)
        assert await repo.get(1, NOW) is None
        await repo.record(1, "Old", "old", 1, 1, "text", NOW - 366 * 86400)
        await repo.record(1, "Name", "name", 1, 2, "text", NOW - 2 * 3600)
        await repo.record(1, "Name", "name", 1, 3, "command", NOW)
        await repo.record(2, "Other", "other", 2, 3, "photo", NOW)
        reloaded = UserStatisticsRepository(tmp_path)
        await reloaded.record(1, "Name", "name", 1, 3, "command", NOW)
        stats = await reloaded.get(1, NOW)
        assert stats and stats.total == 3
        assert stats.daily == (("2026-10-07", 1), ("2026-10-08", 1))
        assert dict(stats.kinds) == {"command": 1, "text": 1}
        assert stats.first_seen == NOW - 366 * 86400 and stats.last_seen == NOW
        assert stats.name == "Name" and stats.username == "name"
        other = await repo.get(2, NOW)
        assert other and other.total == 1
        await repo.delete_user(1)
        assert await repo.get(1, NOW) is None
        other = await repo.get(2, NOW)
        assert other and other.total == 1
        await repo.record(1, "New", "", 1, 3, "text", NOW)
        new = await repo.get(1, NOW)
        assert new and new.total == 1

    asyncio.run(scenario())


def test_chart_handles_zero_and_large_counts(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = UserStatisticsRepository(tmp_path)
        await repo.record(1, "Name", "", 1, 1, "text", NOW)
        stats = await repo.get(1, NOW)
        assert stats
        for at in (NOW, NOW + 366 * 86400):
            data = activity_chart(stats, at)
            with Image.open(BytesIO(data)) as image:
                assert image.format == "PNG" and image.size == (1000, 400)
                colors = image.getcolors(400000)
                assert colors is not None and len(colors) > 2

    asyncio.run(scenario())


def test_middleware_counts_metadata_and_always_continues() -> None:
    async def scenario() -> None:
        repo = AsyncMock(spec=UserStatisticsRepository)
        middleware = UserStatisticsMiddleware(repo)
        handler = AsyncMock(return_value="done")
        event = Message(
            message_id=10,
            date=datetime.fromtimestamp(NOW, timezone.utc),
            chat=Chat(id=7, type="private"),
            from_user=User(id=7, is_bot=False, first_name="Name", username="name"),
            text="/id",
        )
        assert await middleware(handler, event, {}) == "done"
        assert repo.record.await_args.args == (7, "Name", "name", 7, 10, "command", NOW)
        repo.record.side_effect = OSError("db")
        await middleware(handler, event, {})
        await middleware(handler, TelegramObject(), {})
        await middleware(handler, event.model_copy(update={"from_user": None}), {})
        assert event.from_user is not None
        await middleware(
            handler,
            event.model_copy(
                update={
                    "from_user": event.from_user.model_copy(update={"is_bot": True})
                }
            ),
            {},
        )
        assert handler.await_count == 5 and repo.record.await_count == 2

    asyncio.run(scenario())


def test_admin_statistics_is_private_and_does_not_require_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("protogen_delta.handlers.admin.time.time", lambda: NOW)

    async def scenario() -> None:
        repo = UserStatisticsRepository(tmp_path)
        await repo.record(42, "<User>", "user", 42, 1, "text", NOW)
        repository = AsyncMock(wraps=repo)
        router = create_admin_router(
            Mock(),
            Mock(),
            BotState(),
            frozenset({1}),
            cast(UserStatisticsRepository, repository),
        )
        callback = router.message.handlers[0].callback
        message = Mock(
            spec=Message,
            from_user=SimpleNamespace(id=2),
            chat=SimpleNamespace(type="private"),
            text="/userstats 42",
            reply_to_message=None,
            answer=AsyncMock(),
            answer_photo=AsyncMock(),
        )
        await callback(message)
        message.from_user.id = 1
        message.chat.type = "group"
        await callback(message)
        repository.get.assert_not_awaited()
        message.chat.type = "private"
        for text in (
            "/userstats -2",
            "/userstats 0",
            "/userstats 1 2",
            "/userstats " + "9" * 19,
            "/userstats ²",
        ):
            message.text = text
            await callback(message)
        repository.get.assert_not_awaited()
        message.text = "/userstats 42"
        await callback(message)
        caption = message.answer_photo.await_args.kwargs["caption"]
        assert "<User>" in caption and "1 / 1 / 1 / 1" in caption and "365" in caption
        assert message.answer_photo.await_args.kwargs["parse_mode"] is None
        message.text = "/userstats"
        message.reply_to_message = SimpleNamespace(from_user=SimpleNamespace(id=42))
        await callback(message)
        message.reply_to_message = None
        await callback(message)
        assert repository.get.await_args.args[0] == 1
        empty = create_admin_router(Mock(), Mock(), BotState(), frozenset({1}))
        await empty.message.handlers[0].callback(message)
        assert "недоступен" in message.answer.await_args.args[0]

    asyncio.run(scenario())
