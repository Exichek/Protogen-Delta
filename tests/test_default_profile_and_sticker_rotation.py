"""Стандартный образ редактируется отдельно, реакции не приветствуют посреди темы."""

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

from aiogram import Bot
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine
from test_stickers import _entry

from protogen_delta.core.appearance_profile import (
    AppearanceProfile,
    active_appearance_profile,
)
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.stickers import StickersRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.appearance_profile_context import (
    appearance_profile_context,
)
from protogen_delta.services.stickers import ContextualStickerService


def test_default_profile_is_saved_restored_and_separate_from_custom(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        base = AppearanceProfile(
            "Любознательный", "Короткие реплики", "Головоломки", "Без драки"
        )
        custom = AppearanceProfile("CUSTOM")
        server = MiniAppServer(TOKEN, states, response_engine=engine)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            response = await client.patch(
                "/api/profile/appearance/behavior",
                headers=headers,
                json={
                    "profile": asdict(base),
                    "expected_appearance": "",
                    "expected_profile": asdict(AppearanceProfile()),
                },
            )
            assert response.status == 200
            assert (await response.json())["delta_appearance_profile"] == asdict(base)
            assert states.get(42).delta_appearance == ""
            assert "Любознательный" in appearance_profile_context(states.get(42))
            await engine.set_delta_appearance_from_text(42, "CUSTOM_IMAGE")
            await engine.set_delta_appearance_profile(
                42,
                custom,
                expected_appearance="CUSTOM_IMAGE",
                expected_profile=AppearanceProfile(),
            )
            assert active_appearance_profile(states.get(42)) == custom
            response = await client.delete("/api/profile/appearance", headers=headers)
            assert response.status == 200
            assert (await response.json())["delta_appearance_profile"] == asdict(base)
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with restored.use(42) as state:
            assert (
                active_appearance_profile(state) == base
                and not state.delta_appearance_profile
            )
        await restored.reset_user(42)
        async with restored.use(42) as state:
            assert active_appearance_profile(state) == AppearanceProfile()
        model.chat.assert_not_awaited()

    asyncio.run(run())


def test_greeting_tags_do_not_win_mid_conversation_and_recent_reactions_rotate(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = StickersRepository(tmp_path)
        repo.upsert(_entry("hello", "greeting", "neutral", "laugh"))
        repo.upsert(_entry("laugh1", "laugh", "neutral"))
        repo.upsert(_entry("laugh2", "laugh", "neutral"))
        bot = AsyncMock(spec=Bot)
        states = UserStateStore()
        states.get(1).history.append(ConversationTurn("Разбираем код", "Смотрю"))
        now = [100.0]
        service = ContextualStickerService(
            cast(Bot, bot),
            repo,
            states,
            chance=1,
            min_replies=1,
            cooldown_seconds=0,
            choose=lambda items: items[0],
            clock=lambda: now[0],
        )
        assert await service.maybe_send(chat_id=1, user_id=1, context_text="ахаха")
        assert bot.send_sticker.await_args is not None
        assert bot.send_sticker.await_args.kwargs["sticker"] == "file-laugh1"
        assert await service.send_proactive(1, "ахаха")
        assert bot.send_sticker.await_args.kwargs["sticker"] == "file-laugh2"
        assert not await service.maybe_send(chat_id=1, user_id=1, context_text="ахаха")
        assert await service.send_proactive(1, "ахаха") is None
        now[0] += 1801
        assert await service.maybe_send(chat_id=1, user_id=1, context_text="ахаха")
        assert await service.maybe_send(chat_id=1, user_id=1, context_text="Привет")
        assert bot.send_sticker.await_args.kwargs["sticker"] == "file-hello"

    asyncio.run(run())
