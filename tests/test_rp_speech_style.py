"""Настройка речи RP: миграция, сохранение, API и фактическая передача в модель."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.core.user_state import ConversationTurn, UserState, UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.proactive import _ConversationSnapshot


@pytest.mark.parametrize("allowed", [False, True])
def test_speech_choice_is_sent_to_model_only_for_rp(allowed: bool) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.content_mode = "soft"
    state.roleplay_active = True
    state.roleplay_allow_profanity = allowed
    asyncio.run(engine.respond(42, "*осматриваюсь вокруг*"))
    prompt = model.chat.await_args.kwargs["system_prompt"]
    assert (
        "Стиль речи в RP: мат разрешён" if allowed else "Стиль речи в RP: без мата"
    ) in prompt
    state.stop_roleplay()
    assert state.roleplay_allow_profanity is allowed
    asyncio.run(engine.respond(42, "Привет, как дела?"))
    assert "Стиль речи в RP:" not in model.chat.await_args.kwargs["system_prompt"]


def test_style_setting_is_private_to_user_and_conversation_and_survives_restart(
    tmp_path: Path,
) -> None:
    store = UserStateStore(persistence=UserStateRepository(tmp_path))

    async def scenario() -> None:
        async with store.use(42) as state:
            state.roleplay_allow_profanity = True
            state.roleplay_character = "путешественник"
            state.history.append(ConversationTurn("Привет", "Здравствуй"))
        new_store = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with new_store.use(42) as state:
            assert (
                state.roleplay_allow_profanity
                and state.roleplay_character == "путешественник"
            )
            assert len(state.history) == 1
        async with new_store.use(99) as state:
            assert not state.roleplay_allow_profanity
        async with new_store.use_conversation(42, -100) as state:
            assert not state.roleplay_allow_profanity
            state.roleplay_allow_profanity = True
        third = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with third.use_conversation(42, -100) as state:
            assert state.roleplay_allow_profanity
        await third.reset_user(42)
        async with third.use(42) as state:
            assert not state.roleplay_allow_profanity

    asyncio.run(scenario())


def test_migration_defaults_to_no_profanity_without_losing_previous_state(
    tmp_path: Path,
) -> None:
    async def seed() -> None:
        store = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with store.use(42) as state:
            state.roleplay_character = "Тестовый персонаж"
            state.roleplay_preferences = "загадки"
        async with store.use_conversation(42, -100) as state:
            state.roleplay_character = "Групповой персонаж"

    asyncio.run(seed())
    with closing(sqlite3.connect(tmp_path / "user_states.db")) as db, db:
        for table in ("user_states", "conversation_states"):
            db.execute("ALTER TABLE " + table + " DROP COLUMN roleplay_allow_profanity")
    store = UserStateStore(persistence=UserStateRepository(tmp_path))

    async def check() -> None:
        async with store.use(42) as state:
            assert not state.roleplay_allow_profanity
            assert (
                state.roleplay_character == "Тестовый персонаж"
                and state.roleplay_preferences == "загадки"
            )
        async with store.use_conversation(42, -100) as state:
            assert (
                not state.roleplay_allow_profanity
                and state.roleplay_character == "Групповой персонаж"
            )

    asyncio.run(check())
    UserStateRepository(tmp_path)


def test_api_saves_boolean_and_clears_it_with_rp_profile(tmp_path: Path) -> None:
    store = UserStateStore(persistence=UserStateRepository(tmp_path))

    async def scenario() -> None:
        server = MiniAppServer(TOKEN, store)
        async with TestClient(TestServer(server.application())) as client:
            headers = {"X-Telegram-Init-Data": _signed_init_data()}
            initial = await client.get("/api/profile", headers=headers)
            assert (await initial.json())["roleplay_allow_profanity"] is False
            response = await client.patch(
                "/api/profile", json={"roleplay_allow_profanity": True}, headers=headers
            )
            assert (
                response.status == 200
                and (await response.json())["roleplay_allow_profanity"] is True
            )
            other = await client.get(
                "/api/profile",
                headers={"X-Telegram-Init-Data": _signed_init_data(user_id=99)},
            )
            assert (await other.json())["roleplay_allow_profanity"] is False
            response = await client.patch(
                "/api/profile",
                json={"roleplay_allow_profanity": False},
                headers=headers,
            )
            assert (
                response.status == 200
                and (await response.json())["roleplay_allow_profanity"] is False
            )
            await client.patch(
                "/api/profile", json={"roleplay_allow_profanity": True}, headers=headers
            )
            cleared = await client.delete("/api/profile/roleplay", headers=headers)
            assert (await cleared.json())["roleplay_allow_profanity"] is False

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, [], {}])
def test_api_rejects_non_boolean_without_changing_profile(value: Any) -> None:
    store = UserStateStore()

    async def scenario() -> None:
        async with TestClient(
            TestServer(MiniAppServer(TOKEN, store).application())
        ) as client:
            response = await client.patch(
                "/api/profile",
                json={"roleplay_allow_profanity": value},
                headers={"X-Telegram-Init-Data": _signed_init_data()},
            )
            assert response.status == 400
            assert not store.get(42).roleplay_allow_profanity

    asyncio.run(scenario())


def test_proactive_snapshot_detects_style_change_and_stop_retains_setting() -> None:
    state = UserState(roleplay_active=True, roleplay_allow_profanity=True)
    before = _ConversationSnapshot.from_state(state)
    assert before.allow_profanity
    state.roleplay_allow_profanity = False
    assert _ConversationSnapshot.from_state(state) != before
    state.roleplay_allow_profanity = True
    state.stop_roleplay()
    assert state.roleplay_allow_profanity
    state.reset_all()
    assert not state.roleplay_allow_profanity
