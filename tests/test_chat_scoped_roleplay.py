"""RP и история не переходят между чатами, даже после рестарта и reset."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from test_document_handler import _message as document_message
from test_document_handler import _router as document_router
from test_media_handler import JPEG_DATA
from test_media_handler import _message as media_message
from test_media_handler import _router as media_router
from test_response_engine import _create_engine
from test_simple_handlers import _create_message_mock
from test_sticker_requests import STICKER_EXPLANATION, STICKER_QUESTION
from test_stickers import _entry
from test_voice_handler import _message as voice_message
from test_voice_handler import _router as voice_router

from protogen_delta.core.chat_scope import chat_scope_options
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.handlers.delivery import create_reply_delivery
from protogen_delta.handlers.rp import create_rp_router
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.stickers import StickersRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.response_engine import personal_fact_options
from protogen_delta.services.stickers import ContextualStickerService


def _store(path: Path) -> UserStateStore:
    return UserStateStore(
        persistence=UserStateRepository(path), wall_clock=lambda: 1000.0
    )


@pytest.mark.parametrize("chat_type", ["group", "supergroup", "channel", "private"])
def test_chat_scope_is_explicit(chat_type: str) -> None:
    if chat_type == "private":
        assert chat_scope_options(42, chat_type) == {}
        assert personal_fact_options(chat_type, 42) == {}
    else:
        assert chat_scope_options(42, chat_type) == {"chat_id": 42}
        assert personal_fact_options(chat_type, 42) == {
            "chat_id": 42,
            "use_personal_facts": False,
        }
        assert personal_fact_options(chat_type) == {"use_personal_facts": False}


def test_persistent_scopes_preserve_private_data_without_copying_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        states = _store(tmp_path)
        async with states.use(1) as private:
            private.content_mode = "adult"
            private.roleplay_active = True
            private.roleplay_configuration = "female"
            private.roleplay_character = "PRIVATE_CHARACTER"
            private.roleplay_preferences = "PRIVATE_PREFERENCES"
            private.roleplay_boundaries = "PRIVATE_BOUNDARIES"
            private.roleplay_fetishes = ("PRIVATE_THEME",)
            private.delta_appearance = "PRIVATE_APPEARANCE"
            private.emotions.warmth = 0.8
            private.relationship.trust = 0.7
            private.history.append(ConversationTurn("private", "private reply"))
        for user, chat, description in ((1, -10, "A"), (1, -20, "B"), (2, -10, "C")):
            async with states.use_conversation(user, chat) as group:
                assert not group.roleplay_active
                assert not group.history
                assert group.roleplay_character == group.roleplay_preferences == ""
                assert group.roleplay_boundaries == group.delta_appearance == ""
                assert group.roleplay_configuration == "male"
                assert group.roleplay_fetishes == ()
                assert group.emotions.warmth == group.relationship.trust == 0
                assert group.content_mode == ("adult" if user == 1 else "unselected")
                group.roleplay_character = description
                group.roleplay_active = True
                group.delta_appearance = description
                group.roleplay_preferences = description
                group.roleplay_boundaries = description
                group.roleplay_fetishes = (description,)
                group.emotions.warmth = 0.2
                group.relationship.trust = 0.3
                group.content_mode = "soft"  # Cannot change the global age setting.
        # Simulate a new process and independently restore all three group states.
        restored = _store(tmp_path)
        for user, chat, description in ((1, -10, "A"), (1, -20, "B"), (2, -10, "C")):
            async with restored.use_conversation(user, chat) as group:
                assert group.roleplay_active
                assert group.roleplay_character == group.delta_appearance == description
                assert (
                    group.roleplay_preferences
                    == group.roleplay_boundaries
                    == description
                )
                assert group.roleplay_fetishes == (description,)
                assert group.emotions.warmth == 0.2
                assert group.relationship.trust == 0.3
                assert group.content_mode == ("adult" if user == 1 else "unselected")
        async with restored.use(1) as private:
            assert private.roleplay_active
            assert private.roleplay_character == "PRIVATE_CHARACTER"
            assert private.delta_appearance == "PRIVATE_APPEARANCE"
            assert private.content_mode == "adult"
            assert private.emotions.warmth == 0.8
            assert private.relationship.trust == 0.7
            private.content_mode = "soft"
        async with restored.use_conversation(1, -10) as group:
            assert group.content_mode == "soft"
            assert group.roleplay_character == "A"

    asyncio.run(scenario())


def test_group_prompts_and_history_never_read_private_memory() -> None:
    async def scenario() -> None:
        engine, _, model, _, _, _ = _create_engine()
        engine._memory = cast(MemoryService, AsyncMock(spec=MemoryService))
        private = engine._user_states.get(1)
        private.content_mode = "adult"
        private.roleplay_active = True
        private.roleplay_character = "PRIVATE_CHARACTER"
        private.roleplay_preferences = "PRIVATE_PREFERENCES"
        private.roleplay_boundaries = "PRIVATE_BOUNDARIES"
        private.delta_appearance = "PRIVATE_APPEARANCE"
        private.history.append(ConversationTurn("PRIVATE_HISTORY", "PRIVATE_REPLY"))
        for user, chat, marker in (
            (1, -10, "GROUP_A"),
            (1, -20, "GROUP_B"),
            (2, -10, "OTHER_USER"),
        ):
            state = engine._user_states.get_conversation(user, chat)
            state.roleplay_character = marker
            state.delta_appearance = marker + "_APPEARANCE"
            # Enforce privacy inside the engine even without caller's false flag.
            await engine.respond(user, "Привет", chat_id=chat)
            assert model.chat.await_args is not None
            prompt = model.chat.await_args.kwargs["system_prompt"]
            assert "PRIVATE_" not in prompt
            assert marker in prompt
            assert model.chat.await_args.kwargs["history"] == ()
            await engine.respond(
                user, "Как дела?", chat_id=chat, use_personal_facts=False
            )
            assert model.chat.await_args.kwargs["history"] == (
                ConversationTurn("Привет", "Ответ"),
            )
            assert len(state.history) == 2
        assert private.history[0].user_message == "PRIVATE_HISTORY"
        assert len(private.history) == 1
        assert not cast(AsyncMock, engine._memory).mock_calls

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["command", "text", "mixed"])
def test_rp_stop_affects_only_current_chat(stop: str) -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        for chat in (None, -10, -20):
            state = engine._user_states.get_conversation(1, chat)
            state.roleplay_active = True
            state.roleplay_character = str(chat)
            state.roleplay_fetishes = ("bondage",)
            state.emotions.arousal = 0.5
        message, _, _ = _create_message_mock(
            "/rp off" if stop == "command" else "Стоп RP", 1
        )
        cast(Any, message).chat = SimpleNamespace(id=-10, type="supergroup")
        if stop == "mixed":
            await engine.respond(1, "Стоп RP. Как дела?", chat_id=-10)
        else:
            router = (
                create_rp_router(engine)
                if stop == "command"
                else create_text_router(engine)
            )
            await router.message.handlers[0].callback(message)
        for chat in (None, -10, -20):
            state = engine._user_states.get_conversation(1, chat)
            assert state.roleplay_active == (chat != -10)
            assert state.roleplay_character == str(chat)
            assert state.roleplay_fetishes == (() if chat == -10 else ("bondage",))
            assert state.emotions.arousal == (0 if chat == -10 else 0.5)

    asyncio.run(scenario())


def test_private_miniapp_appearance_does_not_change_group() -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        group = engine._user_states.get_conversation(1, -10)
        group.delta_appearance = "group appearance"
        await engine.set_delta_appearance_from_text(1, "Синий сергал")
        assert group.delta_appearance == "group appearance"
        assert engine._user_states.get(1).delta_appearance == "Синий сергал"
        await engine.respond(1, "Верни базовый облик", chat_id=-10)
        assert group.delta_appearance == ""
        assert engine._user_states.get(1).delta_appearance == "Синий сергал"

    asyncio.run(scenario())


def test_full_reset_waits_for_group_delivery_and_deletes_unloaded_scopes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        states = _store(tmp_path)
        for user, chat in ((1, None), (1, -10), (1, -20), (2, -10)):
            async with states.use_conversation(user, chat) as state:
                state.roleplay_active = True
                state.delta_appearance = "old appearance"
        # -20 is not loaded into the new runtime cache, but reset must delete it.
        states = _store(tmp_path)
        engine, _, _, _, _, _ = _create_engine()
        engine._user_states = states
        delivering, release = asyncio.Event(), asyncio.Event()

        async def deliver(reply: str) -> None:
            delivering.set()
            await release.wait()

        response = asyncio.create_task(
            engine.respond_and_deliver(1, "Привет", deliver, chat_id=-10)
        )
        await delivering.wait()
        reset = asyncio.create_task(engine.reset_user(1))
        await asyncio.sleep(0)
        assert not reset.done()
        release.set()
        await response
        await reset
        assert not states.get_conversation(1, -10).history
        assert not states.get_conversation(1, -10).roleplay_active
        repository = UserStateRepository(tmp_path)
        for key in (1, (-10, 1), (-20, 1)):
            assert await repository.load(key) is None
        other = await repository.load((-10, 2))
        assert other and other.roleplay_active
        async with _store(tmp_path).use_conversation(1, -20) as fresh:
            assert not fresh.roleplay_active
            assert fresh.delta_appearance == ""
            assert fresh.content_mode == "unselected"

    asyncio.run(scenario())


def test_failed_delivery_does_not_add_history_to_either_scope() -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        with pytest.raises(RuntimeError):
            await engine.respond_and_deliver(
                1, "Привет", AsyncMock(side_effect=RuntimeError()), chat_id=-10
            )
        assert not engine._user_states.get(1).history
        assert not engine._user_states.get_conversation(1, -10).history
        assert not engine._delivering_users

    asyncio.run(scenario())


def test_group_context_reset_does_not_touch_private_history() -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        for chat in (None, -10, -20):
            engine._user_states.get_conversation(1, chat).history.append(
                ConversationTurn(str(chat), "reply")
            )
        await engine.reset_user_context(1, chat_id=-10)
        assert not engine._user_states.get_conversation(1, -10).history
        assert engine._user_states.get(1).history
        assert engine._user_states.get_conversation(1, -20).history

    asyncio.run(scenario())


def test_stickers_use_only_current_chat_history_and_mood(tmp_path: Path) -> None:
    async def scenario() -> None:
        states = UserStateStore()
        private = states.get(1)
        private.mood = "angry"
        private.history.append(ConversationTurn(STICKER_QUESTION, STICKER_EXPLANATION))
        repository = StickersRepository(tmp_path)
        repository.upsert(_entry("angry", "angry"))
        repository.upsert(_entry("happy", "happy"))
        bot = AsyncMock(spec=Bot)
        service = ContextualStickerService(
            cast(Bot, bot), repository, states, min_replies=1, chance=1
        )
        assert service.is_request(1, "скинь какой-то")
        assert not service.is_request(1, "скинь какой-то", chat_id=-10)
        assert not await service.maybe_send(chat_id=-10, user_id=1, scope_chat_id=-10)
        group = states.get_conversation(1, -10)
        group.history.append(ConversationTurn(STICKER_QUESTION, STICKER_EXPLANATION))
        assert service.is_request(1, "скинь какой-то", chat_id=-10)
        assert not service.is_request(1, "скинь какой-то", chat_id=-20)
        group.mood = "happy"
        assert await service.maybe_send(chat_id=-10, user_id=1, scope_chat_id=-10)
        bot.send_sticker.assert_awaited_once_with(chat_id=-10, sticker="file-happy")

    asyncio.run(scenario())


def test_group_handlers_pass_scope_to_response_and_sticker_delivery() -> None:
    async def scenario() -> None:
        engine, _, model, _, _, _ = _create_engine()
        message, answer, _ = _create_message_mock("Привет", 1)
        cast(Any, message).chat = SimpleNamespace(id=-10, type="group")
        cast(Any, message).sticker = None
        await create_text_router(engine).message.handlers[0].callback(message)
        answer.assert_awaited_once_with("Ответ")
        assert not engine._user_states.get(1).history
        assert engine._user_states.get_conversation(1, -10).history
        service = AsyncMock(spec=ContextualStickerService)
        await create_reply_delivery(
            message, None, cast(ContextualStickerService, service), user_id=1
        )("Готово")
        assert service.maybe_send.await_args is not None
        assert service.maybe_send.await_args.kwargs["scope_chat_id"] == -10
        assert model.chat.await_count == 1

    asyncio.run(scenario())


def test_scoped_row_delete_and_integrity(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _store(tmp_path)
        for chat in (None, -10, -20):
            async with store.use_conversation(1, chat) as state:
                state.roleplay_active = True
        repository = UserStateRepository(tmp_path)
        await repository.delete((-10, 1))
        assert await repository.load((-10, 1)) is None
        assert await repository.load((-20, 1)) is not None
        assert await repository.load(1) is not None
        with closing(sqlite3.connect(tmp_path / "user_states.db")) as db:
            assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)

    asyncio.run(scenario())


def test_scope_cache_expires_without_losing_persistent_roleplay(tmp_path: Path) -> None:
    async def scenario() -> None:
        now = [100.0]
        states = UserStateStore(
            clock=lambda: now[0],
            retention_seconds=10,
            persistence=UserStateRepository(tmp_path),
        )
        async with states.use_conversation(1, -10) as state:
            state.roleplay_character = "persisted group character"
        now[0] = 111.0
        states.get(2)
        assert (-10, 1) not in states._states
        async with states.use_conversation(1, -10) as restored:
            assert restored.roleplay_character == "persisted group character"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["photo", "document", "voice"])
@pytest.mark.parametrize("chat_type", ["group", "supergroup", "channel"])
def test_attachment_handlers_keep_explicit_chat_scope(
    kind: str, chat_type: str
) -> None:
    if kind == "photo":
        router, engine, _ = media_router(JPEG_DATA)
        message, _ = media_message(
            photo=[SimpleNamespace(file_id="photo", file_size=len(JPEG_DATA))]
        )
    elif kind == "document":
        router, engine, _ = document_router(b"document text")
        message, _ = document_message(
            SimpleNamespace(
                file_id="doc",
                file_size=13,
                file_name="info.txt",
                mime_type="text/plain",
            )
        )
    else:
        router, engine, _, _ = voice_router()
        message, _ = voice_message(
            voice=SimpleNamespace(file_id="voice", file_size=10, duration=2)
        )
    cast(Any, message).chat = SimpleNamespace(id=-10, type=chat_type)
    engine.respond_and_deliver.side_effect = None
    asyncio.run(router.message.handlers[0].callback(message))
    assert engine.respond_and_deliver.await_args is not None
    assert engine.respond_and_deliver.await_args.kwargs["chat_id"] == -10
    assert engine.respond_and_deliver.await_args.kwargs["use_personal_facts"] is False


def test_scene_setup_changes_only_group_character_and_configuration() -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        private = engine._user_states.get(1)
        private.roleplay_character = "Private character"
        await engine.respond(1, "Мой персонаж: Белый сергал", chat_id=-10)
        await engine.respond(1, "ты в женской конфигурации", chat_id=-10)
        group = engine._user_states.get_conversation(1, -10)
        assert group.roleplay_character == "Белый сергал"
        assert group.roleplay_configuration == "female"
        assert group.roleplay_active
        assert private.roleplay_character == "Private character"
        assert private.roleplay_configuration == "male"
        assert not private.roleplay_active
        assert not engine._user_states.get_conversation(1, -20).roleplay_active

    asyncio.run(scenario())
