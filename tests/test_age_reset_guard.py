"""Возрастное ограничение переживает очистку диалога и повторный выбор режима."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from test_adult_handler import _create_callback
from test_response_engine import _create_engine

from protogen_delta.core.user_state import (
    EmotionalState,
    RelationshipState,
    UserStateStore,
)
from protogen_delta.handlers.adult import AGE_RESTRICTED_TEXT, create_adult_router
from protogen_delta.repositories.user_state import UserStateRepository


def test_minor_statement_reset_and_adult_button_do_not_call_model(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        store = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = store
        await engine.respond(7, "Мне 15 лет")
        await engine.reset_user(7)
        callback, edit, answer = _create_callback("adult:adult:7", user_id=7)
        await create_adult_router(store).callback_query.handlers[0].callback(callback)
        answer.assert_awaited_once_with(AGE_RESTRICTED_TEXT, show_alert=True)
        edit.assert_not_awaited()
        model.chat.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("restart_before_reset", [False, True])
def test_reset_cannot_erase_minor_restriction(
    tmp_path: Path, restart_before_reset: bool
) -> None:
    async def run() -> None:
        repository = UserStateRepository(tmp_path)
        store = UserStateStore(persistence=repository)
        await store.restrict_minor(7)
        async with store.use_conversation(7, -100) as state:
            state.roleplay_character = "Удалить при сбросе"
        if restart_before_reset:
            store = UserStateStore(persistence=UserStateRepository(tmp_path))
        await store.reset_user(7)
        for states in (
            store,
            UserStateStore(persistence=UserStateRepository(tmp_path)),
        ):
            async with states.use_conversation(7, -100) as state:
                assert state.age_restricted and state.content_mode == "soft"
                assert not state.roleplay_active and not state.roleplay_character
            callback, edit, answer = _create_callback("adult:adult:7", user_id=7)
            await create_adult_router(states).callback_query.handlers[0].callback(
                callback
            )
            answer.assert_awaited_once_with(AGE_RESTRICTED_TEXT, show_alert=True)
            edit.assert_not_awaited()
        async with store.use(8) as other:
            assert not other.age_restricted

    asyncio.run(run())


def test_ram_restriction_survives_reset_and_eviction() -> None:
    async def run() -> None:
        store = UserStateStore()
        await store.restrict_minor(7)
        await store.reset_user(7)
        assert store.remove(7)
        async with store.use(7) as state:
            assert state.age_restricted and state.content_mode == "soft"

    asyncio.run(run())


def test_repository_keeps_restriction_after_delete_and_stale_save(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = UserStateRepository(tmp_path)
        await repo.restrict_conversations(7)
        await repo.delete(7)
        for key in (7, (-100, 7)):
            await repo.save(
                key,
                emotions=EmotionalState(arousal=1),
                relationship=RelationshipState(),
                emotions_updated_at=1,
                roleplay_active=True,
                content_mode="adult",
                age_restricted=False,
            )
            loaded = await repo.load(key)
            assert loaded and loaded.age_restricted and loaded.content_mode == "soft"
            assert not loaded.roleplay_active and not loaded.emotions.arousal
        await repo.delete(7)
        loaded = await UserStateRepository(tmp_path).load(7)
        assert loaded and loaded.age_restricted and not loaded.roleplay_active

    asyncio.run(run())


@pytest.mark.parametrize("group_only", [False, True])
def test_existing_flags_are_migrated_without_dialogue_data(
    tmp_path: Path, group_only: bool
) -> None:
    async def run() -> None:
        repo = UserStateRepository(tmp_path)
        await repo.save(
            (-100, 7) if group_only else 7,
            emotions=EmotionalState(),
            relationship=RelationshipState(),
            emotions_updated_at=1,
            roleplay_active=False,
            age_restricted=True,
        )
        # Моделируем прежнюю базу, где отдельной таблицы ещё не было.
        with closing(sqlite3.connect(tmp_path / "user_states.db")) as db, db:
            db.execute("DROP TABLE IF EXISTS age_restrictions")
        restored = UserStateRepository(tmp_path)
        await restored.delete(7)
        state = await UserStateRepository(tmp_path).load(7)
        assert state and state.age_restricted and state.content_mode == "soft"
        with closing(sqlite3.connect(tmp_path / "user_states.db")) as db:
            assert db.execute("SELECT user_id FROM age_restrictions").fetchall() == [
                (7,)
            ]

    asyncio.run(run())
