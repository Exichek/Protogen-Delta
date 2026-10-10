"""Контекст личной сцены и отмена устаревших фоновых приглашений."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot

from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger


def messenger(
    repository: MemoriesRepository, states: UserStateStore, bot: Any, model: Any
) -> ProactiveMessenger:
    return ProactiveMessenger(
        bot=cast(Bot, bot),
        deepseek=cast(DeepSeekService, model),
        repository=repository,
        user_states=states,
        system_prompt="personality",
        config=ProactiveConfig(idle_seconds=20, cooldown_seconds=100),
        clock=lambda: 40.0,
        local_datetime=lambda: datetime(2026, 10, 10, 12),
    )


def test_active_scene_uses_private_open_roleplay_turns(tmp_path: Path) -> None:
    repository = MemoriesRepository(tmp_path)
    states = UserStateStore()
    state = states.get(7)
    state.roleplay_active = True
    state.history.extend(
        [
            ConversationTurn("*вошёл в старую башню*", "Прошлая сцена", True),
            ConversationTurn("*подошёл к мосту*", "*поднял фонарь* Мост впереди."),
            ConversationTurn("Где найти ключ?", "Ключ остался у стражника."),
            ConversationTurn("Как настроить Python?", "Технический ответ"),
        ]
    )
    group = states.get_conversation(7, -100)
    group.roleplay_active = True
    group.history.append(ConversationTurn("*вошёл в групповой замок*", "Замок"))
    before = tuple(state.history)
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Продолжим сцену у моста?"

    async def run() -> None:
        await repository.note_user_activity(7, 10)
        await repository.remember(7, "topic", "Обсуждали программирование", 11)
        service = messenger(repository, states, bot, model)
        assert await service.run_once() == 1
        assert await service.run_once() == 0

    asyncio.run(run())
    args = model.chat.await_args.kwargs
    assert args["tool_names"] == ()
    assert "мост" in args["user_message"]
    assert "стражника" in args["user_message"]
    for excluded in ("башню", "групповой", "Python", "программирование"):
        assert excluded not in args["user_message"]
    assert "не продолжение RP" not in args["system_prompt"]
    assert "данные сцены" in args["system_prompt"]
    assert tuple(state.history) == before and state.roleplay_active
    bot.send_message.assert_awaited_once_with(7, "Продолжим сцену у моста?")


def test_stopped_private_scene_and_active_group_use_ordinary_nudge(
    tmp_path: Path,
) -> None:
    repository = MemoriesRepository(tmp_path)
    states = UserStateStore()
    state = states.get(7)
    state.roleplay_active = True
    state.history.append(ConversationTurn("*пошёл к мосту*", "Мост"))
    state.stop_roleplay()
    states.get_conversation(7, -100).roleplay_active = True
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Если хочешь, можем обсудить Python."

    async def run() -> None:
        await repository.note_user_activity(7, 10)
        await repository.remember(7, "topic", "Python", 11)
        assert await messenger(repository, states, bot, model).run_once() == 1

    asyncio.run(run())
    args = model.chat.await_args.kwargs
    assert "не продолжение RP" in args["system_prompt"]
    assert "Python" in args["user_message"] and "мост" not in args["user_message"]
    bot.send_message.assert_awaited_once_with(7, model.chat.return_value)


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "Почему ты не отвечаешь?",
        "Если захочешь поболтать, я на связи.",
        "Есть идеи, чем заняться сегодня.",
    ],
)
def test_scene_fallback_invites_continuation_without_inventing_events(
    tmp_path: Path, reply: str
) -> None:
    repository = MemoriesRepository(tmp_path)
    states = UserStateStore()
    states.get(7).roleplay_active = True
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = reply

    async def run() -> None:
        await repository.note_user_activity(7, 10)
        assert await messenger(repository, states, bot, model).run_once() == 1

    asyncio.run(run())
    assert model.chat.await_args.kwargs["user_message"].endswith("[]")
    bot.send_message.assert_awaited_once_with(
        7, "Хочешь продолжить нашу RP-сцену с того места, где остановились?"
    )
    assert not states.get(7).history


@pytest.mark.parametrize(
    "change", ["stop", "start", "reset", "history", "character", "profile", "age"]
)
def test_pending_nudge_is_discarded_when_scene_changes(
    tmp_path: Path, change: str
) -> None:
    repository = MemoriesRepository(tmp_path)
    states = UserStateStore()
    states.get(7).roleplay_active = change != "start"
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)

    async def generate(**kwargs: Any) -> str:
        # The model call must not retain the conversation lock.
        async with states.use_conversation(7) as state:
            if change == "stop":
                state.stop_roleplay()
            elif change == "start":
                state.roleplay_active = True
            elif change == "reset":
                state.reset_all()
            elif change == "history":
                state.history.append(ConversationTurn("*новый ход*", "Новый ответ"))
            elif change == "character":
                state.roleplay_character = "Другой персонаж"
            elif change == "profile":
                state.delta_appearance_profile = '{"personality":"Сдержанный"}'
            else:
                state.age_restricted = True
        return "Хочешь продолжить сцену?"

    model.chat.side_effect = generate

    async def run() -> None:
        await repository.note_user_activity(7, 10)
        assert await messenger(repository, states, bot, model).run_once() == 0

    asyncio.run(asyncio.wait_for(run(), 5))
    bot.send_message.assert_not_awaited()


def test_scene_is_loaded_from_persistence_after_restart(tmp_path: Path) -> None:
    repository = MemoriesRepository(tmp_path)
    persistence = UserStateRepository(tmp_path)
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Продолжим искать дорогу через лес?"

    async def run() -> None:
        states = UserStateStore(persistence=persistence, wall_clock=lambda: 40)
        async with states.use_conversation(7) as state:
            state.roleplay_active = True
            state.history.append(ConversationTurn("*вошёл в лес*", "*осветил тропу*"))
        fresh = UserStateStore(persistence=persistence, wall_clock=lambda: 40)
        await repository.note_user_activity(7, 10)
        assert await messenger(repository, fresh, bot, model).run_once() == 1
        assert fresh.get(7).roleplay_active

    asyncio.run(run())
    assert "лес" in model.chat.await_args.kwargs["user_message"]
