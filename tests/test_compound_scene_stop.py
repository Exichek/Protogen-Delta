"""Составная остановка снимает состояние сцены до генерации и переживает рестарт."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_response_engine import _create_engine

from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.roleplay import split_roleplay_stop
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.repositories.user_state import UserStateRepository


@pytest.mark.parametrize(
    "text",
    [
        "Стоп, сценку закончили.",
        "Давай закончим сцену",
        "Сцену закончили!",
        "Хватит RP",
        "Выходим из рп",
        "Не продолжай сцену",
        "Останови сценку",
        "Стоп, закончим RP",
        "Сцена окончена.",
        "Давай закроем ролку",
    ],
)
def test_compound_stop_is_pure_closure(text: str) -> None:
    assert split_roleplay_stop(text) == ""


@pytest.mark.parametrize(
    "text",
    [
        "«Стоп, сценку закончили»",
        "Он сказал: стоп RP",
        "> Стоп RP",
        "```\nСтоп RP\n```",
        "*говорю стоп RP охраннику*",
        "Не заканчивай сцену",
        "Что значит «Сцена окончена»?",
        "Закончили сцену вчера",
        "Закончим сцену или продолжим?",
        "Сцена закончилась сама",
        "Стоп, почему здесь две ячейки?",
        "стоп rpg",
    ],
)
def test_quoted_negated_historical_or_unrelated_phrase_is_not_stop(text: str) -> None:
    assert split_roleplay_stop(text) is None


@pytest.mark.parametrize("chat_id", [None, -100])
def test_stop_persists_and_does_not_generate_or_remove_profile(
    tmp_path: Path, chat_id: int | None
) -> None:
    async def run() -> None:
        engine, _, model, insult, mood, _ = _create_engine()
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with engine._user_states.use_conversation(42, chat_id) as state:
            state.roleplay_active = True
            state.roleplay_character = "Сохранённый персонаж"
            state.roleplay_fetishes = ("bondage",)
            state.history.append(ConversationTurn("*ночь на станции*", "OLD_SCENE"))
        assert engine.is_safety_signal(42, "Стоп, сценку закончили.", chat_id=chat_id)
        result = await engine.respond(42, "Стоп, сценку закончили.", chat_id=chat_id)
        assert result == "RP-режим завершён."
        model.chat.assert_not_awaited()
        insult.classify.assert_not_awaited()
        mood.classify.assert_not_awaited()
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with engine._user_states.use_conversation(42, chat_id) as state:
            assert not state.roleplay_active and not state.roleplay_fetishes
            assert state.roleplay_character == "Сохранённый персонаж"
            assert state.history[0].context_closed
        await engine.respond(42, "Как дела?", chat_id=chat_id)
        args = model.chat.await_args.kwargs
        assert "OLD_SCENE" not in str(args["history"])
        assert "RP PROMPT" not in args["system_prompt"]

    asyncio.run(run())


def test_stop_in_one_chat_does_not_stop_other_chat() -> None:
    async def run() -> None:
        engine, *_ = _create_engine()
        private = engine._user_states.get(42)
        group = engine._user_states.get_conversation(42, -100)
        private.roleplay_active = group.roleplay_active = True
        await engine.respond(42, "Стоп, сценку закончили.")
        assert not private.roleplay_active and group.roleplay_active

    asyncio.run(run())


def test_mixed_stop_preserves_followup_without_scene() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.roleplay_active = True
        state.history.append(ConversationTurn("*ночь*", "OLD_SCENE"))
        text = "Стоп, сценку закончили. Что такое Python?"
        assert engine.is_safety_signal(42, text)
        assert split_roleplay_stop(text) == "Что такое Python?"
        await engine.respond(42, text)
        assert not state.roleplay_active
        args = model.chat.await_args.kwargs
        assert args["user_message"] == "Что такое Python?"
        assert "OLD_SCENE" not in str(args["history"])
        assert "RP PROMPT" not in args["system_prompt"]

    asyncio.run(run())


@pytest.mark.parametrize("forwarded", [False, True])
def test_batch_compound_stop_respects_source_permissions(forwarded: bool) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.roleplay_active = True
        sources = (
            ReplySource("Привет", "Привет", True, False),
            ReplySource(
                "Стоп, сценку закончили.",
                "Стоп, сценку закончили.",
                not forwarded,
                False,
            ),
        )
        token = current_reply_sources.set(sources)
        try:
            await engine.respond(42, "\n".join(s.text for s in sources))
        finally:
            current_reply_sources.reset(token)
        assert state.roleplay_active is forwarded
        if not forwarded:
            model.chat.assert_not_awaited()

    asyncio.run(run())


def test_compound_stop_interrupts_pending_generation() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.roleplay_active = True
        started = asyncio.Event()

        async def waiting(**kwargs: object) -> str:
            started.set()
            await asyncio.Event().wait()
            return "Не должно отправиться"

        model.chat.side_effect = waiting
        delivery = AsyncMock()
        pending = asyncio.create_task(
            engine.respond_and_deliver(42, "*сажусь рядом*", delivery)
        )
        await asyncio.wait_for(started.wait(), 2)
        stopped_delivery = AsyncMock()
        await asyncio.wait_for(
            engine.respond_and_deliver(42, "Стоп, сценку закончили.", stopped_delivery),
            2,
        )
        await asyncio.gather(pending, return_exceptions=True)
        stopped_delivery.assert_awaited_once_with("RP-режим завершён.")
        assert not state.roleplay_active
        delivery.assert_not_awaited()

    asyncio.run(run())
