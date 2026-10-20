"""Живой отклик после инициативы, реальное настроение и явные технические вопросы."""

import asyncio
from collections import deque
from pathlib import Path

import pytest
from test_proactive_sticker_schedule import _build
from test_response_engine import _create_engine

from protogen_delta.core.user_state import ConversationTurn, UserState, UserStateStore
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.initiative_context import (
    initiative_mood_context,
    initiative_reply_context,
)


@pytest.mark.parametrize(
    "question",
    [
        "Йоу, стикеры отправляешь?",
        "Чё пишешь?",
        "Что хочешь?",
        "Чего хотел?",
        "Зачем позвал?",
        "Почему ты мне написал?",
        "Ты мне писал?",
        "Че шлёшь?",
    ],
)
def test_social_followup_uses_actual_initiative(question: str) -> None:
    state = UserState()
    state.emotions.memory.joy = 0.8
    state.history.append(
        ConversationTurn(
            "", "[Дельта самостоятельно отправил стикер: BEEP]", proactive=True
        )
    )
    context = initiative_reply_context(question, state)
    assert "Продолжение живого разговора" in context
    assert "BEEP" in context and "радость: выраженный фон" in context
    assert "Твои подтверждённые" in context


@pytest.mark.parametrize(
    "question",
    [
        "Как работает система инициативы?",
        "По какому принципу выбираешь стикеры?",
        "Через сколько минут ты пишешь?",
        "Расскажи про таймер отправки",
        "Как настроить интервалы?",
        "Покажи код отправки стикеров",
        "Какая у тебя модель?",
        "«Что хочешь?»",
        '```python\nprint("чё пишешь?")\n```',
    ],
)
def test_mechanics_and_quoted_questions_do_not_get_social_override(
    question: str,
) -> None:
    state = UserState()
    state.history.append(ConversationTurn("", "Позвал поболтать", proactive=True))
    assert initiative_reply_context(question, state) == ""


@pytest.mark.parametrize(
    "field,label",
    [
        ("joy", "радость"),
        ("sadness", "грусть"),
        ("anxiety", "тревожность"),
        ("tiredness", "усталость"),
        ("curiosity", "интерес"),
    ],
)
def test_mood_basis_reflects_saved_emotion_instead_of_forcing_joy(
    field: str, label: str
) -> None:
    state = UserState()
    setattr(state.emotions.memory, field, 0.8)
    context = initiative_mood_context(state)
    assert f"{label}: выраженный фон" in context
    if field != "joy":
        assert "радость: выраженный фон" not in context


def test_mixed_emotions_and_resentment_are_not_replaced_with_cheerfulness() -> None:
    state = UserState()
    state.emotions.memory.joy = 0.3
    state.emotions.memory.sadness = 0.7
    state.emotions.irritation = 0.4
    state.relationship.resentment = 0.8
    context = initiative_mood_context(state)
    assert "радость: заметный фон" in context
    assert "грусть: выраженный фон" in context
    assert "недовольство: выраженный фон" in context


def test_idea_followup_separates_voice_examples_from_confirmed_history() -> None:
    """Игровая тема не подтверждает механику из образца или статус проекта."""
    state = UserState()
    state.history.append(
        ConversationTurn(
            "", "Йоу. Хочешь вернуться к нашей идее с игрой?", proactive=True
        )
    )
    context = initiative_reply_context("Что хотел?", state)
    assert "Примеры голоса и сцен" in context
    assert "предложение сейчас" in context
    assert "не приписывай" in context
    assert "ограниченный инвентарь" not in context
    assert "недоделан" not in context
    assert "а не историю прожитого дня" in context
    assert initiative_reply_context("Покажи код игры", state) == ""


def test_neutral_state_does_not_invent_boredom_or_longing() -> None:
    context = initiative_mood_context(UserState())
    assert "ровный спокойный фон" in context
    assert "скука: выраженный фон" not in context
    state = UserState()
    context = initiative_reply_context("Что хочешь?", state)
    assert "Подтверждённой недавней самостоятельной отправки" in context
    assert "Твои подтверждённые" not in context


def test_ordinary_new_topic_is_not_forced_back_to_older_initiative() -> None:
    state = UserState()
    state.history.extend(
        [
            ConversationTurn("", "Давай обсудим космос", proactive=True),
            ConversationTurn("Расскажи про Луну", "Рассказ о Луне"),
        ]
    )
    assert initiative_reply_context("Как приготовить борщ?", state) == ""
    state.history = deque(
        [ConversationTurn("", "Давай обсудим космос", proactive=True)]
    )
    context = initiative_reply_context("Как приготовить борщ?", state)
    assert "новую конкретную задачу" in context


def test_closed_and_distant_initiative_are_not_claimed_as_recent() -> None:
    state = UserState()
    state.history.append(
        ConversationTurn("", "CLOSED_MARKER", context_closed=True, proactive=True)
    )
    for _ in range(5):
        state.history.append(ConversationTurn("вопрос", "ответ"))
    context = initiative_reply_context("Чего хотел?", state)
    assert "CLOSED_MARKER" not in context and "Твои подтверждённые" not in context


def test_sticker_followup_after_restart_uses_delivered_history_and_accumulated_mood(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [100.0]
        repo, states, bot, model, service = _build(tmp_path, now)
        async with states.use_conversation(7) as state:
            state.emotions.memory.joy = 0.8
            state.emotions.playfulness = 0.5
        await repo.note_user_activity(7, now[0])
        assert await service.run_once() == 0
        now[0] += 1800
        assert await service.run_once() == 1
        bot.send_sticker.assert_awaited_once()
        model.chat.assert_not_awaited()
        engine, _, reply_model, _, _, _ = _create_engine()
        engine._user_states = UserStateStore(
            persistence=UserStateRepository(tmp_path), wall_clock=lambda: now[0]
        )
        await engine.respond(7, "Йоу, стикеры отправляешь?")
        call = reply_model.chat.await_args
        assert call is not None
        prompt = call.kwargs["system_prompt"]
        assert "Продолжение живого разговора" in prompt
        assert "Улыбка" in prompt and "радость: выраженный фон" in prompt
        assert "игривость: заметный фон" in prompt
        assert any(turn.proactive for turn in call.kwargs["history"])
        reply_model.chat.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,label,reply",
    [
        ("sadness", "грусть", "Я сегодня без настроения, но поболтать хочется."),
        ("joy", "радость", "Настроение отличное. Давай обсудим космос!"),
        ("tiredness", "усталость", "Я немного устал. Можно просто спокойно поболтать."),
    ],
)
def test_proactive_text_receives_mood_without_user_blame_or_mechanical_reply(
    tmp_path: Path, field: str, label: str, reply: str
) -> None:
    async def run() -> None:
        now = [100.0]
        repo, states, bot, model, service = _build(tmp_path, now, value=0.99)
        async with states.use_conversation(7) as state:
            setattr(state.emotions.memory, field, 0.8)
        model.chat.return_value = reply
        await repo.note_user_activity(7, now[0])
        assert await service.run_once() == 0
        now[0] += 5400
        assert await service.run_once() == 1
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert f"{label}: выраженный фон" in prompt
        assert "Не объясняй механизм инициативы" in prompt
        bot.send_message.assert_awaited_once_with(7, reply)
        assert states.get(7).history[-1].proactive

    asyncio.run(run())


def test_technical_followup_and_other_user_do_not_receive_private_initiative() -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        state = engine._user_states.get(7)
        state.history.append(
            ConversationTurn("", "PRIVATE_INITIATIVE_MARKER", proactive=True)
        )
        await engine.respond(7, "Как работает таймер отправки стикеров?")
        assert (
            "Продолжение живого разговора"
            not in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine.respond(8, "Чего пишешь?")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "PRIVATE_INITIATIVE_MARKER" not in prompt
        assert "Подтверждённой недавней самостоятельной отправки" in prompt

    asyncio.run(run())
