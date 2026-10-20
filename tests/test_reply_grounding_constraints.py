"""Просьба о себе и числе предложений не должна менять обычный RP."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from test_proactive_roleplay import messenger
from test_response_engine import _create_engine

from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.user_state import UserState, UserStateStore
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.services.conversation_grounding import self_status_context
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.turn_contract import turn_contract_context


@pytest.mark.parametrize(
    "text",
    [
        "А как ты сам?",
        "А ты сам сейчас как?",
        "Как ты сейчас?",
        "Как дела?",
        "Как у тебя дела?",
        "Как ты себя чувствуешь?",
        "Как прошёл твой день?",
        "Как проходит у тебя день?",
        "А что ты сегодня делал до нашего разговора? Коротко.",
        "Чем ты занимался вчера?",
    ],
)
def test_self_question_gets_current_emotions_without_invented_day(text: str) -> None:
    state = UserState()
    state.emotions.memory.sadness = 0.8
    context = self_status_context((text,), state, is_roleplay=False)
    assert "грусть: выраженный фон" in context
    assert "Не описывай целый день" in context
    assert "без технических оговорок" in context
    assert self_status_context((text,), state, is_roleplay=True) == ""


@pytest.mark.parametrize(
    "text",
    [
        "Как ты написал этот код?",
        "У меня был спокойный день",
        "*Как ты сам?*",
        "«Как прошёл твой день?» — пример",
        "> Как ты сам?",
        "«Что ты сегодня делал?» — реплика героя",
        "Что ты сделал с этим кодом?",
        "Привет",
    ],
)
def test_other_questions_and_quoted_examples_do_not_get_self_context(text: str) -> None:
    # Ремарка не является запросом биографии; в рабочем пути она включает RP.
    rp = text.startswith("*")
    assert self_status_context((text,), UserState(), is_roleplay=rp) == ""


@pytest.mark.parametrize(
    "text,count",
    [
        ("Ответь двумя короткими предложениями.", 2),
        ("Ответь одним предложением", 1),
        ("Уложись в три предложения", 3),
        ("Напиши 2 небольших предложения", 2),
    ],
)
def test_count_includes_actions(text: str, count: int) -> None:
    context = turn_contract_context((text,))
    assert f"Ровно {count} предложений всего" in context
    assert "включая ремарки действий" in context


@pytest.mark.parametrize(
    "text",
    [
        "Он сказал «Ответь двумя предложениями»",
        "> Ответь двумя предложениями",
        "Не отвечай двумя предложениями",
        "В документе два предложения",
        "Два предложения были лишними",
    ],
)
def test_quoted_and_negated_count_is_not_an_instruction(text: str) -> None:
    assert "предложений всего" not in turn_contract_context((text,))


def test_forwarded_self_question_is_not_trusted_context() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        text = "Как ты сам?"
        token = current_reply_sources.set((ReplySource(text, text, False, False),))
        try:
            await engine.respond(42, text)
        finally:
            current_reply_sources.reset(token)
        assert (
            "Текущий ответ о себе" not in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine.respond(42, text)
        assert "Текущий ответ о себе" in model.chat.await_args.kwargs["system_prompt"]

    asyncio.run(run())


def test_proactive_formats_sent_text_but_retains_history_source(tmp_path: Path) -> None:
    async def run() -> None:
        repository, states = MemoriesRepository(tmp_path), UserStateStore()
        bot, model = AsyncMock(spec=Bot), AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "*Киваю.* Я рядом."
        await repository.note_user_activity(7, 10)
        assert await messenger(repository, states, bot, model).run_once() == 1
        assert bot.send_message.await_args.args == (7, "Киваю. Я рядом.")
        assert bot.send_message.await_args.kwargs["entities"][0].type == "italic"
        assert states.get(7).history[-1].assistant_message == "*Киваю.* Я рядом."

    asyncio.run(run())
