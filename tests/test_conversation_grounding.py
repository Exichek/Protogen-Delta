"""Проверять основания времени/памяти, границу темы и сохранность голоса."""

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from test_durable_history import _store
from test_proactive_roleplay import messenger
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.conversation_grounding import (
    current_time_context,
    recall_grounding_context,
)
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.prompt_composer import PromptComposer


@pytest.mark.parametrize(
    "utc_hour,period,local_date",
    [
        (0, "ночь", "2026-10-16"),
        (3, "утро", "2026-10-16"),
        (9, "день", "2026-10-16"),
        (15, "вечер", "2026-10-16"),
        (21, "ночь", "2026-10-17"),
    ],
)
def test_time_uses_moscow_and_crosses_date_boundary(
    utc_hour: int, period: str, local_date: str
) -> None:
    at = datetime(2026, 10, 16, utc_hour, tzinfo=timezone.utc).timestamp()
    text = current_time_context(at, is_roleplay=False)
    assert f"{local_date} {(utc_hour + 3) % 24:02d}:00" in text
    assert f", {period}." in text
    assert "не установленное местоположение" in text
    assert "ночь сцены" not in text
    assert "не меняй ночь сцены" in current_time_context(at, is_roleplay=True)


@pytest.mark.parametrize(
    "question",
    [
        "Помнишь, как мы играли?",
        "Что мы тогда обсуждали?",
        "Сколько ячеек было в прошлой игре?",
        "Ты придумал это или я говорил?",
    ],
)
def test_recall_keeps_source_authors_and_does_not_turn_samples_into_evidence(
    question: str,
) -> None:
    turns = (
        ConversationTurn("CLOSED", "CLOSED", context_closed=True),
        ConversationTurn("", "PROACTIVE", proactive=True),
        ConversationTurn("У меня шесть ячеек", "Предлагаю седьмую как новую идею"),
    )
    context = recall_grounding_context((question,), turns)
    sources = json.loads(context.split("\n")[-1])
    assert sources == [
        {
            "user_message": turns[-1].user_message,
            "assistant_message": turns[-1].assistant_message,
        }
    ]
    assert "не являются нашей общей историей" in context
    assert "никогда не было" in context and "согласованную фикцию" in context
    assert "CLOSED" not in context and "PROACTIVE" not in context


@pytest.mark.parametrize(
    "text",
    [
        "Привет",
        "*сажусь рядом*",
        "Расскажи новую историю",
        "Он спросил «Помнишь, как мы играли?»",
        "> Помнишь игру?",
        "```\nПомнишь игру?\n```",
        "`Помнишь игру?`",
    ],
)
def test_recall_guard_is_not_applied_to_ordinary_roleplay_or_quoted_data(
    text: str,
) -> None:
    assert recall_grounding_context((text,), ()) == ""


def test_recall_sources_are_bounded_and_absence_is_not_denial() -> None:
    history = tuple(ConversationTurn("я" * 1000, "ты" * 1000) for _ in range(20))
    context = recall_grounding_context(("Помнишь?",), history)
    sources = json.loads(context.split("\n")[-1])
    assert len(sources) == 4
    assert all(len(value) <= 350 for item in sources for value in item.values())
    assert json.loads(recall_grounding_context(("Помнишь?",), ()).split("\n")[-1]) == []


@pytest.mark.parametrize(
    "text",
    [
        "*прячу последнюю печеньку*",
        "У меня свежий кофе",
        "Наш последний разговор был тёплым",
        "*передаю свежую карту*",
    ],
)
def test_fictional_last_object_does_not_enable_web_search(text: str) -> None:
    assert "web_search" not in PromptComposer.select_tools(text)


@pytest.mark.parametrize(
    "text",
    [
        "Найди последнюю стабильную версию Python",
        "Свежие новости рынков",
        "Актуальные курсы валют",
        "Последние котировки акций",
        "Свежие данные по инфляции",
        "Поищи в интернете анекдот",
    ],
)
def test_actual_fresh_information_still_has_web_search(text: str) -> None:
    assert "web_search" in PromptComposer.select_tools(text)


def test_topic_boundary_survives_small_history_restart_and_can_be_reopened(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        engine._user_states = _store(tmp_path, [1000.0], limit=2)
        await engine.respond(TEST_USER_ID, "С кодом пока всё. Как дела?")
        for text in ("Спасибо, код закрыли", "Давай про игры", "Обнимаю"):
            await engine.respond(TEST_USER_ID, text)
            assert (
                engine._user_states.get(TEST_USER_ID).history[-1].technical_topic_closed
            )
            assert (
                "Не упоминай закрытую тему"
                in model.chat.await_args.kwargs["system_prompt"]
            )
        engine._user_states = _store(tmp_path, [1000.0], limit=2)
        await engine.respond(TEST_USER_ID, "Как ты?")
        assert (
            "Не упоминай закрытую тему" in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine.respond(TEST_USER_ID, "Покажи Python код")
        assert (
            not engine._user_states.get(TEST_USER_ID).history[-1].technical_topic_closed
        )
        assert (
            "Не упоминай закрытую тему"
            not in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine.respond(TEST_USER_ID, "Что думаешь?")
        assert (
            "Не упоминай закрытую тему"
            not in model.chat.await_args.kwargs["system_prompt"]
        )

    asyncio.run(run())


def test_forwarded_closure_cannot_persist_a_boundary_and_new_request_can_override_it() -> (
    None
):
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        token = current_reply_sources.set(
            (ReplySource("С кодом пока всё", "С кодом пока всё", False, False),)
        )
        try:
            await engine.respond(TEST_USER_ID, "С кодом пока всё")
        finally:
            current_reply_sources.reset(token)
        state = engine._user_states.get(TEST_USER_ID)
        assert not state.history[-1].technical_topic_closed
        await engine.respond(TEST_USER_ID, "С кодом пока всё. Но объясни ошибку Python")
        assert not state.history[-1].technical_topic_closed
        # Первоначальная просьба закрыть не должна запрещать новую явную задачу.
        assert "ошибку Python" in model.chat.await_args.kwargs["user_message"]

    asyncio.run(run())


def test_runtime_passes_fresh_clock_and_sources_after_unchanged_examples() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        at = [datetime(2026, 10, 16, 11, 50, tzinfo=timezone.utc).timestamp()]
        engine._user_states = UserStateStore(wall_clock=lambda: at[0])
        # Меняем только синтетическую конфигурацию, чтобы увидеть границы образцов.
        engine._prompt_composer._sections = replace(
            engine._prompt_composer._sections, voice="VOICE EXAMPLE"
        )
        state = engine._user_states.get(TEST_USER_ID)
        state.history.append(ConversationTurn("Шесть ячеек", "Хороший вечер"))
        await engine.respond(TEST_USER_ID, "Помнишь, сколько ячеек?")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "2026-10-16 14:50" in prompt and ", день." in prompt
        assert prompt.index("VOICE EXAMPLE") < prompt.index("Часы приложения")
        assert "Шесть ячеек" in prompt and "Источники для вопроса о прошлом" in prompt
        at[0] += 3600
        await engine.respond(TEST_USER_ID, "Привет")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "2026-10-16 15:50" in prompt
        assert "Источники для вопроса о прошлом" not in prompt
        assert model.chat.await_count == 2

    asyncio.run(run())


def test_old_history_without_new_metadata_loads_safely() -> None:
    history = UserStateRepository._decode_history(
        '[{"user":"Привет","assistant":"Привет"}]'
    )
    assert history == (ConversationTurn("Привет", "Привет"),)
    assert not UserStateRepository._decode_history(
        '[{"user":"x","assistant":"y","technical_topic_closed":"true"}]'
    )[0].technical_topic_closed


@pytest.mark.parametrize(
    "text,closed",
    [
        ("Он написал «Объясни Python код»", True),
        ("> Покажи Python код", True),
        ("Спасибо, код закрыли", True),
        ("Спасибо, что код закрыли", True),
        ("Объясни Python код", False),
        ("Вернёмся к коду", False),
        ("Обсудим репозиторий", False),
        ("Что такое Python?", False),
        ("```python\nprint(1)\n```", False),
        ("https://github.com/owner/project", False),
    ],
)
def test_reopening_needs_own_request_not_quote(text: str, closed: bool) -> None:
    history = (ConversationTurn("Закрыли", "Хорошо", technical_topic_closed=True),)
    assert PromptComposer.technical_topic_closed(history, (text,)) is closed


def test_proactive_keeps_boundary_after_its_own_messages_and_gets_time(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repository = MemoriesRepository(tmp_path)
        states = UserStateStore()
        state = states.get(7)
        state.history.extend(
            (
                ConversationTurn("С кодом всё", "Обнимаю", technical_topic_closed=True),
                ConversationTurn("", "Привет", proactive=True),
            )
        )
        bot, model = AsyncMock(spec=Bot), AsyncMock(spec=DeepSeekService)
        model.chat.return_value = "Как настроение?"
        await repository.note_user_activity(7, 10)
        assert await messenger(repository, states, bot, model).run_once() == 1
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "Не упоминай закрытую тему" in prompt
        assert "Часы приложения сейчас" in prompt
        assert PromptComposer.technical_topic_closed(tuple(state.history), ())

    asyncio.run(run())
