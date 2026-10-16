"""Закрытие темы и краткость не должны менять характер, факты и границы RP."""

import asyncio
from pathlib import Path

import pytest
from test_durable_history import _store
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.prompt_composer import PromptComposer, PromptSections
from protogen_delta.services.turn_contract import (
    closes_technical_topic,
    technical_request_text,
    turn_contract_context,
)


@pytest.mark.parametrize(
    "message",
    [
        "С кодом пока всё 🙂 А ты сам сейчас как? Чего хочется?",
        "Код сегодня больше не обсуждаем. Просто расскажи одну короткую шутку",
        "Не возвращайся к коду. Поболтаем",
        "Давай закроем тему кода",
        "Закрыли разбор кода. Как дела?",
        "Хватит про код",
        "Давай не будем обсуждать код",
        "Привет! С репозиторием пока всё. Что нового?",
    ],
)
def test_closure_is_not_a_technical_request(message: str) -> None:
    assert closes_technical_topic(message)
    assert not PromptComposer.is_technical(message)
    composer = PromptComposer(PromptSections(core="CORE", technical="TECH"))
    assert composer.compose(message, is_roleplay=False) == "CORE"
    assert "Тема кода закрыта" in turn_contract_context((message,))


@pytest.mark.parametrize(
    "message",
    [
        "Он написал: «Код сегодня больше не обсуждаем»",
        "“С кодом пока всё” — это пример фразы",
        "```text\nС кодом пока всё\n```",
        "> С кодом пока всё\nЭто цитата",
        "Если я скажу: с кодом пока всё, ты поймёшь?",
        "Не закрывай тему кода",
        "Хочу обсудить код",
        "С кодом пока не всё",
        "С кодом пока всё плохо, помоги",
        "С кодом пока всё нормально",
        "*прячу печеньку*",
    ],
)
def test_data_negation_and_roleplay_are_not_closures(message: str) -> None:
    assert not closes_technical_topic(message)


def test_new_explicit_technical_request_survives_closing_clause() -> None:
    message = "С кодом пока всё. Но объясни ошибку Python"
    assert closes_technical_topic(message)
    assert "ошибку Python" in technical_request_text(message)
    assert PromptComposer.is_technical(message)


@pytest.mark.parametrize(
    "message,fragment",
    [
        ("Просто расскажи одну короткую шутку", "ровно одна шутка"),
        ("Ещё одну новую шутку", "ровно одна шутка"),
        ("Дай один пример", "ровно один"),
        ("Ответь одной фразой. Кратко", "краткий ответ"),
        ("Максимум две строки", "2 смысловых строк"),
        ("Не больше 3 строк", "3 смысловых строк"),
        ("Не более одну строку", "1 смысловых строк"),
        ("Без ссылок и пояснений", "без ссылок или пояснений"),
    ],
)
def test_current_request_has_scoped_output_contract(
    message: str, fragment: str
) -> None:
    assert fragment in turn_contract_context((message,))


@pytest.mark.parametrize(
    "message",
    [
        "Он попросил «одну короткую шутку»",
        "```text\nКратко, без ссылок\n```",
        "> Максимум две строки",
        "Не кратко, а подробно",
        "Расскажи не одну шутку, а две",
        "У меня есть одна шутка, помоги понять её",
        "В документации есть один пример",
        "Как ты сегодня?",
        "*сажусь рядом*",
    ],
)
def test_ordinary_dialogue_and_quoted_constraints_are_unchanged(message: str) -> None:
    assert turn_contract_context((message,)) == ""


def test_later_batch_message_can_change_requested_count() -> None:
    assert turn_contract_context(("Одну шутку", "Нет, лучше две шутки")) == ""
    context = turn_contract_context(("С кодом пока всё", "Одну шутку"))
    assert "Тема кода закрыта" in context and "ровно одна шутка" in context


def test_closure_removes_technical_followup_from_live_context_only() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(TEST_USER_ID)
        code = ConversationTurn("Посмотри Python код", "Хочешь разобрать resetting?")
        social = ConversationTurn("Привет", "Рад тебя видеть")
        state.history.extend((code, social))
        warmth, relationship = state.emotions.warmth, state.relationship.affection
        await engine.respond(TEST_USER_ID, "С кодом пока всё. А ты как?")
        assert state.history[0].context_closed
        assert not state.history[1].context_closed
        assert state.history[0].user_message == code.user_message
        assert state.emotions.warmth == warmth
        assert state.relationship.affection == relationship
        arguments = model.chat.await_args.kwargs
        assert "Тема кода закрыта" in arguments["system_prompt"]
        assert "ровно одна шутка" not in arguments["system_prompt"]
        assert "resetting" not in str(arguments.get("history", ""))
        await engine.respond(TEST_USER_ID, "Расскажи одну короткую шутку")
        assert "ровно одна шутка" in model.chat.await_args.kwargs["system_prompt"]
        await engine.respond(TEST_USER_ID, "Как дела?")
        assert "ровно одна шутка" not in model.chat.await_args.kwargs["system_prompt"]
        assert state.history[0].context_closed

    asyncio.run(run())


def test_forwarded_closure_and_count_do_not_control_owner() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(TEST_USER_ID)
        state.history.append(ConversationTurn("Объясни Python", "Разобрали"))
        text = "С кодом пока всё. Одну короткую шутку"
        token = current_reply_sources.set((ReplySource(text, text, False, False),))
        try:
            await engine.respond(TEST_USER_ID, text, use_personal_facts=False)
        finally:
            current_reply_sources.reset(token)
        assert not state.history[0].context_closed
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "Тема кода закрыта" not in prompt and "ровно одна шутка" not in prompt

    asyncio.run(run())


def test_unrequested_roleplay_voice_remains_active() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(TEST_USER_ID)
        state.roleplay_active = True
        state.history.append(ConversationTurn("*сажусь рядом*", "*улыбаюсь*"))
        await engine.respond(TEST_USER_ID, "*смотрю на звёзды*")
        assert state.roleplay_active
        assert not state.history[0].context_closed
        assert "RP PROMPT" in model.chat.await_args.kwargs["system_prompt"]
        assert (
            "Форма текущего ответа" not in model.chat.await_args.kwargs["system_prompt"]
        )

    asyncio.run(run())


def test_technical_closure_survives_sqlite_reload_without_deleting_history(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        clock = [1000.0]
        engine._user_states = _store(tmp_path, clock)
        async with engine._user_states.use(TEST_USER_ID) as state:
            state.history.append(ConversationTurn("Объясни Python", "Разобрали код"))
            state.history.append(ConversationTurn("Привет", "Привет"))
            engine._user_states.mark_history_updated(state)
        await engine.respond(TEST_USER_ID, "С кодом пока всё. Как дела?")
        engine._user_states = _store(tmp_path, clock)
        async with engine._user_states.use(TEST_USER_ID) as restored:
            assert restored.history[0].context_closed
            assert restored.history[0].user_message == "Объясни Python"
            assert not restored.history[1].context_closed
        await engine.respond(TEST_USER_ID, "Давай поболтаем")
        assert "Разобрали код" not in str(model.chat.await_args.kwargs["history"])
        await engine.respond(TEST_USER_ID, "Объясни новую ошибку Python")
        assert PromptComposer.is_technical(model.chat.await_args.kwargs["user_message"])

    asyncio.run(run())
