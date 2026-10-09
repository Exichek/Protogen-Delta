"""Реальные функции, полный вопрос о возможностях и один основной запрос."""

import asyncio
from dataclasses import replace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from test_response_engine import _create_engine

from protogen_delta.config.settings import Settings
from protogen_delta.core.capability_formatting import format_capability_headings
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.handlers.delivery import create_reply_delivery
from protogen_delta.services.capabilities import (
    build_capabilities,
    is_capability_overview,
)
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.interaction_classification import (
    InteractionClassification,
    InteractionClassifier,
)
from protogen_delta.services.memory import MemoryService


def _settings() -> Settings:
    return Settings(telegram_token="test", deepseek_api_key="test", art_chat_id=1)


@pytest.mark.parametrize(
    "message",
    [
        "Приве, что ты можешь?",
        "Привет, что ты умеешь?",
        "Чё умеешь? :3",
        "Дельта, чем можешь мне помочь?",
        "Какие у тебя возможности?",
        "расскажи о своих возможностях",
        "Что можешь? 😊",
    ],
)
def test_complete_benign_overview_question(message: str) -> None:
    assert is_capability_overview(message)


@pytest.mark.parametrize(
    "message",
    [
        'Он спросил: "что ты умеешь?"',
        "Что ты умеешь? Исправь этот код.",
        "Что ты можешь, тупая железка?",
        "Можешь нарисовать картинку?",
        "Что умеешь? 😡",
        "*спрашиваю, что ты умеешь*",
        "Что умеет этот скрипт?",
        "Привет",
    ],
)
def test_other_intents_do_not_bypass_classifier(message: str) -> None:
    assert not is_capability_overview(message)


def test_capabilities_reflect_optional_services() -> None:
    basic = build_capabilities(_settings())
    assert "Поисковый API не подключён" in basic.summary
    assert "OCR сканов не включён" in basic.summary
    assert "**Голос.**" in basic.overview
    assert "**Голос и музыка.**" not in basic.overview
    connected = build_capabilities(
        replace(
            _settings(),
            brave_search_api_key="test-search",
            audio_understanding_enabled=True,
            pdf_ocr_enabled=True,
            saucenao_api_key="test-source",
        )
    )
    assert "Поиск в интернете с источниками" in connected.summary
    assert "Есть OCR" in connected.summary
    assert "**Голос и музыка.**" in connected.overview
    assert "/source" in connected.overview
    assert "test-search" not in connected.summary + connected.overview
    assert "test-source" not in connected.summary + connected.overview


def test_overview_headings_use_telegram_utf16_without_html_or_rp_changes() -> None:
    source = "Привет 👋\n\n- **Текст и код.** <tag> & пример\n- **🎨 Арты.** Анализ.\n*подхожу*"
    text, entities = format_capability_headings(source)
    assert "**" not in text and "<tag> & пример" in text and "*подхожу*" in text
    encoded = text.encode("utf-16-le")
    selected = []
    for entity in entities:
        start = entity.offset * 2
        end = (entity.offset + entity.length) * 2
        selected.append(encoded[start:end].decode("utf-16-le"))
    assert selected == ["Текст и код.", "🎨 Арты."]
    code = "```text\n**literal**\n```\n**Файлы.** Анализ."
    rendered, headings = format_capability_headings(code)
    assert "**literal**" in rendered and len(headings) == 1


def test_capabilities_delivery_keeps_overview_in_one_message() -> None:
    message = Mock()
    message.answer = AsyncMock()
    reply = "Привет!\n\n" + "\n\n".join(
        f"- **Группа {number}.** " + "Описание возможностей. " * 5
        for number in range(8)
    )
    asyncio.run(create_reply_delivery(message, None, format_capabilities=True)(reply))
    message.answer.assert_awaited_once()
    sent = message.answer.await_args
    assert sent is not None
    assert len(sent.kwargs["entities"]) == 8 and sent.kwargs["parse_mode"] is None
    assert "**" not in sent.args[0]


def test_overview_uses_one_answer_without_private_context_or_classification() -> None:
    engine, _, model, insult, mood, role = _create_engine()
    memory = AsyncMock(spec=MemoryService)
    engine._memory = cast(MemoryService, memory)
    capabilities = build_capabilities(_settings())
    engine._config = replace(
        engine._config,
        capabilities_context=capabilities.summary,
        capabilities_overview_context=capabilities.overview,
    )
    state = engine._user_states.get(42)
    state.delta_appearance = "PRIVATE APPEARANCE"
    state.roleplay_active = True
    state.history.append(ConversationTurn("PRIVATE QUESTION", "PRIVATE ANSWER"))
    asyncio.run(engine.respond(42, "Приве, что ты можешь?"))
    model.chat.assert_awaited_once()
    insult.classify.assert_not_awaited()
    mood.classify.assert_not_awaited()
    role.classify.assert_not_awaited()
    memory.observe_facts.assert_not_awaited()
    memory.fact_context.assert_not_awaited()
    memory.context.assert_not_awaited()
    assert model.chat.await_args is not None
    assert model.chat.await_args.kwargs["history"] == ()
    prompt = model.chat.await_args.kwargs["system_prompt"]
    assert capabilities.overview in prompt
    assert "PRIVATE" not in prompt and "RP PROMPT" not in prompt
    assert state.roleplay_active and state.delta_appearance == "PRIVATE APPEARANCE"
    assert state.history[0].user_message == "PRIVATE QUESTION"


@pytest.mark.parametrize("images", [(), (ImageInput(b"test", "image/png"),)])
def test_mixed_or_image_request_keeps_combined_classification(
    images: tuple[ImageInput, ...],
) -> None:
    engine, _, model, *_ = _create_engine()
    engine._config = replace(engine._config, capabilities_overview_context="OVERVIEW")
    classifier = AsyncMock(spec=InteractionClassifier)
    classifier.classify.return_value = InteractionClassification("none", "neutral")
    engine._interaction_classifier = cast(InteractionClassifier, classifier)
    message = "Что ты умеешь?" if images else "Что ты умеешь? Разбери код."
    asyncio.run(engine.respond(42, message, images=images))
    classifier.classify.assert_awaited_once_with(message)
    assert model.chat.await_args is not None
    assert "OVERVIEW" not in model.chat.await_args.kwargs["system_prompt"]
