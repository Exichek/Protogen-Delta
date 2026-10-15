"""Не превращать отсутствие сведений или другую доставку в подтверждённую инициативу."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_creator_memory_and_profile import _service as creator_service
from test_input_batches import Harness, message
from test_response_engine import _create_engine
from test_source_access import source

from protogen_delta.core.user_state import ConversationTurn, UserState
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.code_reviews import CodeReviewsRepository
from protogen_delta.services.code_review import CodeReviewService
from protogen_delta.services.creator_messages import CreatorMessageService
from protogen_delta.services.initiative_context import (
    InitiativeEvidence,
    initiative_reply_context,
)


def test_missing_evidence_does_not_presuppose_or_deny_prior_delivery() -> None:
    state = UserState()
    state.emotions.memory.sadness = 0.8
    state.history.append(ConversationTurn("вопрос", "Обычный ответ"))
    context = initiative_reply_context("Зачем написал?", state)
    facts = json.loads(InitiativeEvidence.from_state(state).context().split(": ", 1)[1])
    assert facts == {
        "delivery_status": "not_in_context",
        "messages": [],
        "past_motive": None,
        "emotion_scope": "current_reply_only",
    }
    assert "Продолжение живого разговора после инициативы" not in context
    assert "Это не доказывает" in context and "предложить поболтать" in context
    assert "грусть: выраженный фон" in context
    assert "что хотел сказать" not in context


def test_delivery_confirms_words_but_not_cause_or_previous_mood() -> None:
    state = UserState()
    state.history.append(ConversationTurn("", "Хочу обсудить игру", proactive=True))
    state.emotions.memory.joy = 0.8
    context = initiative_reply_context("Чего хотел?", state)
    facts = json.loads(InitiativeEvidence.from_state(state).context().split(": ", 1)[1])
    assert facts["delivery_status"] == "confirmed" and facts["messages"] == [
        "Хочу обсудить игру"
    ]
    assert (
        facts["past_motive"] is None and facts["emotion_scope"] == "current_reply_only"
    )
    assert (
        "радость: выраженный фон" in context and "не объясняют задним числом" in context
    )
    assert "Примеры голоса и сцен" in context


def test_closed_distant_and_oversized_history_keeps_evidence_bounded() -> None:
    state = UserState()
    state.history.append(
        ConversationTurn("", "CLOSED", proactive=True, context_closed=True)
    )
    for index in range(4):
        state.history.append(ConversationTurn("", str(index) * 900, proactive=True))
    evidence = InitiativeEvidence.from_state(state)
    assert len(evidence.messages) == 2 and all(
        len(text) == 500 for text in evidence.messages
    )
    assert "CLOSED" not in evidence.context()
    for _ in range(4):
        state.history.append(ConversationTurn("вопрос", "ответ"))
    assert InitiativeEvidence.from_state(state).messages == ()


def test_code_review_overrides_old_proactive_hint_without_erasing_history(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service = CodeReviewService(
            AsyncMock(),
            AsyncMock(),
            source(tmp_path),
            CodeReviewsRepository(tmp_path),
            clock=lambda: 10000,
        )
        await service.record_delivery(10, "REVIEW_MARKER", automatic=True)
        engine, _, model, *_ = _create_engine()
        engine._code_reviews = service
        engine._user_states.get(10).history.append(
            ConversationTurn("", "OLD_INITIATIVE", proactive=True)
        )
        await engine.respond(10, "Чего хотел?")
        request = model.chat.await_args.kwargs
        assert "REVIEW_MARKER" in request["system_prompt"]
        assert "Данные приложения об инициативе" not in request["system_prompt"]
        assert any(
            turn.assistant_message == "OLD_INITIATIVE" for turn in request["history"]
        )

    asyncio.run(run())


def test_creator_delivery_keeps_actual_authorship_over_old_initiative(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service, _, _, states = creator_service(tmp_path)
        await service.send(456, "CREATOR_MARKER", "message")
        async with states.use_conversation(456) as state:
            state.history.append(ConversationTurn("", "OLD_INITIATIVE", proactive=True))
        engine, _, model, *_ = _create_engine()
        engine._user_states = states
        engine._creator_messages = service
        await engine.respond(456, "Что ты прислал?")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "CREATOR_MARKER" in prompt and "автор текста — создатель" in prompt
        assert "Данные приложения об инициативе" not in prompt

    asyncio.run(run())


def test_configured_empty_service_does_not_suppress_known_initiative() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        engine._creator_messages = AsyncMock(spec=CreatorMessageService)
        engine._creator_messages.context.return_value = ""
        engine._user_states.get(7).history.append(
            ConversationTurn("", "KNOWN_INITIATIVE", proactive=True)
        )
        await engine.respond(
            7, "Что хотел? Подтверждённая отправка — просто мои слова."
        )
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "Данные приложения об инициативе" in prompt
        assert '"delivery_status": "confirmed"' in prompt

    asyncio.run(run())


@pytest.mark.parametrize("batch", [False, True])
def test_explicit_reply_target_suppresses_old_initiative_for_whole_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch: bool
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        h.engine._user_states.get(123456).history.append(
            ConversationTurn("", "OLD_INITIATIVE", proactive=True)
        )
        quoted = message(90, "QUOTE_MARKER")
        try:
            await h.feed(message(1, "Чего хотел?", reply_to_message=quoted))
            if batch:
                await h.feed(message(2, "И поясни подробнее"))
            await h.drain()
            prompt = h.model.chat.await_args.kwargs["system_prompt"]
            assert (
                "QUOTE_MARKER" in prompt
                and "Данные приложения об инициативе" not in prompt
            )
            assert h.model.chat.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_active_roleplay_does_not_get_one_sentence_social_override() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(7)
        state.roleplay_active = True
        state.content_mode = "soft"
        state.history.append(
            ConversationTurn("", "Позвал продолжить сцену", proactive=True)
        )
        await engine.respond(7, "*смотрю на тебя* Чего хотел?")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "RP PROMPT" in prompt and "Данные приложения об инициативе" not in prompt
        assert "Один-два коротких" not in prompt

    asyncio.run(run())


def test_evidence_scenarios_preview_does_not_spend_api_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Шесть сценариев проверяют реальные ветви сборки, но не качество речи заглушки."""
    from test_behavior_evaluation import _service, _settings, evaluation

    from protogen_delta.services.deepseek import DeepSeekService

    transport = AsyncMock(side_effect=AssertionError("Сеть запрещена"))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)
    cases = evaluation.load_cases(
        evaluation.CASES_PATH.with_name("initiative_evidence_cases.json")
    )

    async def run() -> None:
        service = _service(max_calls=12)
        try:
            report = await evaluation.evaluate(
                cases, service, _settings(), output=tmp_path / "evidence.json"
            )
        finally:
            await service.close()
        assert len(report["results"]) == 6 and report["calls"] == 12
        assert all(
            row["checks"] == "passed" and row["quality_review"] == "not_run"
            for row in report["results"]
        )

    asyncio.run(run())
    transport.assert_not_awaited()
