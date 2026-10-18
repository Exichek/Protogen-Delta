"""Текущий сайт, несколько источников и исправление опечатки без смены темы."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from test_deepseek_service import _create_service
from test_response_engine import TEST_USER_ID, _create_engine
from test_tool_loop import response

from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.tools import Tool, ToolExecutor, ToolRegistry
from protogen_delta.services.web_followup import resolve_web_followup

OLD = "https://api-docs.deepseek.com/api/search/"
COD = "https://www.callofduty.com/blog/2026/10/news"
NEWS = "https://example.org/news"
OLD_TURN = ConversationTurn("Изучи " + OLD, "Разобрал параметры API")
GAME = ConversationTurn(
    "Найди последнюю инфу о MW4", f"Источники: [CoD]({COD}), [Новости]({NEWS})."
)
QUESTION = "Что еще на сайте интересного видишь? Анализиуйц"


def test_multiple_current_sources_are_clarified_without_old_topic() -> None:
    result = resolve_web_followup(QUESTION, (OLD_TURN, GAME))
    assert result is not None and not result.urls
    assert COD in result.question and NEWS in result.question
    assert OLD not in result.question


@pytest.mark.parametrize("message", ["Анализируй*", "проанализируй*", "Разбери*"])
def test_typo_correction_keeps_original_reference_even_after_wrong_answer(
    message: str,
) -> None:
    wrong = ConversationTurn(QUESTION, "Я читал " + OLD)
    result = resolve_web_followup(message, (OLD_TURN, GAME, wrong))
    assert result is not None and COD in result.question and OLD not in result.question


@pytest.mark.parametrize(
    "message",
    [
        "Что еще на этом сайте?",
        "Что на странице интересного?",
        "Анализируй",
        "Прочитай",
    ],
)
def test_single_current_source_is_explicitly_read(message: str) -> None:
    game = replace(GAME, assistant_message=f"Источник: {COD}")
    result = resolve_web_followup(message, (OLD_TURN, game))
    assert result is not None and result.urls == (COD,) and not result.question
    assert "не доказывают" in result.model_message(message)
    assert OLD not in result.model_message(message)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("первый", (COD,)),
        ("2", (NEWS,)),
        ("example.org", (NEWS,)),
        ("Все", (COD, NEWS)),
    ],
)
def test_choice_after_clarification_uses_offered_sources(
    message: str, expected: tuple[str, ...]
) -> None:
    question = resolve_web_followup(QUESTION, (GAME,))
    assert question is not None
    clarification = ConversationTurn(QUESTION, question.question)
    result = resolve_web_followup(message, (OLD_TURN, GAME, clarification))
    assert result is not None and result.urls == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Разбери все источники", (COD, NEWS)),
        ("Что на первом сайте?", (COD,)),
        ("Что на сайте example.org?", (NEWS,)),
    ],
)
def test_explicit_source_selection(message: str, expected: tuple[str, ...]) -> None:
    result = resolve_web_followup(message, (OLD_TURN, GAME))
    assert result is not None and result.urls == expected


@pytest.mark.parametrize(
    "message",
    [
        "Привет",
        "*обнимаю*",
        "Что такое сайты?",
        "Не читай этот сайт",
        "Без интернета, что на сайте?",
        "Он сказал «Что ещё на сайте?»",
        "```Что на сайте?```",
        "Открой https://new.example/page",
        "Что на сайте? " + "x" * 300,
    ],
)
def test_unrelated_or_explicit_new_requests_are_not_rewritten(message: str) -> None:
    assert resolve_web_followup(message, (OLD_TURN, GAME)) is None


@pytest.mark.parametrize(
    "barrier",
    [
        replace(GAME, context_closed=True),
        replace(GAME, proactive=True),
        ConversationTurn("Давай про котов", "Пушистые"),
    ],
)
def test_followup_never_searches_past_latest_topic(barrier: ConversationTurn) -> None:
    result = resolve_web_followup(QUESTION, (OLD_TURN, barrier))
    assert result is None or OLD not in result.question and OLD not in result.urls


def test_latest_user_url_has_priority_over_assistant_suggestions() -> None:
    turn = ConversationTurn("Посмотри " + COD, "Ещё есть " + OLD)
    result = resolve_web_followup(QUESTION, (turn,))
    assert result is not None and result.urls == (COD,)


def test_unavailable_ordinal_does_not_silently_read_another_page() -> None:
    result = resolve_web_followup(
        "Что на втором сайте?", (replace(GAME, assistant_message=COD),)
    )
    assert result is not None and not result.urls and COD in result.question


def test_missing_page_asks_for_link_without_claiming_to_have_read_it() -> None:
    result = resolve_web_followup(
        QUESTION, (ConversationTurn("Про игру", "Интересно"),)
    )
    assert result is not None and not result.urls and "Пришли ссылку" in result.question
    assert resolve_web_followup("Анализируй*", ()) is None
    assert (
        resolve_web_followup("Анализируй*", (ConversationTurn("Привет", OLD),)) is None
    )


def test_engine_clarifies_then_reads_selected_site_and_preserves_saved_input() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(TEST_USER_ID)
        state.history.extend((OLD_TURN, GAME))
        answer = await engine.respond(TEST_USER_ID, QUESTION)
        assert answer.startswith("Какой сайт разобрать?") and OLD not in answer
        model.chat.assert_not_awaited()
        await engine.respond(TEST_USER_ID, "1")
        kwargs = model.chat.await_args.kwargs
        assert COD in kwargs["user_message"] and OLD not in kwargs["user_message"]
        assert kwargs["tool_names"] == {"fetch_web_page"}
        assert state.history[-1].user_message == "1"

    asyncio.run(run())


def test_engine_correction_does_not_adopt_previous_wrong_site() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(TEST_USER_ID)
        state.history.extend(
            (
                OLD_TURN,
                replace(GAME, assistant_message=COD),
                ConversationTurn(QUESTION, "Я читал " + OLD),
            )
        )
        await engine.respond(TEST_USER_ID, "Анализируй*")
        kwargs = model.chat.await_args.kwargs
        assert COD in kwargs["user_message"] and OLD not in kwargs["user_message"]
        assert kwargs["tool_names"] == {"fetch_web_page"}

    asyncio.run(run())


def test_repeated_corrections_do_not_reanchor_to_wrong_answers() -> None:
    history = (
        GAME,
        ConversationTurn(QUESTION, OLD),
        ConversationTurn("Анализируй*", OLD),
    )
    result = resolve_web_followup("Анализируй*", history)
    assert result is not None and COD in result.question and OLD not in result.question
    for previous in ("«Что на сайте?»", "Не читай этот сайт"):
        assert (
            resolve_web_followup("Прочитай*", (GAME, ConversationTurn(previous, OLD)))
            is None
        )


def test_selected_page_is_actually_fetched_before_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, create, _ = _create_service(monkeypatch)
    page = AsyncMock(return_value={"url": COD, "text": "CURRENT_GAME_PAGE"})
    old_page = AsyncMock()
    service._tools = ToolExecutor(
        ToolRegistry(
            [
                Tool("fetch_web_page", "read", {"url": "url"}, ("url",), page),
                Tool("other", "other", {}, (), old_page),
            ]
        )
    )
    call = response(1, json.dumps({"url": COD}))
    call.choices[0].message.tool_calls[0].function.name = "fetch_web_page"
    create.side_effect = [call, response()]
    result = resolve_web_followup(
        QUESTION, (OLD_TURN, replace(GAME, assistant_message=COD))
    )
    assert result is not None
    asyncio.run(
        service.chat(
            "system",
            result.model_message(QUESTION),
            history=(OLD_TURN, GAME),
            tool_names={"fetch_web_page"},
        )
    )
    page.assert_awaited_once_with({"url": COD})
    old_page.assert_not_awaited()
    assert (
        create.await_args_list[0].kwargs["tool_choice"]["function"]["name"]
        == "fetch_web_page"
    )
    assert create.await_args is not None
    assert "CURRENT_GAME_PAGE" in str(create.await_args.kwargs["messages"])


@pytest.mark.parametrize("scope", ["rp", "attachment", "override", "other_user"])
def test_engine_does_not_reuse_page_in_unrelated_context(scope: str) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(TEST_USER_ID)
        state.history.append(replace(GAME, assistant_message=COD))
        owner = TEST_USER_ID
        options: dict[str, object] = {}
        if scope == "rp":
            state.roleplay_active = True
        elif scope == "attachment":
            options["attachment_text"] = "Текст документа"
        elif scope == "override":
            options["model_message_override"] = "Данные вложения"
        else:
            owner += 1
        await engine.respond(owner, QUESTION, **options)  # type: ignore[arg-type]
        assert COD not in model.chat.await_args.kwargs["user_message"]

    asyncio.run(run())
