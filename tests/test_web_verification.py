"""Не выдавать уверенный веб-черновик без проверяемой опоры на прочитанный текст."""

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_tool_loop import response

import protogen_delta.services.deepseek as module
from protogen_delta.core.llm_budget import llm_user
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.repositories.llm_usage import LlmUsageRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.prompt_composer import PromptComposer
from protogen_delta.services.tools import Tool, ToolExecutor, ToolRegistry
from protogen_delta.services.web_verification import UNVERIFIED_WEB_REPLY, WebEvidence

URL = "https://example.org/research"
QUOTE = "This experiment studied African cichlids, not goldfish."


def _evidence() -> WebEvidence:
    evidence = WebEvidence()
    evidence.observe(
        "fetch_web_page",
        json.dumps({"url": URL}),
        json.dumps({"ok": True, "data": {"text": QUOTE, "checked_at": "2026-10-20"}}),
    )
    return evidence


def _audit(
    answer: str = "Исследовали цихлид. " + URL, quote: str = QUOTE, url: str = URL
) -> str:
    return json.dumps({"answer": answer, "evidence": [{"url": url, "quote": quote}]})


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "{}",
        _audit(url="https://invented.org"),
        _audit(quote="A fabricated claim about goldfish memory."),
        _audit(answer="Уверенный ответ без ссылки"),
        _audit(answer="Ссылка подменена: https://invented.org"),
        _audit(quote="short"),
        json.dumps({"answer": "текст", "evidence": []}),
        json.dumps({"answer": 5, "evidence": []}),
    ],
)
def test_evidence_rejects_unsupported_verification(raw: str) -> None:
    assert _evidence().accept(raw) == UNVERIFIED_WEB_REPLY


def test_matching_source_and_literal_quote_are_required() -> None:
    evidence = _evidence()
    assert evidence.accept(_audit()) == "Исследовали цихлид. " + URL
    assert QUOTE not in evidence.accept(_audit())
    assert (
        evidence.accept(
            _audit(quote="This experiment\n studied African cichlids, not goldfish.")
        )
        != UNVERIFIED_WEB_REPLY
    )


def test_quote_cannot_change_case_of_scientific_units() -> None:
    original = "The measured power was 5 mW under test conditions."
    evidence = WebEvidence()
    evidence.observe(
        "fetch_web_page",
        json.dumps({"url": URL}),
        json.dumps({"ok": True, "data": {"text": original}}),
    )
    assert (
        evidence.accept(_audit(quote=original.replace("mW", "MW")))
        == UNVERIFIED_WEB_REPLY
    )


def test_balanced_parentheses_in_source_url_are_preserved() -> None:
    url = "https://example.org/article_(2026)"
    evidence = WebEvidence()
    evidence.observe(
        "fetch_web_page",
        json.dumps({"url": url}),
        json.dumps({"ok": True, "data": {"text": QUOTE}}),
    )
    assert (
        evidence.accept(_audit(answer=f"Прочитано [здесь]({url}).", url=url))
        != UNVERIFIED_WEB_REPLY
    )


@pytest.mark.parametrize(
    "name,result",
    [
        ("read_source", {"ok": True, "data": {"text": QUOTE}}),
        ("fetch_web_page", {"ok": False}),
        ("fetch_web_page", {"ok": True, "data": {"available": False, "text": QUOTE}}),
        (
            "fetch_web_page",
            {"ok": True, "data": {"content_scope": "metadata_only", "text": QUOTE}},
        ),
        ("fetch_web_page", {"ok": True, "data": {"text": ""}}),
    ],
)
def test_unreadable_or_private_result_does_not_become_web_evidence(
    name: str, result: dict
) -> None:
    evidence = WebEvidence()
    evidence.observe(name, json.dumps({"url": URL}), json.dumps(result))
    assert not evidence.pages


@pytest.mark.parametrize(
    "question",
    [
        "Shazam умеет узнавать песню, если я её напою?",
        "Умеет ли сервис узнавать напев?",
        "Правда, что золотые рыбки помнят три секунды?",
        "Что такое эффект Мпембы и насколько он доказан?",
        "Какая сейчас последняя стабильная версия Python?",
    ],
)
def test_factual_checks_request_search_without_waiting_for_model_confidence(
    question: str,
) -> None:
    assert "web_search" in PromptComposer.select_tools(question)


def _tool_reply(name: str, arguments: dict) -> SimpleNamespace:
    result = response(1, json.dumps(arguments))
    result.choices[0].message.tool_calls[0].function.name = name
    return result


def test_live_loop_reads_source_then_checks_species_and_rewrites_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = AsyncMock(
        return_value={"results": [{"url": URL, "description": "Fish remember 48 days"}]}
    )
    read = AsyncMock(return_value={"text": QUOTE})
    draft = response()
    draft.choices[0].message.content = "Золотые рыбки помнят минимум сорок восемь дней."
    audited = response()
    audited.choices[0].message.content = _audit()
    client = Mock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _tool_reply("web_search", {"query": "goldfish memory experiment"}),
            _tool_reply("fetch_web_page", {"url": URL}),
            draft,
            audited,
        ]
    )
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "key",
        "https://test.local",
        "test",
        verify_web_answers=True,
        tools=ToolExecutor(
            ToolRegistry(
                [
                    Tool(
                        "web_search", "search", {"query": "query"}, ("query",), search
                    ),
                    Tool("fetch_web_page", "read", {"url": "url"}, ("url",), read),
                ]
            )
        ),
    )
    question = "Правда, что золотые рыбки помнят три секунды?"
    result = asyncio.run(
        service.chat(
            "persona",
            question,
            history=[ConversationTurn("PRIVATE OLD TEXT", "private answer")],
            tool_names=PromptComposer.select_tools(question),
        )
    )
    assert result == "Исследовали цихлид. " + URL
    search.assert_awaited_once()
    read.assert_awaited_once()
    calls = client.chat.completions.create.await_args_list
    assert len(calls) == 4
    assert calls[0].kwargs["tool_choice"]["function"]["name"] == "web_search"
    assert calls[1].kwargs["tool_choice"]["function"]["name"] == "fetch_web_page"
    audit = calls[-1].kwargs
    assert "tools" not in audit
    assert audit["response_format"] == {"type": "json_object"}
    material = json.loads(audit["messages"][1]["content"])
    assert material["pages"][0]["text"] == QUOTE
    assert "PRIVATE OLD TEXT" not in json.dumps(audit["messages"])


@pytest.mark.parametrize("result", [{"results": []}, {"results": [{"url": URL}]}])
def test_search_snippets_cannot_be_returned_as_verified_answer(
    monkeypatch: pytest.MonkeyPatch, result: dict
) -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        side_effect=[_tool_reply("web_search", {"query": "test query"}), response()]
    )
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "key",
        "https://test.local",
        "test",
        verify_web_answers=True,
        tools=ToolExecutor(
            ToolRegistry(
                [
                    Tool(
                        "web_search",
                        "search",
                        {"query": "query"},
                        ("query",),
                        AsyncMock(return_value=result),
                    ),
                    Tool(
                        "fetch_web_page", "read", {"url": "url"}, ("url",), AsyncMock()
                    ),
                ]
            )
        ),
    )
    assert (
        asyncio.run(
            service.chat(
                "system",
                "Найди информацию о версии",
                tool_names={"web_search", "fetch_web_page"},
            )
        )
        == UNVERIFIED_WEB_REPLY
    )
    assert client.chat.completions.create.await_count == 2


def test_optional_greeting_does_not_add_audit_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=response())
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "key",
        "https://test.local",
        "test",
        verify_web_answers=True,
        tools=ToolExecutor(
            ToolRegistry([Tool("web_search", "search", {}, (), AsyncMock())])
        ),
    )
    assert (
        asyncio.run(service.chat("system", "Привет", tool_names={"web_search"}))
        == "Итог"
    )
    assert client.chat.completions.create.await_count == 1


def test_missing_required_search_fails_closed_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=response())
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "key",
        "https://test.local",
        "test",
        verify_web_answers=True,
        tools=ToolExecutor(ToolRegistry([])),
    )
    assert (
        asyncio.run(
            service.chat("system", "Найди последнюю версию", tool_names={"web_search"})
        )
        == UNVERIFIED_WEB_REPLY
    )
    client.chat.completions.create.assert_not_awaited()


def test_supplement_and_audit_use_normal_llm_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = Mock()
    client.max_retries = 0
    client.with_options.return_value = client
    audit = response()
    audit.choices[0].message.content = _audit()
    addition = response()
    addition.choices[0].message.content = '{"text":""}'
    client.chat.completions.create = AsyncMock(side_effect=[audit, addition])
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "key", "https://test.local", "test", verify_web_answers=True
    )
    service.set_usage_repository(LlmUsageRepository(tmp_path))

    async def run() -> None:
        with llm_user(123):
            assert (
                await service._verified_web_reply(
                    "question", "draft", _evidence(), True
                )
                != UNVERIFIED_WEB_REPLY
            )
            assert await service.draft_supplement("system", "question") == '{"text":""}'

    asyncio.run(run())
    with closing(sqlite3.connect(tmp_path / "llm_usage.db")) as db:
        assert db.execute(
            "SELECT requests FROM llm_daily WHERE user_id=123"
        ).fetchone() == (2,)
    assert all(
        "tools" not in call.kwargs
        for call in client.chat.completions.create.await_args_list
    )
