"""Все предложенные страницы, частичный результат и общий бюджет чтения."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionMessage
from test_deepseek_service import _create_response, _create_service

from protogen_delta.services.link_review import compact_page_result, review_links
from protogen_delta.services.tools import Tool, ToolExecutor, ToolRegistry


def _plain_response(content: str | None) -> SimpleNamespace:
    response = _create_response(content)
    response.choices[0].message = ChatCompletionMessage.model_validate(
        {"role": "assistant", "content": content}
    )
    return response


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "Посмотри https://example.com/a https://example.com/b",
            ("https://example.com/a", "https://example.com/b"),
        ),
        (
            "https://example.com/a\nhttps://example.com/b",
            ("https://example.com/a", "https://example.com/b"),
        ),
        ("Не открывай https://example.com/a https://example.com/b", ()),
        ("Он написал «прочитай https://example.com/a https://example.com/b»", ()),
        ("Посмотри ```https://example.com/a https://example.com/b```", ()),
        ("Посмотри https://example.com/a https://example.com/a", ()),
        (
            "Нужно проверить документ\n<document_content>"
            "https://example.com/a https://example.com/b</document_content>",
            (),
        ),
    ],
)
def test_multiple_links_require_actual_request_and_exclude_data(
    message: str, expected: tuple[str, ...]
) -> None:
    assert review_links(message) == expected


def test_two_pages_are_parallel_and_one_error_does_not_hide_the_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        service, _, create, _ = _create_service(monkeypatch)
        create.return_value = _plain_response("Сравнение по доступным сведениям")
        arrived = 0
        ready = asyncio.Event()

        async def page(args: dict[str, str]) -> dict[str, object]:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 1)
            if args["url"].endswith("/b"):
                raise ValueError("unavailable")
            return {"url": args["url"], "text": "PAGE_MARKER_1", "title": "Первый"}

        service._tools = ToolExecutor(
            ToolRegistry(
                [Tool("fetch_web_page", "read", {"url": "url"}, ("url",), page)]
            )
        )
        answer = await service.chat(
            "system",
            "Сравни https://example.com/a https://example.com/b",
            tool_names={"fetch_web_page"},
        )
        assert create.await_args is not None
        assert "PAGE_MARKER_1" in str(create.await_args.kwargs["messages"])
        assert "https://example.com/a — получен текст страницы" in answer
        assert "https://example.com/b — не прочитана" in answer
        assert arrived == 2 and create.await_count == 1

    asyncio.run(run())


def test_six_prefetched_pages_use_the_whole_tool_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        service, _, create, _ = _create_service(monkeypatch)
        first = _create_response(None)
        first.choices[0].message = ChatCompletionMessage.model_validate(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Москва"}',
                        },
                    }
                ],
            }
        )
        create.side_effect = [first, _plain_response("Итог")]
        page = AsyncMock(return_value={"text": "PAGE"})
        weather = AsyncMock(return_value={"temp": 10})
        service._tools = ToolExecutor(
            ToolRegistry(
                [
                    Tool("fetch_web_page", "read", {"url": "url"}, ("url",), page),
                    Tool(
                        "get_weather", "weather", {"city": "city"}, ("city",), weather
                    ),
                ]
            )
        )
        answer = await service.chat(
            "system",
            "Прочитай " + " ".join(f"https://example.com/{i}" for i in range(8)),
            tool_names={"fetch_web_page", "get_weather"},
        )
        assert page.await_count == 6
        weather.assert_not_awaited()
        assert "Ещё 2 ссылок не проверены" in answer
        assert create.await_args is not None
        assert "tool_budget_exhausted" in str(create.await_args.kwargs["messages"])
        assert create.await_args_list[0].kwargs["tool_choice"] == "none"

    asyncio.run(run())


@pytest.mark.parametrize("scope", ["metadata_only", "metadata_and_transcript"])
def test_video_evidence_retains_scope_and_marks_truncation(scope: str) -> None:
    raw = json.dumps(
        {
            "ok": True,
            "data": {
                "content_scope": scope,
                "text": "x" * 5000,
                "transcript": "y" * 5000,
                "transcript_automatic": True,
                "transcript_language": "ru",
                "truncated": False,
            },
        }
    )
    compact, status = compact_page_result(raw)
    data = json.loads(compact)["data"]
    assert data["truncated"] and len(data["text"]) == len(data["transcript"]) == 2000
    assert data["transcript_automatic"] is True and data["content_scope"] == scope
    assert (
        "только метаданные" in status
        if scope == "metadata_only"
        else "не видеоряд" in status
    )


def test_normal_read_tool_calls_run_parallel_and_keep_their_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        service, _, create, _ = _create_service(monkeypatch)
        first = _create_response(None)
        first.choices[0].message = ChatCompletionMessage.model_validate(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call{i}",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": str(i)}),
                        },
                    }
                    for i in range(2)
                ],
            }
        )
        create.side_effect = [first, _plain_response("Итог")]
        arrived = 0
        ready = asyncio.Event()

        async def weather(args: dict[str, str]) -> dict[str, str]:
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 1)
            return args

        service._tools = ToolExecutor(
            ToolRegistry(
                [Tool("get_weather", "weather", {"city": "city"}, ("city",), weather)]
            )
        )
        assert (
            await service.chat(
                "system", "Какая погода в двух городах?", tool_names={"get_weather"}
            )
            == "Итог"
        )
        assert create.await_args is not None
        results = [
            m for m in create.await_args.kwargs["messages"] if m["role"] == "tool"
        ]
        assert [m["tool_call_id"] for m in results] == ["call0", "call1"]
        assert [json.loads(m["content"])["data"]["city"] for m in results] == ["0", "1"]

    asyncio.run(run())


def test_long_tracking_url_is_allowed_but_other_argument_limits_are_preserved() -> None:
    handler = AsyncMock(return_value={"text": "ok"})
    executor = ToolExecutor(
        ToolRegistry(
            [Tool("fetch_web_page", "read", {"url": "url"}, ("url",), handler)]
        )
    )
    assert json.loads(
        asyncio.run(
            executor.execute(
                "fetch_web_page",
                json.dumps({"url": "https://example.com/?tracking=" + "x" * 500}),
            )
        )
    )["ok"]
    assert not json.loads(
        asyncio.run(executor.execute("fetch_web_page", json.dumps({"url": "x" * 2049})))
    )["ok"]
    assert handler.await_count == 1
