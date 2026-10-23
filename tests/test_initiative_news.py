"""Датированные первоисточники, цитаты, пределы обращений и отсутствие выдуманных новостей."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from protogen_delta.repositories.web_api_usage import WebApiLimitError
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.initiative_news import (
    InitiativeNewsService,
    NewsMessage,
    news_url,
)
from protogen_delta.services.web_search import WebSearchService

NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
URL = "https://news.xbox.com/2026/10/23/cooperative-game/"
TEXT = "Компания объявила новую кооперативную игру. Дата выхода пока не названа. Подробности обещаны позднее."


def web_and_model() -> tuple[Mock, AsyncMock]:
    web = Mock(spec=WebSearchService)
    web.enabled = True
    web.search = AsyncMock(
        return_value={"results": [{"url": URL, "published": NOW.isoformat()}]}
    )
    web.read = AsyncMock(
        return_value={
            "url": URL,
            "title": "Анонс",
            "text": TEXT,
            "published": NOW.isoformat(),
        }
    )
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = json.dumps(
        {
            "answer": "Анонсирована новая кооперативная игра, дата выхода пока неизвестна. [S1]",
            "evidence": [{"source": 1, "passages": [1]}],
        }
    )
    return web, model


async def generate(web: Mock, model: AsyncMock, **kwargs: Any) -> NewsMessage | None:
    values: dict[str, Any] = dict(
        user_id=9001,
        at=NOW.timestamp(),
        topics=("games",),
        excluded_urls=(),
        model=model,
        system_prompt="CHARACTER",
        alive=AsyncMock(return_value=True),
    )
    values.update(kwargs)
    return await InitiativeNewsService(web).generate(**values)


def test_one_verified_news_has_page_date_and_intact_source() -> None:
    async def run() -> None:
        web, model = web_and_model()
        result = await generate(web, model)
        assert result is not None
        assert result.url == URL and result.topic == "games"
        assert "Источник от 23.10.2026: " + URL in result.text
        assert "[S1]" not in result.text
        web.search.assert_awaited_once()
        assert web.search.await_args.args[0]["freshness"] == "day"
        assert "9001" not in web.search.await_args.args[0]["query"]
        web.read.assert_awaited_once_with({"url": URL})
        model.chat.assert_awaited_once()
        args = model.chat.await_args.kwargs
        assert args["tool_names"] == () and args["json_response"] is True
        material = json.loads(args["user_message"])
        assert material["pages"][0]["publication_date_source"] == "page_metadata"
        assert material["pages"][0]["passages"][0]["text"] == TEXT

    asyncio.run(run())


@pytest.mark.parametrize(
    "case",
    [
        "old_index",
        "old_page",
        "future_page",
        "no_page_date",
        "metadata_only",
        "empty_page",
        "unverified_redirect",
        "external_redirect",
        "untrusted_host",
    ],
)
def test_unconfirmed_or_stale_source_never_reaches_model(case: str) -> None:
    async def run() -> None:
        web, model = web_and_model()
        row = web.search.return_value["results"][0]
        page = web.read.return_value
        if case == "old_index":
            row["published"] = (NOW - timedelta(days=10)).isoformat()
        elif case == "old_page":
            page["published"] = (NOW - timedelta(days=10)).isoformat()
        elif case == "future_page":
            page["published"] = (NOW + timedelta(days=1)).isoformat()
        elif case == "no_page_date":
            page.pop("published")
        elif case == "metadata_only":
            page["content_scope"] = "metadata_only"
        elif case == "empty_page":
            page["text"] = ""
        elif case == "unverified_redirect":
            page["final_url_verified"] = False
        elif case == "external_redirect":
            page["url"] = "https://example.com/article"
        else:
            row["url"] = "https://news.xbox.com.example.com/fake"
        assert await generate(web, model) is None
        model.chat.assert_not_awaited()

    asyncio.run(run())


def test_same_article_with_tracking_is_not_sent_again() -> None:
    async def run() -> None:
        web, model = web_and_model()
        web.search.return_value["results"][0]["url"] = URL + "?utm_source=news#part"
        assert await generate(web, model, excluded_urls=(URL,)) is None
        web.read.assert_not_awaited()
        model.chat.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "raw",
    [
        "[[SILENCE]]",
        "{}",
        '{"answer":"Не проверял","evidence":[]}',
        json.dumps(
            {"answer": "Факт [S2]", "evidence": [{"source": 1, "passages": [1]}]}
        ),
        json.dumps(
            {
                "answer": "Факт https://example.com/fake",
                "evidence": [{"source": 1, "passages": [1]}],
            }
        ),
        json.dumps(
            {
                "answer": "я" * 421 + " [S1]",
                "evidence": [{"source": 1, "passages": [1]}],
            }
        ),
    ],
)
def test_invalid_unsupported_or_long_answer_is_silence(raw: str) -> None:
    async def run() -> None:
        web, model = web_and_model()
        model.chat.return_value = raw
        assert await generate(web, model) is None
        model.chat.assert_awaited_once()

    asyncio.run(run())


def test_no_more_than_two_pages_are_read() -> None:
    async def run() -> None:
        web, model = web_and_model()
        web.search.return_value["results"] = [{"url": URL + str(n)} for n in range(5)]
        web.read.side_effect = RuntimeError("Недоступно")
        assert await generate(web, model) is None
        assert web.read.await_count == 2
        model.chat.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["before_search", "before_read", "before_model"])
def test_new_input_or_shutdown_stops_work_before_next_operation(stage: str) -> None:
    async def run() -> None:
        web, model = web_and_model()
        alive = AsyncMock(
            side_effect={
                "before_search": [False],
                "before_read": [True, False],
                "before_model": [True, True, False],
            }[stage]
        )
        assert await generate(web, model, alive=alive) is None
        model.chat.assert_not_awaited()
        if stage == "before_search":
            web.search.assert_not_awaited()
        if stage != "before_model":
            web.read.assert_not_awaited()

    asyncio.run(run())


def test_web_quota_error_does_not_try_another_page() -> None:
    async def run() -> None:
        web, model = web_and_model()
        web.read.side_effect = WebApiLimitError("limit")
        with pytest.raises(WebApiLimitError):
            await generate(web, model)
        web.read.assert_awaited_once()
        model.chat.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "value",
    [
        "http://news.xbox.com/a",
        "https://user:pass@news.xbox.com/a",
        "https://news.xbox.com:22/a",
        "https://news.xbox.com:bad/a",
        None,
        "x" * 2049,
    ],
)
def test_invalid_source_url_is_rejected(value: object) -> None:
    assert news_url(value) == ""


def test_tracking_normalization_keeps_semantic_query() -> None:
    assert (
        news_url("https://news.xbox.com/a?post=12&utm_campaign=x&gclid=y#section")
        == "https://news.xbox.com/a?post=12"
    )


def test_publication_date_retains_the_publishers_timezone() -> None:
    async def run() -> None:
        web, model = web_and_model()
        web.read.return_value["published"] = "2026-10-22T23:30:00-08:00"
        result = await generate(web, model)
        assert result is not None
        assert "Источник от 22.10.2026" in result.text

    asyncio.run(run())
