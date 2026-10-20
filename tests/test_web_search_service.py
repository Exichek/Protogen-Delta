"""Проверки поставщиков, ограничений, кеша и безопасного резервного чтения."""

import asyncio
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from protogen_delta.config.settings import Settings, load_settings
from protogen_delta.repositories.web_api_usage import (
    WebApiLimitError,
    WebApiUsageRepository,
)
from protogen_delta.services import web_search as module
from protogen_delta.services.market_data import MarketDataService
from protogen_delta.services.market_http import MarketHTTP
from protogen_delta.services.tools import ToolExecutor, default_registry
from protogen_delta.services.tools.http import ProviderHTTPError, UnsafePublicURLError
from protogen_delta.services.web_search import WebSearchService

ROW = {
    "title": "Документация",
    "url": "https://example.org/doc",
    "summary": "Описание",
    "date": "2026-10-18T01:00:00Z",
}
NOW = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WebSearchService:
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    result = WebSearchService(
        WebApiUsageRepository(tmp_path),
        deepseek_key="deepseek-secret",
        brave_key="brave-secret",
        now=lambda: NOW,
    )
    monkeypatch.setattr(result, "_post", AsyncMock(return_value={"results": [ROW]}))
    monkeypatch.setattr(module, "_validate_public_url", AsyncMock())
    return result


def test_primary_shape_filters_and_shared_registration(
    service: WebSearchService,
) -> None:
    async def run() -> None:
        args = {
            "query": "новости Python",
            "count": "3",
            "language": "ru",
            "freshness": "week",
        }
        output = await service.search(args)
        assert output["source"] == "DeepSeek Search API"
        assert output["results"] == [
            {
                "title": ROW["title"],
                "url": ROW["url"],
                "description": "Описание",
                "published": ROW["date"],
            }
        ]
        assert isinstance(service._post, AsyncMock)
        assert service._post.await_args is not None
        body = service._post.await_args.args[1]
        assert body == {
            "query": "новости Python lang:ru after:2026-10-11",
            "count": 3,
            "need_content": False,
        }
        registry = default_registry(web_search_service=service)
        result = json.loads(
            await ToolExecutor(registry).execute("web_search", json.dumps(args))
        )
        assert result["ok"] and result["data"] == output
        assert registry.tools["fetch_web_page"].handler == service.read
        assert service._post.await_count == 1

    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 401, 402, 403, 429, 500, 504])
def test_failure_uses_brave_and_opens_circuit(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    async def run() -> None:
        post = AsyncMock(side_effect=ProviderHTTPError(status))
        monkeypatch.setattr(service, "_post", post)
        brave = AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Резерв",
                        "url": "https://example.org",
                        "description": "Описание",
                    }
                ]
            }
        )
        monkeypatch.setattr(module, "web_search", brave)
        assert (await service.search({"query": "один"}))["source"] == "Brave Search API"
        until = service._blocked["deepseek_search"]
        assert (await service.search({"query": "другой"}))[
            "source"
        ] == "Brave Search API"
        assert post.await_count == 1 and brave.await_count == 2
        assert service._blocked["deepseek_search"] == until
        assert brave.await_args is not None
        assert brave.await_args.kwargs["api_key"] == "brave-secret"

    asyncio.run(run())


def test_empty_search_does_not_pay_for_fallback(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_post", AsyncMock(return_value={"results": []}))
    brave = AsyncMock()
    monkeypatch.setattr(module, "web_search", brave)
    assert asyncio.run(service.search({"query": "ничего"}))["results"] == []
    brave.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": None},
        {"results": [42]},
        {"results": [{"title": "x", "url": "http://127.0.0.1"}]},
    ],
)
def test_malformed_results_use_reserve(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    monkeypatch.setattr(service, "_post", AsyncMock(return_value=payload))
    monkeypatch.setattr(module, "web_search", AsyncMock(return_value={"results": []}))
    assert (
        asyncio.run(service.search({"query": "поиск"}))["source"] == "Brave Search API"
    )


@pytest.mark.parametrize(
    "args",
    [
        {"query": ""},
        {"query": "x"},
        {"query": "x" * 101},
        {"query": "test", "count": "0"},
        {"query": "test", "count": "11"},
        {"query": "test", "count": "x"},
        {"query": "test", "language": "de"},
        {"query": "test", "freshness": "forever"},
    ],
)
def test_invalid_arguments_never_reserve(
    service: WebSearchService, args: dict[str, str]
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(service.search(args))
    assert isinstance(service._post, AsyncMock)
    service._post.assert_not_awaited()


def test_output_size_and_missing_metadata_are_bounded(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "title": "x" * 400,
            "url": "https://example.org/" + "x" * 1500,
            "summary": "x" * 900,
            "date": {"bad": True},
        }
        for _ in range(10)
    ]
    monkeypatch.setattr(service, "_post", AsyncMock(return_value={"results": rows}))
    result = asyncio.run(service.search({"query": "query", "count": "10"}))
    assert 0 < len(result["results"]) < 10
    assert len(json.dumps(result, ensure_ascii=False)) < 16000
    assert result["results"][0]["published"] is None


@pytest.mark.parametrize(
    "url",
    [
        None,
        123,
        "https://localhost/a",
        "http://a.local/a",
        "https://a.localhost",
        "http://127.0.0.1",
        "http://[::1]",
        "http://x:bad",
        "http://x:99999",
        "https://u:p@example.org",
        "file:///etc/passwd",
        "https://example.org/" + "x" * 2048,
        "http://[bad",
    ],
)
def test_bad_result_urls_are_rejected(url: Any) -> None:
    assert not module._public_result_url(url)


def test_public_result_url_accepts_global_ip() -> None:
    assert module._public_result_url("https://8.8.8.8/doc")


def test_cache_singleflight_cancellation_expiry_and_copy(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        clock = [0.0]
        service._clock = lambda: clock[0]
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(*args: Any) -> dict[str, Any]:
            entered.set()
            await release.wait()
            return {"results": [ROW]}

        post = AsyncMock(side_effect=slow)
        monkeypatch.setattr(service, "_post", post)
        first = asyncio.create_task(service.search({"query": "query"}))
        await entered.wait()
        second = asyncio.create_task(service.search({"query": "query"}))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        result = await second
        result["results"].clear()
        assert (await service.search({"query": "query"}))["results"]
        assert post.await_count == 1
        clock[0] = 301
        await service.search({"query": "query"})
        assert post.await_count == 2
        await service.close()
        with pytest.raises(RuntimeError):
            await service.search({"query": "query"})

    asyncio.run(run())


def test_reader_is_free_first_and_only_fetches_on_failure(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        local = AsyncMock(
            return_value={"url": "https://example.org", "text": "обычный текст"}
        )
        monkeypatch.setattr(module, "fetch_web_page", local)
        assert (await service.read({"url": "https://example.org"}))[
            "text"
        ] == "обычный текст"
        assert isinstance(service._post, AsyncMock)
        service._post.assert_not_awaited()
        local.side_effect = TimeoutError()
        service._post.return_value = {"title": "Страница", "content": "x" * 13000}
        result = await service.read({"url": "https://example.org/other"})
        assert result["source"] == "DeepSeek Fetch API"
        assert len(result["text"]) == 12000 and result["truncated"]
        assert result["final_url_verified"] is False
        assert service._post.await_args is not None
        assert service._post.await_args.args == (
            "fetch",
            {"url": "https://example.org/other"},
        )

    asyncio.run(run())


@pytest.mark.parametrize("failure_at", ["initial", "redirect", "youtube"])
def test_unsafe_or_specialized_page_does_not_reach_fetch(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    url = (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        if failure_at == "youtube"
        else "https://example.org"
    )
    if failure_at == "initial":
        monkeypatch.setattr(
            module,
            "_validate_public_url",
            AsyncMock(side_effect=UnsafePublicURLError("blocked")),
        )
    monkeypatch.setattr(
        module,
        "fetch_web_page",
        AsyncMock(
            side_effect=(
                ValueError("video")
                if failure_at == "youtube"
                else UnsafePublicURLError("redirect blocked")
            )
        ),
    )
    with pytest.raises(ValueError):
        asyncio.run(service.read({"url": url}))
    assert isinstance(service._post, AsyncMock)
    service._post.assert_not_awaited()


@pytest.mark.parametrize(
    "payload", [{}, {"content": 42}, {"content": ""}, {"content": "x", "title": None}]
)
def test_fetch_error_is_not_cached(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    monkeypatch.setattr(module, "fetch_web_page", AsyncMock(return_value={"text": ""}))
    monkeypatch.setattr(service, "_post", AsyncMock(return_value=payload))
    with pytest.raises(RuntimeError):
        asyncio.run(service.read({"url": "https://example.org"}))
    assert not service._cache


def test_disabled_and_brave_only_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        disabled = WebSearchService(WebApiUsageRepository(tmp_path))
        assert not disabled.enabled
        assert "web_search" not in default_registry(web_search_service=disabled).tools
        with pytest.raises(RuntimeError):
            await disabled.search({"query": "query"})
        monkeypatch.setattr(
            module, "fetch_web_page", AsyncMock(side_effect=ValueError("offline"))
        )
        monkeypatch.setattr(module, "_validate_public_url", AsyncMock())
        with pytest.raises(RuntimeError):
            await disabled.read({"url": "https://example.org"})
        brave = WebSearchService(WebApiUsageRepository(tmp_path), brave_key="key")
        monkeypatch.setattr(
            module, "web_search", AsyncMock(return_value={"results": []})
        )
        assert (await brave.search({"query": "query"}))["source"] == "Brave Search API"

    asyncio.run(run())


def test_atomic_budgets_survive_restart_and_rollover(tmp_path: Path) -> None:
    async def run() -> None:
        stamp = [NOW.timestamp()]
        usage = WebApiUsageRepository(
            tmp_path, daily_limit=3, brave_monthly_limit=1, clock=lambda: stamp[0]
        )
        results = await asyncio.gather(
            *(usage.reserve("deepseek", "search") for _ in range(10)),
            return_exceptions=True,
        )
        assert sum(item is None for item in results) == 3
        with pytest.raises(WebApiLimitError):
            await WebApiUsageRepository(tmp_path, 3, clock=lambda: stamp[0]).reserve(
                "deepseek", "fetch"
            )
        stamp[0] += 86400
        await usage.reserve("brave", "search")
        with pytest.raises(WebApiLimitError):
            await usage.reserve("brave", "search")
        stamp[0] = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
        await usage.reserve("brave", "search")
        with pytest.raises(ValueError):
            await usage.reserve("brave", "fetch")
        with closing(sqlite3.connect(tmp_path / "llm_usage.db")) as db:
            assert [
                row[1] for row in db.execute("PRAGMA table_info(web_api_usage)")
            ] == ["day", "provider", "operation", "requests"]

    asyncio.run(run())


@pytest.mark.parametrize("limits", [(0, 1), (1, 0), (True, 1), (2**31, 1)])
def test_budget_rejects_invalid_limits(tmp_path: Path, limits: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        WebApiUsageRepository(tmp_path, *limits)


def test_budget_fails_closed_and_tool_reports_limit(
    tmp_path: Path, service: WebSearchService
) -> None:
    async def run() -> None:
        service._usage = WebApiUsageRepository(tmp_path, daily_limit=1)
        await service.search({"query": "first"})
        result = json.loads(
            await ToolExecutor(default_registry(web_search_service=service)).execute(
                "web_search", '{"query":"second"}'
            )
        )
        assert result == {"ok": False, "error": "web_api_limit_exceeded"}
        with closing(sqlite3.connect(tmp_path / "llm_usage.db")) as db:
            db.execute("DROP TABLE web_api_usage")
        with pytest.raises(WebApiLimitError):
            await service.search({"query": "third"})

    asyncio.run(run())


def test_key_isolation_and_settings_options(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings("tg", "model-key", 1)
    assert settings.web_search_api_key == "model-key" and settings.web_search_available
    assert (
        replace(settings, llm_provider="openai-compatible").web_search_api_key is None
    )
    assert (
        replace(settings, deepseek_base_url="https://other.example").web_search_api_key
        is None
    )
    assert replace(settings, deepseek_search_enabled=False).web_search_api_key is None
    assert (
        replace(
            settings,
            llm_provider="openai-compatible",
            deepseek_search_api_key="dedicated",
        ).web_search_api_key
        == "dedicated"
    )
    monkeypatch.setenv("TELEGRAM_TOKEN", "tg")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("ART_CHAT_ID", "1")
    monkeypatch.setenv("WEB_API_DAILY_LIMIT", "17")
    monkeypatch.setenv("BRAVE_MONTHLY_SEARCH_LIMIT", "500")
    monkeypatch.setenv("DEEPSEEK_SEARCH_ENABLED", "false")
    loaded = load_settings()
    assert loaded.web_api_daily_limit == 17 and loaded.brave_monthly_search_limit == 500
    assert loaded.web_search_api_key is None


@pytest.mark.parametrize("status", [200, 401])
def test_post_keeps_key_on_fixed_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    class Response:
        async def __aenter__(self) -> "Response":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class Session(Response):
        post = Mock()

    response = Response()
    response.status = status  # type: ignore[attr-defined]
    session = Session()
    session.post.return_value = response
    monkeypatch.setattr(module, "_session", lambda proxy: session)
    monkeypatch.setattr(
        module, "_read_limited", AsyncMock(return_value=b'{"results":[]}')
    )
    service = WebSearchService(
        WebApiUsageRepository(tmp_path), deepseek_key="private-key"
    )
    if status == 200:
        assert asyncio.run(service._post("search", {"query": "test"})) == {
            "results": []
        }
    else:
        with pytest.raises(ProviderHTTPError):
            asyncio.run(service._post("search", {"query": "test"}))
    assert session.post.call_args.args == ("https://api.deepseek.com/v1/tools/search",)
    assert session.post.call_args.kwargs["allow_redirects"] is False
    assert (
        session.post.call_args.kwargs["headers"]["Authorization"]
        == "Bearer private-key"
    )


def test_markets_use_shared_search_service(service: WebSearchService) -> None:
    async def run() -> None:
        data = MarketDataService(MarketHTTP(), web_search_service=service)
        output = await data.news()
        assert any(item["url"] == ROW["url"] for item in output["items"])
        assert isinstance(service._post, AsyncMock)
        assert service._post.await_args is not None
        assert service._post.await_args.args[0] == "search"
        await data.http.close()

    asyncio.run(run())


def test_cache_and_pending_queue_remain_bounded(service: WebSearchService) -> None:
    async def run() -> None:
        for index in range(130):
            await service._cached(
                str(index), AsyncMock(return_value={"text": "public"})
            )
        assert len(service._cache) == 128 and "0" not in service._cache
        entered = asyncio.Event()

        async def wait() -> dict[str, Any]:
            entered.set()
            await asyncio.Event().wait()
            return {}

        tasks = [
            asyncio.create_task(service._cached("wait" + str(i), wait))
            for i in range(32)
        ]
        await entered.wait()
        with pytest.raises(RuntimeError, match="заняты"):
            await service._cached("overflow", wait)
        await service.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert not service._pending and not service._cache

    asyncio.run(run())


def test_no_successful_source_and_long_url_fail_safely(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        monkeypatch.setattr(service, "_post", AsyncMock(side_effect=TimeoutError()))
        monkeypatch.setattr(
            module, "web_search", AsyncMock(side_effect=ProviderHTTPError(429))
        )
        result = json.loads(
            await ToolExecutor(default_registry(web_search_service=service)).execute(
                "web_search", '{"query":"query"}'
            )
        )
        assert result == {"ok": False, "error": "source_unavailable"}
        with pytest.raises(ValueError, match="длинная"):
            await service.read({"url": "https://example.org/" + "a" * 2048})
        assert not service._cache

    asyncio.run(run())


def test_expired_provider_circuit_can_recover(
    service: WebSearchService, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        clock = [1.0]
        service._clock = lambda: clock[0]
        service._blocked["deepseek_search"] = 30
        monkeypatch.setattr(
            module, "web_search", AsyncMock(return_value={"results": []})
        )
        assert (await service.search({"query": "first"}))[
            "source"
        ] == "Brave Search API"
        clock[0] = 31
        assert (await service.search({"query": "second"}))[
            "source"
        ] == "DeepSeek Search API"

    asyncio.run(run())
