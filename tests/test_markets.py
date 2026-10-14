"""Проверки датированных финансовых данных, расчётов, авторизации и изоляции чата."""

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientError
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.llm_budget import current_llm_user
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.market_advisor import MarketAdvisor
from protogen_delta.services.market_data import (
    MarketDataService,
    decode_json,
    selected_symbols,
)
from protogen_delta.services.market_http import MarketHTTP
from protogen_delta.services.market_models import (
    CurrencyRates,
    MarketDataError,
    candle_rows,
    finite_number,
    price_change,
    public_link,
    source_time,
    table_rows,
)

RATES = (
    b'<ValCurs Date="14.10.2026"><Valute><CharCode>USD</CharCode>'
    b"<Nominal>1</Nominal><Value>90,00</Value><Name>Dollar</Name></Valute>"
    b"<Valute><CharCode>KZT</CharCode><Nominal>100</Nominal><Value>18,00</Value>"
    b"<Name>Tenge</Name></Valute><Valute><CharCode>bad</CharCode>"
    b"<Nominal>1</Nominal><Value>0</Value></Valute></ValCurs>"
)
BAR = {"t": "2026-10-13T00:00:00Z", "o": 100, "h": 120, "l": 90, "c": 110, "v": 500}


class FakeHTTP(MarketHTTP):
    """Подставить ответы источников без сети и настоящих ключей."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(
            "test-key" if enabled else None, "test-secret" if enabled else None
        )
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.fail = False
        self.broken: object | None = None

    async def get(self, url: str, parameters: Any, *, ttl: float = 30) -> bytes:
        payload: dict[str, Any]
        self.calls.append((url, dict(parameters)))
        if self.fail:
            raise MarketDataError("Источник недоступен.")
        if self.broken is not None:
            return json.dumps(self.broken).encode()
        if "XML_daily" in url:
            return RATES
        if "candles" in url:
            payload = {
                "candles": {
                    "columns": ["begin", "open", "high", "low", "close", "volume"],
                    "data": [["2026-10-13 10:00:00", 100, 120, 90, 110, 500]],
                }
            }
        elif "moex.com" in url:
            payload = {
                "securities": {
                    "columns": ["SECID", "SHORTNAME", "CURRENCYID", "PREVPRICE"],
                    "data": [["SBER", "Сбербанк", "SUR", 100]],
                },
                "marketdata": {
                    "columns": [
                        "SECID",
                        "LAST",
                        "LASTTOPREVPRICE",
                        "VOLTODAY",
                        "SYSTIME",
                    ],
                    "data": [["SBER", 110, None, 500, "2026-10-14 15:00:00"]],
                },
            }
        elif "snapshots" in url:
            snapshots = {
                symbol: {
                    "latestTrade": {"p": 110, "t": "2026-10-14T12:00:00Z"},
                    "dailyBar": {"v": 500},
                    "prevDailyBar": {"c": 100},
                }
                for symbol in parameters["symbols"].split(",")
            }
            payload = {"snapshots": snapshots} if "crypto" in url else snapshots
        elif "/bars" in url:
            payload = {"bars": {parameters["symbols"]: [BAR]}, "next_page_token": None}
        else:
            payload = {
                "news": [
                    {
                        "headline": "Test <script>",
                        "summary": "Rates changed",
                        "source": "Provider",
                        "url": "https://example.org/news",
                        "created_at": "2026-10-14T10:00:00Z",
                    },
                    {"url": "http://127.0.0.1/private"},
                    "bad",
                ]
            }
        return json.dumps(payload).encode()


@pytest.mark.parametrize(
    "value", [None, True, False, [], {}, "NaN", "inf", "-inf", 1e16, "bad"]
)
def test_financial_numbers_reject_nonfinite(value: object) -> None:
    assert finite_number(value) is None
    assert price_change(1e15, 1e-320) is None


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org",
        "https://localhost",
        "https://127.0.0.1/a",
        "https://[::1]",
        "https://user:secret@example.org",
        "https://internal.local",
        "https://[invalid",
        "x" * 1501,
        None,
    ],
)
def test_news_links_reject_private_and_unsafe(url: object) -> None:
    assert public_link(url) is None


def test_dates_tables_and_candles_validate_provider_data() -> None:
    assert finite_number("1.25", positive=True) == 1.25
    assert finite_number(-1, positive=True) is None
    assert source_time("2026-10-14T12:00:00Z") == "2026-10-14T12:00:00+00:00"
    assert source_time("invalid") is None and source_time(None) is None
    assert public_link("https://example.org/news")
    assert public_link("https://8.8.8.8/")
    for raw in (
        None,
        {},
        {"columns": ["a", "a"], "data": [[1, 2]]},
        {"columns": [1], "data": []},
    ):
        assert table_rows(raw) == []
    assert table_rows({"columns": ["a"], "data": [[1], [1, 2], "x"]}) == [{"a": 1}]
    assert candle_rows(None) == []
    bad = [
        None,
        {},
        {**BAR, "o": True},
        {**BAR, "h": 100},
        {**BAR, "l": 111},
        {**BAR, "v": -1},
    ]
    valid = candle_rows([BAR, BAR, *bad])
    assert len(valid) == 1 and valid[0]["volume"] is None
    assert decode_json(b"{}") == {}
    for encoded in (b"[]", b"bad"):
        with pytest.raises(MarketDataError):
            decode_json(encoded)


def test_currency_cross_rates_use_nominal_and_decimal() -> None:
    rates = CurrencyRates.parse(RATES)
    assert rates.rub_per_unit["KZT"] == Decimal("0.18")
    assert rates.convert("1000", "RUB", "KZT")["result"] == "5555.56"
    assert rates.convert("2,5", "USD", "RUB")["result"] == "225"
    assert rates.convert("100", "KZT", "USD")["result"] == "0.2"
    assert rates.convert("0", "USD", "USD")["result"] == "0"
    assert rates.convert(".5", "USD", "RUB")["result"] == "45"
    with pytest.raises(ValueError):
        rates.convert("0e999999999", "USD", "RUB")
    assert rates.convert("1000000000000", "USD", "KZT")["result"] == "500000000000000"
    assert rates.payload()["effective_date"] == "2026-10-14"
    for amount, source, target in [
        ("NaN", "RUB", "USD"),
        ("-1", "RUB", "USD"),
        ("bad", "RUB", "USD"),
        ("1e13", "RUB", "USD"),
        ("1", "XXX", "USD"),
        ("1" * 41, "RUB", "USD"),
    ]:
        with pytest.raises(ValueError):
            rates.convert(amount, source, target)
    for raw in (
        b"bad",
        b"<ValCurs />",
        b'<ValCurs Date="14.10.2026"/>',
        RATES.replace(b"90,00", b"NaN"),
    ):
        if b"NaN" in raw:
            assert "USD" not in CurrencyRates.parse(raw).rub_per_unit
        else:
            with pytest.raises(MarketDataError):
                CurrencyRates.parse(raw)


@pytest.mark.parametrize(
    "market,raw",
    [
        ("bad", ""),
        ("us", "http://localhost"),
        ("ru", "../.env"),
        ("crypto", "BTC/RUB"),
        ("us", ",".join(str(i) for i in range(21))),
        ("us", "A" * 401),
    ],
)
def test_market_input_cannot_select_urls_or_secrets(market: str, raw: str) -> None:
    with pytest.raises(ValueError):
        selected_symbols(market, raw)


@pytest.mark.parametrize(
    "amount,expected",
    [
        ("3744", "3744"),
        ("106837.606838", "106837.61"),
        ("238.237897", "238.24"),
        ("1.005", "1.01"),
        ("1.0049999", "1"),
        ("0.995", "1"),
        ("1000000000000", "1000000000000"),
    ],
)
def test_currency_result_rounds_once_and_has_at_most_two_decimals(
    amount: str, expected: str
) -> None:
    """Денежный итог округляется один раз без хвоста нулей и потери больших целых."""
    rates = CurrencyRates.parse(RATES)
    assert rates.convert(amount, "RUB", "RUB")["result"] == expected


def test_quotes_charts_and_partial_unavailability() -> None:
    async def scenario() -> None:
        http = FakeHTTP()
        data = MarketDataService(http)
        for market, symbol in [("us", "AAPL"), ("ru", "SBER"), ("crypto", "BTC/USD")]:
            result = await data.quotes(market, symbol)
            quote = result["quotes"][0]
            assert quote["price"] == 110 and quote["change_pct"] == pytest.approx(10)
            assert quote["as_of"] == "2026-10-14T12:00:00+00:00"
            chart = await data.chart(market, symbol)
            assert chart["candles"][0]["close"] == 110
        assert selected_symbols("us", " aapl,AAPL,SPY ") == ("AAPL", "SPY")
        assert selected_symbols("ru")
        assert (
            "24 часа" in (await data.quotes("crypto", "BTC/USD"))["quotes"][0]["notice"]
        )
        assert (await data.quotes("ru", "INVALID"))["quotes"][0]["price"] is None
        assert (await data.rates()).rub_per_unit["USD"] == 90
        news = await data.news()
        assert len(news["items"]) == 1 and news["notices"]
        before = len(http.calls)
        assert await data.news() == news
        assert len(http.calls) == before
        http.fail = True
        assert (await data.quotes("us", "SPY"))["notices"]
        noauth = MarketDataService(FakeHTTP(False))
        assert (await noauth.quotes("us"))["quotes"] == []
        assert (await noauth.news())["notices"]
        with pytest.raises(MarketDataError):
            await noauth.chart("us", "SPY")
        for symbol, period in [("AAPL,SPY", "1m"), ("SPY", "bad")]:
            with pytest.raises(ValueError):
                await data.chart("us", symbol, period)
        http.fail = False
        http.broken = {"snapshots": []}
        assert (await data.quotes("crypto"))["notices"]
        http.broken = {"AAPL": {"latestTrade": "bad"}}
        assert (await data.quotes("us", "AAPL"))["quotes"][0]["price"] is None
        http.broken = {"AAPL": 42}
        assert (await data.quotes("us", "AAPL"))["quotes"][0]["price"] is None
        http.broken = {"bars": [], "next_page_token": "next"}
        assert (await data.chart("us", "AAPL"))["truncated"]

    asyncio.run(scenario())


def test_news_world_search_sources_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        search = AsyncMock(
            return_value={
                "results": [
                    {
                        "url": "https://example.org/world",
                        "title": "World",
                        "description": "Summary",
                        "published": "2026-10-14",
                    },
                    {"url": "https://localhost/a"},
                ]
            }
        )
        monkeypatch.setattr("protogen_delta.services.market_data.web_search", search)
        data = MarketDataService(FakeHTTP(), brave_key="test-brave")
        assert len((await data.news())["items"]) == 2
        assert search.await_args is not None
        assert search.await_args.kwargs["api_key"] == "test-brave"
        data._news_cache = None
        data.http = FakeHTTP()
        cast(FakeHTTP, data.http).fail = True
        search.side_effect = TimeoutError
        news = await data.news()
        assert len(news["notices"]) == 2

    asyncio.run(scenario())


def test_http_isolation_bounded_cache_and_duplicate_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        status = 200
        content = Mock()

        async def __aenter__(self) -> "Response":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    response = Response()

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Response:
            calls.append({"url": url, **kwargs})
            return response

    monkeypatch.setattr(
        "protogen_delta.services.market_http._session", lambda proxy: Session()
    )

    async def read(value: object) -> bytes:
        await asyncio.sleep(0)
        return b"{}"

    monkeypatch.setattr("protogen_delta.services.market_http._read_limited", read)

    async def scenario() -> None:
        http = MarketHTTP("test-key", "test-secret")
        url = "https://data.alpaca.markets/v2/stocks/snapshots"
        assert (
            await asyncio.gather(*(http.get(url, {"symbols": "SPY"}) for _ in range(5)))
            == [b"{}"] * 5
        )
        assert len(calls) == 1
        assert calls[0]["headers"]["APCA-API-SECRET-KEY"] == "test-secret"
        assert not calls[0]["allow_redirects"]
        await http.get(url, {"symbols": "SPY"})
        assert len(calls) == 1
        await http.get("https://www.cbr.ru/scripts/XML_daily.asp", {})
        assert "APCA-API-SECRET-KEY" not in calls[-1]["headers"]
        for invalid in (
            "https://example.org",
            "http://iss.moex.com",
            "https://user@data.alpaca.markets",
            "https://data.alpaca.markets/?secret=1",
        ):
            with pytest.raises(ValueError):
                await http.get(invalid, {})
        for status in (401, 403, 429, 500, 302):
            response.status = status
            with pytest.raises(MarketDataError):
                await http.get(url, {"test": str(status)})
            before = len(calls)
            with pytest.raises(MarketDataError):
                await http.get(url, {"test": str(status)})
            assert len(calls) == before
        response.status = 200
        monkeypatch.setattr(
            "protogen_delta.services.market_http._read_limited",
            AsyncMock(side_effect=ClientError("secret-body")),
        )
        with pytest.raises(MarketDataError, match="временно"):
            await http.get(url, {"test": "network"})
        for i in range(150):
            http._remember(str(i), b"{}", 300)
        assert len(http._cache) == 128
        http._cache["expired"] = (monotonic() - 1, b"old")
        monkeypatch.setattr(
            "protogen_delta.services.market_http._read_limited",
            AsyncMock(return_value=b"{}"),
        )
        http._alpaca_requests.extend([monotonic()] * 180)
        with pytest.raises(MarketDataError, match="Лимит"):
            await http.get(url, {"test": "limit"})
        await http.close()
        assert not http._cache

    asyncio.run(scenario())


def test_advisor_separate_history_budget_owner_failure_and_reset() -> None:
    async def scenario() -> None:
        data = MarketDataService(FakeHTTP())
        model = Mock(spec=DeepSeekService)
        owners = []

        async def answer(*args: Any, **kwargs: Any) -> str:
            owners.append(current_llm_user.get())
            return "Проверенный ответ"

        model.chat = AsyncMock(side_effect=answer)
        advisor = MarketAdvisor(data, model)
        result = await advisor.answer(42, "Сравни риски", "ru", "SBER")
        assert result["thinking"] and owners == [42] and current_llm_user.get() is None
        assert model.chat.await_args.kwargs["tool_names"] == ()
        assert model.chat.await_args.kwargs["history"] == ()
        await advisor.answer(43, "Цена?", "ru", "SBER")
        assert not model.chat.await_args.kwargs["history"]
        await advisor.answer(42, "А теперь?", "ru", "SBER")
        assert len(model.chat.await_args.kwargs["history"]) == 1
        assert "Python" not in model.chat.await_args.args[0]
        for _ in range(7):
            await advisor.answer(42, "Цена?", "ru", "SBER")
        assert len(advisor._sessions[42].turns) == 6
        model.chat.side_effect = RuntimeError("failed")
        with pytest.raises(RuntimeError):
            await advisor.answer(42, "Цена?", "ru", "SBER")
        assert len(advisor._sessions[42].turns) == 6 and not advisor._active
        await advisor.clear(42)
        assert 42 not in advisor._sessions and 43 in advisor._sessions
        for question in ("", "x" * 2001):
            with pytest.raises(ValueError):
                await advisor.answer(1, question, "us", "SPY")
        with pytest.raises(ValueError):
            await advisor.answer(1, "x", "us", "SPY", "bad")
        advisor._sessions[43].updated -= 7201
        model.chat.side_effect = answer
        await advisor.answer(44, "Цена?", "ru", "SBER")
        assert 43 not in advisor._sessions
        await advisor.close()
        assert not advisor._sessions

    asyncio.run(scenario())


def test_advisor_clear_cancels_active_response_without_restoring_history() -> None:
    async def scenario() -> None:
        model = Mock(spec=DeepSeekService)
        entered = asyncio.Event()

        async def wait(*args: Any, **kwargs: Any) -> str:
            entered.set()
            await asyncio.Event().wait()
            return "never"

        model.chat = AsyncMock(side_effect=wait)
        advisor = MarketAdvisor(MarketDataService(FakeHTTP()), model)
        task = asyncio.create_task(advisor.answer(42, "Цена?", "us", "SPY"))
        await entered.wait()
        with pytest.raises(MarketDataError, match="занят"):
            await advisor.answer(42, "Ещё?", "us", "SPY")
        await advisor.clear(42)
        assert task.cancelled() and 42 not in advisor._sessions and not advisor._active

    asyncio.run(scenario())


def test_market_routes_auth_validation_calculation_and_isolation() -> None:
    async def scenario() -> None:
        data = MarketDataService(FakeHTTP())
        model = Mock(spec=DeepSeekService)
        model.chat = AsyncMock(return_value="Ответ")
        advisor = MarketAdvisor(data, model)
        states = UserStateStore()
        app = MiniAppServer(TOKEN, states, market_advisor=advisor).application()
        async with TestClient(TestServer(app)) as client:
            for path in ("quotes", "chart", "news", "rates"):
                response = await client.get("/api/markets/" + path)
                assert response.status == 401
            assert not cast(FakeHTTP, data.http).calls
            js = await client.get("/static/markets.js")
            assert js.status == 200
            headers = {"X-Telegram-Init-Data": _signed_init_data()}
            for path in (
                "quotes?market=ru&symbols=SBER",
                "chart?market=ru&symbol=SBER",
                "news",
                "rates",
            ):
                response = await client.get("/api/markets/" + path, headers=headers)
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
            response = await client.post(
                "/api/markets/convert",
                headers=headers,
                json={"amount": "1000", "from": "RUB", "to": "KZT"},
            )
            assert (
                response.status == 200
                and (await response.json())["result"] == "5555.56"
            )
            response = await client.post(
                "/api/markets/consult",
                headers=headers,
                json={"text": "Анализ рисков", "market": "ru", "symbols": "SBER"},
            )
            assert response.status == 200 and (await response.json())["thinking"]
            assert (
                not states.get(42).history and not states.get(42).emotions.memory.events
            )
            response = await client.post(
                "/api/markets/consult", headers=headers, json={"text": "x"}
            )
            assert response.status == 429
            response = await client.delete("/api/markets/history", headers=headers)
            assert response.status == 200 and not advisor._sessions
            other = {"X-Telegram-Init-Data": _signed_init_data(user_id=43)}
            invalid_payloads: list[Any] = [
                {"user_id": "42", "text": "x"},
                {"history": "spoof", "text": "x"},
                [],
                {"text": 1},
            ]
            for payload in invalid_payloads:
                response = await client.post(
                    "/api/markets/consult", headers=other, json=payload
                )
                assert response.status in (400, 429)

    asyncio.run(scenario())


def test_market_routes_unavailable_and_provider_errors() -> None:
    async def scenario() -> None:
        app = MiniAppServer(TOKEN, UserStateStore()).application()
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(app)) as client:
            assert (
                await client.get("/api/markets/rates", headers=headers)
            ).status == 503
        http = FakeHTTP()
        http.fail = True
        advisor = MarketAdvisor(MarketDataService(http), Mock(spec=DeepSeekService))
        app = MiniAppServer(
            TOKEN, UserStateStore(), market_advisor=advisor
        ).application()
        async with TestClient(TestServer(app)) as client:
            assert (
                await client.get("/api/markets/rates", headers=headers)
            ).status == 503
            assert (
                await client.get("/api/markets/chart?symbol=../.env", headers=headers)
            ).status == 400
            assert (
                await client.get(
                    "/api/markets/news?url=https://localhost", headers=headers
                )
            ).status == 400

    asyncio.run(scenario())


def test_consultant_client_has_no_regular_tools() -> None:
    model = DeepSeekService(
        "test", "https://example.org", "test", adaptive_thinking=True
    )
    model._tools = cast(Any, object())
    consultant = model.for_consultant()
    assert consultant._tools is None and model._tools is not None
    assert consultant.for_reasoning()._provider_options["reasoning_effort"] == "medium"


def test_market_history_removed_on_full_memory_reset(tmp_path: Path) -> None:
    from protogen_delta.repositories.memories import MemoriesRepository
    from protogen_delta.services.memory import MemoryService

    async def scenario() -> None:
        memory = MemoryService(MemoriesRepository(tmp_path))
        cleanup = AsyncMock()
        memory.extra_cleanup = cleanup
        await memory.delete_user(42)
        cleanup.assert_awaited_once_with(42)

    asyncio.run(scenario())
