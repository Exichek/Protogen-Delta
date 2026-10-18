"""Интервалы и глубина берутся у источника; недостающие заявки не синтезируются."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_markets import BAR, FakeHTTP
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.market_advisor import MarketAdvisor
from protogen_delta.services.market_data import MarketDataService, selected_symbols
from protogen_delta.services.market_models import MarketDataError
from protogen_delta.services.market_series import validate_chart_window


def test_watchlists_expand_without_exceeding_request_limit() -> None:
    assert len(selected_symbols("ru")) == len(selected_symbols("us")) == 20
    assert len(selected_symbols("crypto")) == 12
    assert {"OZON", "T", "SBERP", "PLZL", "MOEX"} <= set(selected_symbols("ru"))
    assert {"META", "AMD", "DIA", "VOO", "COIN"} <= set(selected_symbols("us"))


@pytest.mark.parametrize("market", ["ru", "us", "crypto"])
@pytest.mark.parametrize(
    "interval,moex,alpaca",
    [
        ("1m", "1", "1Min"),
        ("10m", "10", "10Min"),
        ("1h", "60", "1Hour"),
        ("1d", "24", "1Day"),
    ],
)
def test_native_intervals_are_independent_of_range(
    market: str, interval: str, moex: str, alpaca: str
) -> None:
    async def run() -> None:
        http = FakeHTTP()
        service = MarketDataService(http)
        symbol = (
            "SBER" if market == "ru" else "BTC/USD" if market == "crypto" else "SPY"
        )
        result = await service.chart(market, symbol, "1d", interval)
        assert result["interval"] == interval
        assert result["period"] == "1d"
        assert http.calls[0][1]["interval" if market == "ru" else "timeframe"] == (
            moex if market == "ru" else alpaca
        )
        assert (
            result["timezone"]
            == {"ru": "Europe/Moscow", "us": "America/New_York", "crypto": "UTC"}[
                market
            ]
        )
        assert result["candles"][0]["open"] == 100

    asyncio.run(run())


@pytest.mark.parametrize(
    "period,interval",
    [("bad", "1d"), ("1m", "5m"), ("1y", "1h"), ("1w", "1m"), ("1m", "10m")],
)
def test_expensive_or_unsupported_windows_are_rejected_before_http(
    period: str, interval: str
) -> None:
    http = FakeHTTP()
    with pytest.raises(ValueError):
        asyncio.run(MarketDataService(http).chart("ru", "SBER", period, interval))
    assert not http.calls
    with pytest.raises(ValueError):
        validate_chart_window(period, interval)


def test_moex_pagination_preserves_more_than_500_minutes() -> None:
    async def run() -> None:
        http = FakeHTTP()
        started = datetime(2026, 10, 17, tzinfo=timezone.utc)
        pages = []
        for offset, count in [(0, 500), (500, 25)]:
            pages.append(
                json.dumps(
                    {
                        "candles": {
                            "columns": [
                                "begin",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ],
                            "data": [
                                [
                                    (started + timedelta(minutes=i)).isoformat(),
                                    100,
                                    120,
                                    90,
                                    110,
                                    1,
                                ]
                                for i in range(offset, offset + count)
                            ],
                        }
                    }
                ).encode()
            )
        http.get = AsyncMock(side_effect=pages)  # type: ignore[method-assign]
        result = await MarketDataService(http).chart("ru", "SBER", "1d", "1m")
        assert len(result["candles"]) == 525 and not result["truncated"]
        assert http.get.await_args_list[1].args[1]["start"] == "500"

    asyncio.run(run())


def test_alpaca_pagination_retains_order_and_detects_repeated_tokens() -> None:
    async def run() -> None:
        http = FakeHTTP()
        pages = [
            json.dumps(
                {"bars": {"BTC/USD": [BAR]}, "next_page_token": "page2"}
            ).encode(),
            json.dumps(
                {"bars": {"BTC/USD": [{**BAR, "t": "2026-10-14T00:00:00Z"}]}}
            ).encode(),
        ]
        http.get = AsyncMock(side_effect=pages)  # type: ignore[method-assign]
        result = await MarketDataService(http).chart("crypto", "BTC/USD")
        assert len(result["candles"]) == 2 and not result["truncated"]
        assert http.get.await_args_list[1].args[1]["page_token"] == "page2"
        repeated = json.dumps(
            {"bars": {"BTC/USD": [BAR]}, "next_page_token": "same"}
        ).encode()
        http.get = AsyncMock(return_value=repeated)  # type: ignore[method-assign]
        with pytest.raises(MarketDataError):
            await MarketDataService(http).chart("crypto", "BTC/USD")

    asyncio.run(run())


@pytest.mark.parametrize("token", [42, "x" * 2001])
def test_provider_cannot_supply_invalid_page_token(token: object) -> None:
    http = FakeHTTP()
    http.broken = {"bars": {"SPY": [BAR]}, "next_page_token": token}
    with pytest.raises(MarketDataError):
        asyncio.run(MarketDataService(http).chart("us", "SPY"))


def test_malformed_bar_page_and_multiple_symbols_are_rejected() -> None:
    http = FakeHTTP()
    http.broken = {"bars": {"SPY": "wrong"}}
    with pytest.raises(MarketDataError):
        asyncio.run(MarketDataService(http).chart("us", "SPY"))
    with pytest.raises(ValueError):
        asyncio.run(MarketDataService(http).chart("us", "SPY,QQQ"))
    with pytest.raises(ValueError):
        asyncio.run(MarketDataService(http).orderbook("us", "SPY,QQQ"))


def test_crypto_depth_sorted_aggregated_and_spread_is_exact() -> None:
    http = FakeHTTP()
    http.broken = {
        "orderbooks": {
            "BTC/USD": {
                "b": [
                    {"p": 100, "s": 2},
                    {"p": 99, "s": 3},
                    {"p": 100, "s": 4},
                    {"p": 200, "s": 0},
                    {"p": "NaN", "s": 1},
                    None,
                ],
                "a": [{"p": 101, "s": 5}, {"p": 100.01, "s": 1}],
                "t": "2026-10-18T07:00:00Z",
            }
        }
    }
    result = asyncio.run(MarketDataService(http).orderbook("crypto", "BTC/USD"))
    assert result["kind"] == "depth" and result["unit"] == "BTC"
    assert result["bids"] == [
        {"price": 100, "quantity": 6},
        {"price": 99, "quantity": 3},
    ]
    assert result["asks"][0]["price"] == 100.01
    assert result["spread"] == 0.01 and not result["crossed"]


def test_crypto_depth_is_limited_to_ten_and_crossed_book_is_flagged() -> None:
    http = FakeHTTP()
    http.broken = {
        "orderbooks": {
            "BTC/USD": {
                "b": [{"p": i, "s": 1} for i in range(100, 130)],
                "a": [{"p": 99, "s": 2}],
            }
        }
    }
    result = asyncio.run(MarketDataService(http).orderbook("crypto", "BTC/USD"))
    assert len(result["bids"]) == 10 and result["crossed"]
    assert result["spread"] is None


def test_moex_full_depth_is_not_replaced_with_made_up_orders() -> None:
    http = FakeHTTP()
    http.broken = {
        "orderbook": {
            "columns": ["BUYSELL", "PRICE", "QUANTITY"],
            "data": [["B", 100, 7], ["S", 101, 3]],
        }
    }
    result = asyncio.run(MarketDataService(http).orderbook("ru", "SBER"))
    assert result["kind"] == "depth" and result["unit"] == "лотов"
    assert result["bids"] == [{"price": 100, "quantity": 7}]


def test_moex_unavailable_depth_uses_only_actual_best_prices() -> None:
    http = FakeHTTP()
    quote = json.dumps(
        {
            "securities": {"columns": ["SECID", "LOTSIZE"], "data": [["SBER", 10]]},
            "marketdata": {
                "columns": [
                    "SECID",
                    "BID",
                    "OFFER",
                    "BIDDEPTH",
                    "OFFERDEPTH",
                    "SYSTIME",
                ],
                "data": [["SBER", 100, 100.01, None, None, "2026-10-18 10:00:00"]],
            },
        }
    ).encode()
    http.get = AsyncMock(side_effect=[b"<html>Access required</html>", quote])  # type: ignore[method-assign]
    result = asyncio.run(MarketDataService(http).orderbook("ru", "SBER"))
    assert result["kind"] == "top" and result["lot_size"] == 10
    assert result["bids"] == [{"price": 100, "quantity": None}]
    assert result["spread"] == 0.01
    assert "глубина MOEX недоступна" in result["notice"]
    assert result["as_of"] == "2026-10-18T07:00:00+00:00"


def test_stock_quote_is_level_one_with_share_sizes() -> None:
    http = FakeHTTP()
    http.broken = {
        "quotes": {
            "SPY": {
                "bp": 100,
                "ap": 101,
                "bs": 50,
                "as": 70,
                "t": "2026-10-18T07:00:00Z",
            }
        }
    }
    result = asyncio.run(MarketDataService(http).orderbook("us", "SPY"))
    assert result["kind"] == "top" and result["unit"] == "акций"
    assert result["bids"][0]["quantity"] == 50
    assert http.calls[0][1]["feed"] == "iex"


@pytest.mark.parametrize(
    "market,symbol", [("ru", "SBER"), ("us", "SPY"), ("crypto", "BTC/USD")]
)
def test_source_failure_leaves_book_unavailable(market: str, symbol: str) -> None:
    http = FakeHTTP()
    http.fail = True
    result = asyncio.run(MarketDataService(http).orderbook(market, symbol))
    assert result["kind"] == "unavailable"
    assert not result["bids"] and not result["asks"] and result["spread"] is None


def test_no_stock_credentials_do_not_trigger_http() -> None:
    http = FakeHTTP(enabled=False)
    result = asyncio.run(MarketDataService(http).orderbook("us", "SPY"))
    assert result["kind"] == "unavailable" and not http.calls


def test_public_crypto_quotes_work_without_stock_credentials() -> None:
    http = FakeHTTP(enabled=False)
    result = asyncio.run(MarketDataService(http).quotes("crypto", "BTC/USD"))
    assert result["quotes"][0]["price"] == 110
    assert "crypto/us/snapshots" in http.calls[0][0]


def test_new_routes_authenticate_before_data_and_carry_interval() -> None:
    async def run() -> None:
        http = FakeHTTP()
        advisor = MarketAdvisor(MarketDataService(http), Mock(spec=DeepSeekService))
        server = MiniAppServer(TOKEN, UserStateStore(), market_advisor=advisor)
        async with TestClient(TestServer(server.application())) as client:
            assert (
                await client.get("/api/markets/orderbook?market=ru&symbol=SBER")
            ).status == 401
            assert not http.calls
            headers = {"X-Telegram-Init-Data": _signed_init_data(user_id=42)}
            response = await client.get(
                "/api/markets/chart?market=ru&symbol=SBER&period=1w&interval=1h",
                headers=headers,
            )
            assert (
                response.status == 200 and (await response.json())["interval"] == "1h"
            )
            assert (
                await client.get(
                    "/api/markets/orderbook?market=ru&symbol=SBER", headers=headers
                )
            ).status == 200

    asyncio.run(run())
