"""Котировки, дневные свечи, финансовые новости и валютный калькулятор."""

import asyncio
import re
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any

from aiohttp import ClientError

from protogen_delta.services.market_depth import read_orderbook
from protogen_delta.services.market_http import MarketHTTP
from protogen_delta.services.market_models import (
    MOSCOW,
    CurrencyRates,
    MarketDataError,
    MarketQuote,
)
from protogen_delta.services.market_models import decode_json as decode_json
from protogen_delta.services.market_models import (
    finite_number,
    price_change,
    public_link,
    source_time,
    table_rows,
)
from protogen_delta.services.market_series import ALPACA, MOEX, read_chart
from protogen_delta.services.tools.web import web_search
from protogen_delta.services.web_search import WebSearchService

WATCHLISTS = {
    "us": (
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "VOO",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "AMD",
        "NFLX",
        "COIN",
        "JPM",
        "V",
        "XOM",
        "WMT",
        "DIS",
    ),
    "ru": (
        "SBER",
        "SBERP",
        "GAZP",
        "LKOH",
        "ROSN",
        "GMKN",
        "NVTK",
        "TATN",
        "YDEX",
        "OZON",
        "T",
        "PLZL",
        "CHMF",
        "MGNT",
        "MOEX",
        "VTBR",
        "IRAO",
        "ALRS",
        "MTSS",
        "AFLT",
    ),
    "crypto": (
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "LTC/USD",
        "AVAX/USD",
        "DOGE/USD",
        "XRP/USD",
        "ADA/USD",
        "BCH/USD",
        "LINK/USD",
        "DOT/USD",
        "UNI/USD",
    ),
}
US_NAMES = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco Nasdaq-100 ETF",
    "DIA": "SPDR Dow Jones ETF",
    "IWM": "iShares Russell 2000 ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "COIN": "Coinbase",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "XOM": "Exxon Mobil",
    "WMT": "Walmart",
    "DIS": "Walt Disney",
}


def selected_symbols(market: str, raw: str = "") -> tuple[str, ...]:
    """Принимать только тикеры, максимум двадцать; адрес источника выбрать нельзя."""
    if market not in WATCHLISTS or len(raw) > 400:
        raise ValueError("Выбери акции США, России или криптовалюты.")
    symbols = (
        tuple(
            dict.fromkeys(
                item.strip().upper() for item in raw.split(",") if item.strip()
            )
        )
        if raw
        else WATCHLISTS[market]
    )
    pattern = (
        r"[A-Z0-9]{1,12}/USD" if market == "crypto" else r"[A-Z0-9][A-Z0-9.-]{0,15}"
    )
    if not 1 <= len(symbols) <= 20 or any(
        re.fullmatch(pattern, item) is None for item in symbols
    ):
        raise ValueError(
            "Некорректный список тикеров; максимум 20. Криптовалюты — BASE/USD."
        )
    return symbols


class MarketDataService:
    """Собрать факты источников без обращения к модели и без торговых операций."""

    def __init__(
        self,
        http: MarketHTTP,
        *,
        brave_key: str | None = None,
        web_search_service: WebSearchService | None = None,
        proxy_url: str | None = None,
    ) -> None:
        self.http = http
        self._brave_key, self._proxy = brave_key, proxy_url
        self._web_search = web_search_service
        self._news_cache: tuple[datetime, dict[str, Any]] | None = None
        self._news_lock = asyncio.Lock()
        self._charts: OrderedDict[
            tuple[str, str, str, str], tuple[float, dict[str, Any]]
        ] = OrderedDict()
        self._chart_pending: dict[
            tuple[str, str, str, str], asyncio.Task[dict[str, Any]]
        ] = {}
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(self._chart_pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._chart_pending.clear()
        self._charts.clear()
        await self.http.close()

    async def quotes(self, market: str, raw: str = "") -> dict[str, Any]:
        symbols = selected_symbols(market, raw)
        quotes: list[MarketQuote] = []
        notices: list[str] = []
        try:
            if market == "ru":
                payload = decode_json(
                    await self.http.get(
                        MOEX + ".json",
                        {
                            "iss.meta": "off",
                            "iss.only": "securities,marketdata",
                            "securities.columns": "SECID,SHORTNAME,CURRENCYID,PREVPRICE",
                            "marketdata.columns": "SECID,LAST,LASTTOPREVPRICE,VOLTODAY,SYSTIME",
                        },
                    )
                )
                metadata = {
                    row.get("SECID"): row
                    for row in table_rows(payload.get("securities"))
                }
                data = {
                    row.get("SECID"): row
                    for row in table_rows(payload.get("marketdata"))
                }
                for symbol in symbols:
                    meta, row = metadata.get(symbol, {}), data.get(symbol, {})
                    price = finite_number(row.get("LAST"), positive=True)
                    changed = (
                        finite_number(row.get("LASTTOPREVPRICE"))
                        if price is not None
                        else None
                    )
                    quotes.append(
                        MarketQuote(
                            market,
                            symbol,
                            str(meta.get("SHORTNAME") or symbol)[:100],
                            "RUB",
                            price,
                            (
                                changed
                                if changed is not None
                                else price_change(price, meta.get("PREVPRICE"))
                            ),
                            finite_number(row.get("VOLTODAY")),
                            source_time(row.get("SYSTIME"), local=MOSCOW),
                            "Московская биржа",
                            "https://www.moex.com/",
                            "Изменение к предыдущему закрытию. Публичные данные могут "
                            "поступать с задержкой; время — обновление источника MOEX.",
                        )
                    )
            elif market == "crypto" or self.http.alpaca_enabled:
                path = (
                    "/v2/stocks/snapshots"
                    if market == "us"
                    else "/v1beta3/crypto/us/snapshots"
                )
                params = {"symbols": ",".join(symbols)}
                if market == "us":
                    params["feed"] = "iex"
                payload = decode_json(await self.http.get(ALPACA + path, params))
                snapshots = payload if market == "us" else payload.get("snapshots", {})
                if not isinstance(snapshots, dict):
                    raise MarketDataError("Источник вернул некорректные котировки.")
                for symbol in symbols:
                    snapshot = snapshots.get(symbol, {})
                    if not isinstance(snapshot, dict):
                        snapshot = {}
                    trade = snapshot.get("latestTrade") or {}
                    bar = snapshot.get("dailyBar") or {}
                    previous = snapshot.get("prevDailyBar") or {}
                    if not all(
                        isinstance(item, dict) for item in (trade, bar, previous)
                    ):
                        trade, bar, previous = {}, {}, {}
                    price = finite_number(trade.get("p"), positive=True)
                    quotes.append(
                        MarketQuote(
                            market,
                            symbol,
                            US_NAMES.get(symbol, symbol) if market == "us" else symbol,
                            "USD",
                            price,
                            price_change(price, previous.get("c")),
                            finite_number(bar.get("v")),
                            source_time(trade.get("t")),
                            "Alpaca · IEX" if market == "us" else "Alpaca Crypto",
                            "https://alpaca.markets/",
                            (
                                "IEX: часть рынка США; изменение к предыдущему закрытию, объём текущей дневной свечи."
                                if market == "us"
                                else "Изменение к предыдущей дневной свече UTC, "
                                "объём текущей свечи; это не скользящие 24 часа."
                            ),
                        )
                    )
            else:
                notices.append(
                    "Alpaca ещё не подключена: на сервере нужна пара Key ID и Secret Key. "
                    "Российские акции и валюты доступны отдельно."
                )
        except MarketDataError as error:
            notices.append(str(error))
        return {
            "market": market,
            "quotes": [quote.payload() for quote in quotes],
            "notices": notices,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "ranking_scope": "Рост и падение только среди выбранных инструментов.",
            "alpaca_enabled": self.http.alpaca_enabled,
        }

    async def chart(
        self, market: str, symbol: str, period: str = "1m", interval: str = "1d"
    ) -> dict[str, Any]:
        symbols = selected_symbols(market, symbol)
        if len(symbols) != 1:
            raise ValueError("Для графика нужен один тикер.")
        if self._closed:
            raise MarketDataError("Раздел рынков остановлен.")
        key = market, symbols[0], period, interval
        cached = self._charts.get(key)
        if cached and cached[0] > monotonic():
            self._charts.move_to_end(key)
            return deepcopy(cached[1])
        if key not in self._chart_pending:
            if len(self._chart_pending) >= 32:
                raise MarketDataError("Графики сейчас заняты. Попробуй позже.")
            task = asyncio.create_task(self._load_chart(key))
            self._chart_pending[key] = task
            task.add_done_callback(self._consume_chart)
        return deepcopy(await asyncio.shield(self._chart_pending[key]))

    @staticmethod
    def _consume_chart(task: asyncio.Task[dict[str, Any]]) -> None:
        if not task.cancelled():
            task.exception()

    async def _load_chart(self, key: tuple[str, str, str, str]) -> dict[str, Any]:
        try:
            result = await read_chart(self.http, *key)
            # Сохраняем весь снимок, включая подлинное время проверки источника.
            self._charts[key] = monotonic() + 60, result
            self._charts.move_to_end(key)
            while len(self._charts) > 128:
                self._charts.popitem(last=False)
            return result
        finally:
            self._chart_pending.pop(key, None)

    async def orderbook(self, market: str, symbol: str) -> dict[str, Any]:
        symbols = selected_symbols(market, symbol)
        if len(symbols) != 1:
            raise ValueError("Для стакана нужен один тикер.")
        return await read_orderbook(self.http, market, symbols[0])

    async def rates(self) -> CurrencyRates:
        return CurrencyRates.parse(
            await self.http.get("https://www.cbr.ru/scripts/XML_daily.asp", {}, ttl=300)
        )

    async def news(self) -> dict[str, Any]:
        """Отделить финансовую ленту от найденных мировых публикаций за сутки."""
        async with self._news_lock:
            now = datetime.now(timezone.utc)
            if self._news_cache and (now - self._news_cache[0]).total_seconds() < 300:
                return self._news_cache[1]
            result: dict[str, Any] = {
                "items": [],
                "notices": [],
                "checked_at": now.isoformat(),
            }
            if self.http.alpaca_enabled:
                try:
                    payload = decode_json(
                        await self.http.get(
                            ALPACA + "/v1beta1/news",
                            {
                                "start": (now - timedelta(days=2)).isoformat(),
                                "limit": "12",
                                "include_content": "false",
                                "sort": "desc",
                            },
                            ttl=300,
                        )
                    )
                    rows = payload.get("news", [])
                    for row in rows[:12] if isinstance(rows, list) else []:
                        if not isinstance(row, dict) or not (
                            link := public_link(row.get("url"))
                        ):
                            continue
                        result["items"].append(
                            {
                                "title": str(row.get("headline", ""))[:300],
                                "summary": str(row.get("summary", ""))[:1000],
                                "url": link,
                                "source": str(row.get("source", "Alpaca"))[:100],
                                "published": source_time(row.get("created_at")),
                                "kind": "Финансовая лента",
                            }
                        )
                except MarketDataError as error:
                    result["notices"].append(str(error))
            if (
                self._web_search.enabled
                if self._web_search is not None
                else bool(self._brave_key)
            ):
                try:
                    search_args = {
                        "query": "мировые финансовые новости инфляция ставки центральных банков нефть рынки",
                        "freshness": "day",
                        "count": "8",
                        "language": "ru",
                    }
                    world = (
                        await self._web_search.search(search_args)
                        if self._web_search is not None
                        else await web_search(
                            search_args,
                            api_key=self._brave_key or "",
                            proxy_url=self._proxy,
                        )
                    )
                    for row in world.get("results", []):
                        if not (link := public_link(row.get("url"))):
                            continue
                        result["items"].append(
                            {
                                "title": row["title"],
                                "summary": row.get("description", ""),
                                "url": link,
                                "source": "Веб-поиск · "
                                + (
                                    public_link(link)
                                    and link.split("/")[2]
                                    or "источник"
                                ),
                                "published": row.get("published"),
                                "kind": "Мировые события · поиск за сутки",
                            }
                        )
                except ValueError, ClientError, TimeoutError, RuntimeError:
                    result["notices"].append(
                        "Поиск мировых новостей временно недоступен."
                    )
            else:
                result["notices"].append(
                    "Мировые публикации доступны после подключения веб-поиска на сервере."
                )
            if not self.http.alpaca_enabled:
                result["notices"].append("Финансовая лента Alpaca пока не подключена.")
            self._news_cache = now, result
            return result
