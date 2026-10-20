"""Фактические уровни заявок; лучшие цены не выдаются за полную глубину."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from protogen_delta.services.market_http import MarketHTTP
from protogen_delta.services.market_models import (
    MOSCOW,
    MarketDataError,
    decode_json,
    finite_number,
    source_time,
    table_rows,
)
from protogen_delta.services.market_series import ALPACA, MOEX


def _level(price: object, size: object) -> dict[str, float | None] | None:
    parsed = finite_number(price, positive=True)
    quantity = finite_number(size, positive=True)
    return {"price": parsed, "quantity": quantity} if parsed is not None else None


def _levels(rows: object, *, buy: bool) -> list[dict[str, float | None]]:
    if not isinstance(rows, list):
        return []
    combined: dict[float, float | None] = {}
    for row in rows[:1000]:
        if not isinstance(row, dict):
            continue
        level = _level(row.get("p"), row.get("s"))
        if level is None:
            continue
        price, size = level["price"], level["quantity"]
        assert price is not None
        if size is None:
            continue
        if price not in combined:
            combined[price] = size
        else:
            previous = combined[price]
            combined[price] = (
                finite_number(previous + size, positive=True)
                if previous is not None
                else None
            )
    return [
        {"price": price, "quantity": combined[price]}
        for price in sorted(combined, reverse=buy)[:10]
    ]


async def read_orderbook(http: MarketHTTP, market: str, symbol: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "market": market,
        "symbol": symbol,
        "bids": [],
        "asks": [],
        "kind": "unavailable",
        "currency": "RUB" if market == "ru" else "USD",
        "unit": (
            "лотов"
            if market == "ru"
            else "акций" if market == "us" else symbol.split("/")[0]
        ),
        "lot_size": None,
        "as_of": None,
        "source": {
            "ru": "Московская биржа · TQBR",
            "us": "Alpaca · IEX",
            "crypto": "Alpaca Crypto · US",
        }[market],
        "notice": "",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        if market == "ru":
            try:
                payload = decode_json(
                    await http.get(
                        MOEX + f"/{symbol}/orderbook.json",
                        {"iss.meta": "off", "iss.only": "orderbook"},
                        ttl=30,
                    )
                )
                if "orderbook" not in payload:
                    raise MarketDataError("Стакан не предоставлен источником.")
                rows = table_rows(payload.get("orderbook"))
                for key, side in (("bids", "B"), ("asks", "S")):
                    result[key] = _levels(
                        [
                            {"p": row.get("PRICE"), "s": row.get("QUANTITY")}
                            for row in rows
                            if row.get("BUYSELL") == side
                        ],
                        buy=side == "B",
                    )
                result["kind"] = "depth"
                result["notice"] = (
                    "До 10 уровней с каждой стороны, объём в лотах. Публичные данные могут задерживаться."
                )
            except MarketDataError:
                payload = decode_json(
                    await http.get(
                        MOEX + f"/{symbol}.json",
                        {
                            "iss.meta": "off",
                            "iss.only": "securities,marketdata",
                            "securities.columns": "SECID,LOTSIZE",
                            "marketdata.columns": "SECID,BID,OFFER,BIDDEPTH,OFFERDEPTH,SYSTIME",
                        },
                        ttl=30,
                    )
                )
                rows = table_rows(payload.get("marketdata"))
                row = next((r for r in rows if r.get("SECID") == symbol), {})
                for key, price, size in (
                    ("bids", "BID", "BIDDEPTH"),
                    ("asks", "OFFER", "OFFERDEPTH"),
                ):
                    level = _level(row.get(price), row.get(size))
                    result[key] = [level] if level else []
                meta = table_rows(payload.get("securities"))
                result["lot_size"] = finite_number(
                    next(
                        (r.get("LOTSIZE") for r in meta if r.get("SECID") == symbol),
                        None,
                    ),
                    positive=True,
                )
                result["as_of"] = source_time(row.get("SYSTIME"), local=MOSCOW)
                result["kind"] = "top"
                result["notice"] = (
                    "Полная глубина MOEX недоступна через текущее подключение. "
                    "Показаны только лучшие цены; объём — если предоставлен "
                    "источником, в лотах. Время — обновление источника; "
                    "данные могут задерживаться."
                )
        elif market == "crypto":
            payload = decode_json(
                await http.get(
                    ALPACA + "/v1beta3/crypto/us/latest/orderbooks",
                    {"symbols": symbol},
                    ttl=15,
                )
            )
            books = payload.get("orderbooks", {})
            book = books.get(symbol, {}) if isinstance(books, dict) else {}
            if not isinstance(book, dict):
                raise MarketDataError("Источник вернул некорректный стакан.")
            result.update(
                bids=_levels(book.get("b"), buy=True),
                asks=_levels(book.get("a"), buy=False),
                as_of=source_time(book.get("t")),
                kind="depth",
                notice=(
                    "До 10 уровней Alpaca US с каждой стороны. Это снимок заявок "
                    "одной площадки, не всего крипторынка."
                ),
            )
        else:
            if not http.alpaca_enabled:
                raise MarketDataError(
                    "Лучшие цены США доступны после подключения Alpaca."
                )
            payload = decode_json(
                await http.get(
                    ALPACA + "/v2/stocks/quotes/latest",
                    {"symbols": symbol, "feed": "iex"},
                    ttl=15,
                )
            )
            quotes = payload.get("quotes", {})
            quote = quotes.get(symbol, {}) if isinstance(quotes, dict) else {}
            if not isinstance(quote, dict):
                raise MarketDataError("Источник вернул некорректную котировку.")
            for key, price, size in (("bids", "bp", "bs"), ("asks", "ap", "as")):
                level = _level(quote.get(price), quote.get(size))
                result[key] = [level] if level else []
            result.update(
                kind="top",
                as_of=source_time(quote.get("t")),
                notice="Лучшие цены IEX, не полный стакан и не весь рынок США. Объём в акциях.",
            )
    except MarketDataError as error:
        result["notice"] = str(error)
    bid = result["bids"][0]["price"] if result["bids"] else None
    ask = result["asks"][0]["price"] if result["asks"] else None
    result["crossed"] = bid is not None and ask is not None and ask < bid
    result["spread"] = (
        finite_number(Decimal(str(ask)) - Decimal(str(bid)))
        if bid is not None and ask is not None and ask >= bid
        else None
    )
    result["spread_pct"] = (
        finite_number(result["spread"] / bid * 100)
        if bid and result["spread"] is not None
        else None
    )
    return result
