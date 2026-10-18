"""Отдельные период просмотра и размер свечи с ограниченной пагинацией."""

from datetime import datetime, timedelta, timezone
from typing import Any

from protogen_delta.services.market_http import MarketHTTP
from protogen_delta.services.market_models import (
    MOSCOW,
    MarketDataError,
    candle_rows,
    decode_json,
    table_rows,
)

ALPACA = "https://data.alpaca.markets"
MOEX = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities"
PERIOD_DAYS = {"1d": 1, "1w": 7, "1m": 31, "3m": 93, "1y": 366}
INTERVALS = {
    "1m": ("1", "1Min", 1, "1 мин"),
    "10m": ("10", "10Min", 7, "10 мин"),
    "1h": ("60", "1Hour", 31, "1 час"),
    "1d": ("24", "1Day", 366, "1 день"),
}


def validate_chart_window(period: str, interval: str) -> None:
    if period not in PERIOD_DAYS or interval not in INTERVALS:
        raise ValueError("Выбери доступные период и размер свечи.")
    if PERIOD_DAYS[period] > INTERVALS[interval][2]:
        raise ValueError("Для этого периода выбери более крупную свечу.")


async def read_chart(
    http: MarketHTTP, market: str, symbol: str, period: str, interval: str
) -> dict[str, Any]:
    validate_chart_window(period, interval)
    if market == "us" and not http.alpaca_enabled:
        raise MarketDataError("Для графиков акций США подключи Alpaca на сервере.")
    end = datetime.now(timezone.utc)
    if market == "us":
        end -= timedelta(minutes=16)
    start = end - timedelta(days=PERIOD_DAYS[period])
    moex_interval, alpaca_interval, _, label = INTERVALS[interval]
    params = (
        {
            "iss.meta": "off",
            "iss.only": "candles",
            "from": start.astimezone(MOSCOW).strftime("%Y-%m-%d %H:%M:%S"),
            "till": end.astimezone(MOSCOW).strftime("%Y-%m-%d %H:%M:%S"),
            "interval": moex_interval,
            "candles.columns": "open,high,low,close,volume,begin,end",
            "start": "0",
        }
        if market == "ru"
        else {
            "symbols": symbol,
            "timeframe": alpaca_interval,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": "1000",
            "sort": "asc",
        }
    )
    if market == "us":
        params.update({"feed": "iex", "adjustment": "all"})
    url = (
        MOEX + f"/{symbol}/candles.json"
        if market == "ru"
        else ALPACA
        + ("/v2/stocks/bars" if market == "us" else "/v1beta3/crypto/us/bars")
    )
    rows: list[dict[str, Any]] = []
    truncated = False
    tokens: set[str] = set()
    for _ in range(4):
        payload = decode_json(await http.get(url, dict(params), ttl=60))
        if market == "ru":
            page = table_rows(payload.get("candles"))
            more = len(page) >= 500
            params["start"] = str(int(params["start"]) + len(page))
        else:
            bars = payload.get("bars", {})
            raw_page = bars.get(symbol, []) if isinstance(bars, dict) else []
            if not isinstance(raw_page, list):
                raise MarketDataError("Источник вернул некорректные свечи.")
            page = [row for row in raw_page if isinstance(row, dict)]
            token = payload.get("next_page_token")
            more = bool(token)
            if more:
                if not isinstance(token, str) or len(token) > 2000 or token in tokens:
                    raise MarketDataError(
                        "Источник вернул некорректную страницу свечей."
                    )
                tokens.add(token)
                params["page_token"] = token
        rows.extend(page)
        truncated = more
        if not more or not page or len(rows) >= 2000:
            break
    if len(rows) > 2000:
        truncated = True
    candles = candle_rows(rows, moex=market == "ru")
    source = {
        "ru": "Московская биржа · TQBR",
        "us": "Alpaca · IEX",
        "crypto": "Alpaca Crypto · US",
    }[market]
    notice = {
        "ru": "Биржевые сделки TQBR; публичные данные могут задерживаться. Внебиржевые сделки брокера сюда не входят.",
        "us": (
            "IEX покрывает часть рынка США. Последние 16 минут исключены; "
            "цены скорректированы на корпоративные события."
        ),
        "crypto": "Данные площадки Alpaca US, а не всего крипторынка.",
    }[market]
    return {
        "market": market,
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "interval_label": label,
        "timezone": {"ru": "Europe/Moscow", "us": "America/New_York", "crypto": "UTC"}[
            market
        ],
        "candles": candles,
        "source": source,
        "notice": f"Свеча: {label}. {notice} Последняя свеча может формироваться.",
        "truncated": truncated,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
