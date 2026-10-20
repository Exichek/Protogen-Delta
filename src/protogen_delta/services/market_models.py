"""Проверенные числа, даты, свечи и официальные валютные курсы."""

import ipaddress
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree

MOSCOW = timezone(timedelta(hours=3))


class MarketDataError(ValueError):
    """Данные источника недоступны; никаких фиктивных котировок."""


def decode_json(raw: bytes) -> dict[str, Any]:
    """Отклонить страницу входа и другой ответ, не являющийся объектом JSON."""
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except ValueError, RecursionError:
        pass
    raise MarketDataError("Источник вернул некорректные данные.")


def finite_number(value: object, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        number = float(value)
    except ValueError, OverflowError:
        return None
    if not math.isfinite(number) or abs(number) > 1e15 or positive and number <= 0:
        return None
    return number


def source_time(value: object, *, local: timezone = timezone.utc) -> str | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if at.tzinfo is None:
            at = at.replace(tzinfo=local)
        return at.astimezone(timezone.utc).isoformat()
    except ValueError, OverflowError:
        return None


def public_link(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 1500:
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username
            or parsed.password
            or host.endswith(".local")
        ):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            if "." not in host:
                return None
        return value
    except ValueError:
        return None


def table_rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    columns, rows = value.get("columns"), value.get("data")
    if (
        not isinstance(columns, list)
        or not isinstance(rows, list)
        or len(columns) > 128
    ):
        return []
    if not all(isinstance(column, str) for column in columns) or len(
        set(columns)
    ) != len(columns):
        return []
    return [
        dict(zip(columns, row))
        for row in rows[:2000]
        if isinstance(row, list) and len(row) == len(columns)
    ]


@dataclass(frozen=True, slots=True)
class MarketQuote:
    market: str
    symbol: str
    name: str
    currency: str
    price: float | None
    change_pct: float | None
    volume: float | None
    as_of: str | None
    source: str
    source_url: str
    notice: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def price_change(price: float | None, previous: object) -> float | None:
    base = finite_number(previous, positive=True)
    return (
        finite_number((price / base - 1) * 100)
        if price is not None and base is not None
        else None
    )


def candle_rows(rows: object, *, moex: bool = False) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    valid: dict[str, dict[str, Any]] = {}
    for row in rows[:2000]:
        if not isinstance(row, dict):
            continue
        date = source_time(
            row.get("begin") if moex else row.get("t"),
            local=MOSCOW if moex else timezone.utc,
        )
        prices = [
            finite_number(row.get(name), positive=True)
            for name in (
                ("open", "high", "low", "close") if moex else ("o", "h", "l", "c")
            )
        ]
        if date is None or any(price is None for price in prices):
            continue
        opened, high, low, closed = prices
        assert (
            opened is not None
            and high is not None
            and low is not None
            and closed is not None
        )
        if low > min(opened, closed) or high < max(opened, closed) or low > high:
            continue
        volume = finite_number(row.get("volume") if moex else row.get("v"))
        valid[date] = {
            "date": date,
            "open": opened,
            "high": high,
            "low": low,
            "close": closed,
            "volume": volume if volume is not None and volume >= 0 else None,
        }
    return [valid[date] for date in sorted(valid)]


@dataclass(frozen=True, slots=True)
class CurrencyRates:
    effective_date: str
    rub_per_unit: dict[str, Decimal]
    names: dict[str, str]

    @classmethod
    def parse(cls, raw: bytes) -> "CurrencyRates":
        try:
            root = ElementTree.fromstring(raw)
            date = datetime.strptime(root.attrib["Date"], "%d.%m.%Y").date().isoformat()
            rates, names = {"RUB": Decimal(1)}, {"RUB": "Российский рубль"}
            for row in root.findall("Valute"):
                code = row.findtext("CharCode", "")
                if len(code) != 3 or not code.isascii() or not code.isupper():
                    continue
                nominal = Decimal(row.findtext("Nominal", "0").replace(",", "."))
                value = Decimal(row.findtext("Value", "0").replace(",", "."))
                if (
                    nominal.is_finite()
                    and value.is_finite()
                    and 0 < nominal <= 1000000
                    and Decimal("0.000001") <= value <= 1000000
                ):
                    rates[code] = value / nominal
                    names[code] = row.findtext("Name", code)[:120]
            if len(rates) < 2:
                raise ValueError("empty")
            return cls(date, rates, names)
        except (
            ElementTree.ParseError,
            KeyError,
            ValueError,
            InvalidOperation,
            OverflowError,
        ):
            raise MarketDataError("ЦБ вернул некорректные данные курсов.") from None

    def convert(self, amount: str, source: str, target: str) -> dict[str, Any]:
        if (
            len(amount) > 40
            or source not in self.rub_per_unit
            or target not in self.rub_per_unit
        ):
            raise ValueError("Проверь сумму и выбранные валюты.")
        if (
            re.fullmatch(
                r"(?:[0-9]{1,13}(?:[.,][0-9]{0,12})?|[.,][0-9]{1,12})", amount.strip()
            )
            is None
        ):
            raise ValueError(
                "Введи обычную неотрицательную сумму, до 12 знаков после запятой."
            )
        try:
            number = Decimal(amount.replace(",", "."))
        except InvalidOperation:
            raise ValueError("Сумма должна быть числом.") from None
        if not number.is_finite() or not 0 <= number <= Decimal("1e12"):
            raise ValueError("Сумма должна быть от 0 до одного триллиона.")
        with localcontext() as context:
            context.prec = 64
            rate = self.rub_per_unit[source] / self.rub_per_unit[target]
            result = (number * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return {
            "amount": str(number),
            "from": source,
            "to": target,
            "rate": str(rate),
            "result": format(result, "f").rstrip("0").rstrip("."),
            "effective_date": self.effective_date,
            "source": "Банк России",
            "source_url": "https://www.cbr.ru/currency_base/daily/",
            "notice": "Официальный курс ЦБ, без банковских комиссий и спреда.",
        }

    def payload(self) -> dict[str, Any]:
        return {
            "effective_date": self.effective_date,
            "currencies": [
                {"code": code, "name": self.names[code], "rub_per_unit": str(rate)}
                for code, rate in self.rub_per_unit.items()
            ],
            "source": "Банк России",
            "source_url": "https://www.cbr.ru/currency_base/daily/",
        }
