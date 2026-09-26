"""Реестр разрешённых инструментов и ограниченное выполнение внешних запросов."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import partial
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from xml.etree import ElementTree

from aiohttp import ClientSession, ClientTimeout
from aiohttp_socks import ProxyConnector

logger = logging.getLogger(__name__)
MOSCOW = timezone(timedelta(hours=3), "Europe/Moscow")
ToolHandler = Callable[[dict[str, str]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    required: tuple[str, ...]
    handler: ToolHandler


class ToolRegistry:
    """Хранить только явно зарегистрированные функции, без eval или shell."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Повтор имени инструмента")

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "description": description}
                            for name, description in tool.parameters.items()
                        },
                        "required": list(tool.required),
                        "additionalProperties": False,
                    },
                },
            }
            for tool in self.tools.values()
        ]


class ToolExecutor:
    """Проверять аргументы, ограничивать время и размер результата."""

    def __init__(self, registry: ToolRegistry, timeout: float = 12.0) -> None:
        self.registry = registry
        self.timeout = timeout

    async def execute(self, name: str, raw_arguments: str) -> str:
        started = perf_counter()
        outcome = "error"
        try:
            if name not in self.registry.tools or len(raw_arguments) > 4096:
                raise ValueError("Неизвестный инструмент или слишком большие аргументы")
            tool = self.registry.tools[name]
            args = json.loads(raw_arguments)
            if (
                not isinstance(args, dict)
                or set(args) - set(tool.parameters)
                or set(tool.required) - set(args)
                or any(not isinstance(v, str) or len(v) > 200 for v in args.values())
            ):
                raise ValueError("Некорректные аргументы инструмента")
            result = await asyncio.wait_for(tool.handler(args), self.timeout)
            output = json.dumps({"ok": True, "data": result}, ensure_ascii=False)
            if len(output) > 16000:
                raise ValueError("Ответ источника слишком большой")
            outcome = "ok"
            return output
        except ValueError, KeyError, TypeError:
            return json.dumps({"ok": False, "error": "invalid_arguments_or_data"})
        except Exception:
            # Не передаём модели исключения с URL, ключами или содержимым запросов.
            return json.dumps({"ok": False, "error": "source_unavailable"})
        finally:
            logger.info(
                "Tool name=%s outcome=%s duration=%.3fs",
                name if name in self.registry.tools else "unknown",
                outcome,
                perf_counter() - started,
            )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


async def _fetch(
    url: str, parameters: dict[str, str], *, proxy_url: str | None = None
) -> bytes:
    """Запрашивать только фиксированные адреса провайдеров, максимум 1 MiB."""

    if proxy_url:
        connector = ProxyConnector.from_url(proxy_url)
        async with ClientSession(
            connector=connector, timeout=ClientTimeout(total=8)
        ) as session:
            async with session.get(
                url, params=parameters, allow_redirects=False
            ) as response:
                if response.status != 200:
                    raise ValueError("Источник вернул ошибку")
                data = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    data.extend(chunk)
                    if len(data) > 1_048_576:
                        raise ValueError("Ответ слишком большой")
                return bytes(data)

    def read() -> bytes:
        request = Request(
            url + "?" + urlencode(parameters),
            headers={"User-Agent": "ProtogenDelta/0.1 (live data)"},
        )
        with build_opener(NoRedirect()).open(request, timeout=5) as response:
            body = response.read(1_048_577)
        if len(body) > 1_048_576:
            raise ValueError("Ответ слишком большой")
        return cast(bytes, body)

    return await asyncio.to_thread(read)


async def get_current_time(args: dict[str, str]) -> dict[str, Any]:
    zone = args.get("timezone", "Europe/Moscow")
    if zone not in ("Europe/Moscow", "UTC"):
        raise ValueError("Поддерживаются Europe/Moscow и UTC")
    now = datetime.now(MOSCOW if zone == "Europe/Moscow" else timezone.utc)
    return {"datetime": now.isoformat(), "timezone": zone, "source": "system_clock"}


async def get_exchange_rate(
    args: dict[str, str], *, proxy_url: str | None = None
) -> dict[str, Any]:
    base, quote = args.get("base", "USD").upper(), args.get("quote", "RUB").upper()
    if not all(re.fullmatch("[A-Z]{3}", code) for code in (base, quote)):
        raise ValueError("Нужны трёхбуквенные коды валют")
    url = "https://www.cbr.ru/scripts/XML_daily.asp"
    data = await _fetch(
        url,
        {"date_req": datetime.now(MOSCOW).strftime("%d/%m/%Y")},
        proxy_url=proxy_url,
    )
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("Некорректный XML")
    root = ElementTree.fromstring(data)
    rates = {"RUB": Decimal(1)}
    for item in root.findall("Valute"):
        code = item.findtext("CharCode", "")
        rates[code] = Decimal(item.findtext("Value", "0").replace(",", ".")) / Decimal(
            item.findtext("Nominal", "1")
        )
    rate = rates[base] / rates[quote]
    if not rate.is_finite() or rate <= 0:
        raise ValueError("Некорректный курс")
    return {
        "base": base,
        "quote": quote,
        "rate": str(rate.quantize(Decimal("0.000001"))),
        "effective_date": root.attrib["Date"],
        "source": url,
        "kind": "Официальный курс ЦБ РФ, не курс покупки/продажи банка",
    }


async def get_weather(
    args: dict[str, str], *, proxy_url: str | None = None
) -> dict[str, Any]:
    city = args["city"].strip()
    country = args.get("country_code", "").upper()
    if len(city) < 2 or (country and not re.fullmatch("[A-Z]{2}", country)):
        raise ValueError("Укажите город и при необходимости ISO-код страны")
    query = {"name": city, "count": "3", "language": "ru", "format": "json"}
    if country:
        query["countryCode"] = country
    locations = json.loads(
        await _fetch(
            "https://geocoding-api.open-meteo.com/v1/search", query, proxy_url=proxy_url
        )
    ).get("results", [])
    region = args.get("region", "").strip().casefold()
    if region:
        locations = [
            place
            for place in locations
            if str(place.get("admin1", "")).casefold() == region
        ]
    if not locations:
        return {"status": "not_found", "city": city}
    if len(locations) > 1:
        return {
            "status": "ambiguous",
            "instruction": "Уточни у пользователя страну/регион; не выбирай молча",
            "candidates": [
                {key: place.get(key) for key in ("name", "country", "admin1")}
                for place in locations
            ],
        }
    place = locations[0]
    url = "https://api.open-meteo.com/v1/forecast"
    weather = json.loads(
        await _fetch(
            url,
            {
                "latitude": str(place["latitude"]),
                "longitude": str(place["longitude"]),
                "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": "2",
                "timezone": "auto",
            },
            proxy_url=proxy_url,
        )
    )
    return {
        "place": {key: place.get(key) for key in ("name", "country", "admin1")},
        "timezone": weather["timezone"],
        "current": weather["current"],
        "current_units": weather["current_units"],
        "daily": weather["daily"],
        "daily_units": weather["daily_units"],
        "source": "https://open-meteo.com/",
    }


def default_registry(*, proxy_url: str | None = None) -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "get_current_time",
                "Точная текущая дата и время системных часов.",
                {"timezone": "Europe/Moscow (по умолчанию) или UTC"},
                (),
                get_current_time,
            ),
            Tool(
                "get_exchange_rate",
                "Актуальный официальный курс ЦБ РФ с датой действия.",
                {
                    "base": "Код валюты, например USD",
                    "quote": "Код валюты, например RUB",
                },
                ("base", "quote"),
                partial(get_exchange_rate, proxy_url=proxy_url),
            ),
            Tool(
                "get_weather",
                "Погода сейчас и прогноз на сегодня/завтра. Город обязателен.",
                {
                    "city": "Название города",
                    "country_code": "ISO-код страны: RU, DE и т.д.",
                    "region": "Регион из списка кандидатов при неоднозначном городе",
                },
                ("city",),
                partial(get_weather, proxy_url=proxy_url),
            ),
        ]
    )
