"""Инструменты времени, официального курса валют и погоды."""

import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from xml.etree import ElementTree

from protogen_delta.services.tools.http import fetch_provider

MOSCOW = timezone(timedelta(hours=3), "Europe/Moscow")


async def get_current_time(args: dict[str, str]) -> dict[str, Any]:
    """Вернуть точное текущее время в поддерживаемой временной зоне."""
    zone = args.get("timezone", "Europe/Moscow")
    if zone not in ("Europe/Moscow", "UTC"):
        raise ValueError("Поддерживаются Europe/Moscow и UTC")
    now = datetime.now(MOSCOW if zone == "Europe/Moscow" else timezone.utc)
    return {"datetime": now.isoformat(), "timezone": zone, "source": "system_clock"}


async def get_exchange_rate(
    args: dict[str, str], *, proxy_url: str | None = None
) -> dict[str, Any]:
    """Получить официальный валютный курс ЦБ РФ."""
    base, quote = args.get("base", "USD").upper(), args.get("quote", "RUB").upper()
    if not all(re.fullmatch("[A-Z]{3}", code) for code in (base, quote)):
        raise ValueError("Нужны трёхбуквенные коды валют")
    url = "https://www.cbr.ru/scripts/XML_daily.asp"
    data = await fetch_provider(
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
    """Получить текущую погоду и двухдневный прогноз Open-Meteo."""
    city = args["city"].strip()
    country = args.get("country_code", "").upper()
    if len(city) < 2 or (country and not re.fullmatch("[A-Z]{2}", country)):
        raise ValueError("Укажите город и при необходимости ISO-код страны")
    query = {"name": city, "count": "3", "language": "ru", "format": "json"}
    if country:
        query["countryCode"] = country
    locations = json.loads(
        await fetch_provider(
            "https://geocoding-api.open-meteo.com/v1/search",
            query,
            proxy_url=proxy_url,
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
        await fetch_provider(
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
