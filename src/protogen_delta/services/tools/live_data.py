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


def _weather_condition(code: object) -> str | None:
    """Расшифровать WMO-код без догадок при пропущенном значении."""
    descriptions = {
        0: "Ясно",
        1: "Преимущественно ясно",
        2: "Переменная облачность",
        3: "Пасмурно",
        45: "Туман",
        48: "Туман с изморозью",
        51: "Слабая морось",
        53: "Умеренная морось",
        55: "Сильная морось",
        56: "Слабая переохлаждённая морось",
        57: "Сильная переохлаждённая морось",
        61: "Слабый дождь",
        63: "Умеренный дождь",
        65: "Сильный дождь",
        66: "Слабый переохлаждённый дождь",
        67: "Сильный переохлаждённый дождь",
        71: "Слабый снег",
        73: "Умеренный снег",
        75: "Сильный снег",
        77: "Снежные зёрна",
        80: "Слабый ливень",
        81: "Умеренный ливень",
        82: "Сильный ливень",
        85: "Слабый снежный ливень",
        86: "Сильный снежный ливень",
        95: "Гроза",
        96: "Гроза со слабым градом",
        99: "Гроза с сильным градом",
    }
    return descriptions.get(code) if type(code) is int else None


async def get_weather(
    args: dict[str, str], *, proxy_url: str | None = None, forecast_days: int = 2
) -> dict[str, Any]:
    """Получить условия сейчас, 24 часа осадков и двухдневный прогноз."""
    if type(forecast_days) is not int or not 2 <= forecast_days <= 10:
        raise ValueError("Прогноз доступен на 2–10 дней")
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
    # Геокодер возвращает также аэропорты и нечёткие совпадения.
    locations = [
        p
        for p in locations
        if not p.get("feature_code") or str(p["feature_code"]).startswith("PPL")
    ]
    exact = [
        p for p in locations if str(p.get("name", "")).casefold() == city.casefold()
    ]
    if exact:
        locations = exact
    location_id = args.get("location_id", "")
    if location_id:
        if not re.fullmatch(r"[0-9]{1,20}", location_id):
            raise ValueError("Некорректный ID города")
        locations = [p for p in locations if str(p.get("id", "")) == location_id]
    region = args.get("region", "").strip().casefold()
    if region:
        chosen = [
            place
            for place in locations
            if str(place.get("admin1", "")).casefold() == region
        ]
        if not chosen:
            # Модель может указать английское имя региона, а русская выдача
            # содержит «Миссури». Сопоставляем по ID, не угадывая регион.
            english = json.loads(
                await fetch_provider(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    {**query, "language": "en"},
                    proxy_url=proxy_url,
                )
            ).get("results", [])
            matching_ids = {
                place.get("id")
                for place in english
                if place.get("id") is not None
                and str(place.get("admin1", "")).casefold() == region
            }
            chosen = [place for place in locations if place.get("id") in matching_ids]
        locations = chosen
    if not locations:
        return {"status": "not_found", "city": city}
    if country and not region and len(locations) > 1:
        first_population = int(locations[0].get("population", 0) or 0)
        other_population = max(
            int(place.get("population", 0) or 0) for place in locations[1:]
        )
        # Явная страна и мегаполис среди крошечных тёзок: Москва, Россия
        # не требует выбора между столицей и селом. Springfield остаётся неоднозначным.
        if first_population >= 500000 and first_population > 100 * max(
            1, other_population
        ):
            locations = locations[:1]
    if len(locations) > 1:
        return {
            "status": "ambiguous",
            "instruction": "Уточни у пользователя страну/регион; не выбирай молча",
            "candidates": [
                {
                    key: place.get(key)
                    for key in (
                        "name",
                        "country",
                        "admin1",
                        "country_code",
                        "id",
                        "admin2",
                    )
                }
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
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,rain,showers,snowfall,weather_code,surface_pressure,"
                    "wind_speed_10m,wind_gusts_10m"
                ),
                "hourly": (
                    "temperature_2m,apparent_temperature,precipitation_probability,"
                    "precipitation,rain,showers,snowfall,weather_code,"
                    "wind_speed_10m,wind_gusts_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,precipitation_sum,rain_sum,"
                    "showers_sum,snowfall_sum,wind_gusts_10m_max"
                ),
                "forecast_hours": "24",
                "wind_speed_unit": "ms",
                "forecast_days": str(forecast_days),
                "timezone": "Europe/Moscow",
            },
            proxy_url=proxy_url,
        )
    )
    return {
        "place": {
            key: place.get(key)
            for key in ("name", "country", "admin1", "country_code", "id", "admin2")
        },
        "coordinates": {"latitude": place["latitude"], "longitude": place["longitude"]},
        "timezone": weather["timezone"],
        "current": weather["current"],
        "current_units": weather["current_units"],
        "daily": weather["daily"],
        "daily_units": weather["daily_units"],
        "hourly": weather.get("hourly", {}),
        "hourly_units": weather.get("hourly_units", {}),
        "conditions": {
            "current": _weather_condition(weather["current"].get("weather_code")),
            "hourly": [
                _weather_condition(code)
                for code in weather.get("hourly", {}).get("weather_code", [])
            ],
            "daily": [
                _weather_condition(code)
                for code in weather["daily"].get("weather_code", [])
            ],
        },
        "utc_offset_seconds": weather.get("utc_offset_seconds"),
        "source": "https://open-meteo.com/",
    }
