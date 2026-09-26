"""Публичный интерфейс системы внешних инструментов."""

from functools import partial

from protogen_delta.services.tools.base import Tool, ToolExecutor, ToolRegistry
from protogen_delta.services.tools.live_data import (
    get_current_time,
    get_exchange_rate,
    get_weather,
)
from protogen_delta.services.tools.web import fetch_web_page, web_search

__all__ = [
    "Tool",
    "ToolExecutor",
    "ToolRegistry",
    "default_registry",
    "fetch_web_page",
    "get_current_time",
    "get_exchange_rate",
    "get_weather",
    "web_search",
]


def default_registry(
    *,
    proxy_url: str | None = None,
    brave_search_api_key: str | None = None,
) -> ToolRegistry:
    """Создать стандартный набор доступных DeepSeek инструментов."""
    tools = [
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
        Tool(
            "fetch_web_page",
            "Прочитать текст публичной HTTP(S)-страницы по точному URL пользователя или поиска.",
            {"url": "Полный публичный URL страницы"},
            ("url",),
            partial(fetch_web_page, proxy_url=proxy_url),
        ),
    ]
    if brave_search_api_key:
        tools.append(
            Tool(
                "web_search",
                "Найти актуальную информацию в интернете с URL и датами источников.",
                {
                    "query": "Краткий поисковый запрос",
                    "count": "Число результатов от 1 до 10, по умолчанию 5",
                    "language": "Язык результатов: ru или en",
                    "freshness": "Период: day, week, month, year или пусто",
                },
                ("query",),
                partial(
                    web_search,
                    api_key=brave_search_api_key,
                    proxy_url=proxy_url,
                ),
            )
        )
    return ToolRegistry(tools)
