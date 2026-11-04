"""Авторизация, кеш, ошибки источника и изоляция страницы погоды."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp import weather as weather_module
from protogen_delta.miniapp.server import MiniAppServer


def test_weather_api_requires_auth_and_reuses_only_public_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = AsyncMock(return_value={"current": {"temperature_2m": 9}})
    monkeypatch.setattr(weather_module, "get_weather", fetch)

    async def scenario() -> None:
        async with TestClient(
            TestServer(
                MiniAppServer(
                    TOKEN, UserStateStore(), weather_proxy_url="socks5://proxy:1080"
                ).application()
            )
        ) as client:
            assert (await client.get("/api/weather?city=Элиста")).status == 401
            fetch.assert_not_awaited()
            for user in (42, 43):
                response = await client.get(
                    "/api/weather?city=Элиста&country_code=RU",
                    headers={"X-Telegram-Init-Data": _signed_init_data(user_id=user)},
                )
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert await response.json() == {"current": {"temperature_2m": 9}}
            fetch.assert_awaited_once_with(
                {"city": "Элиста", "country_code": "RU"},
                proxy_url="socks5://proxy:1080",
                forecast_days=10,
            )
            assert (
                await client.get(
                    "/api/weather?city=Элиста",
                    headers={"X-Telegram-Init-Data": _signed_init_data()},
                )
            ).status == 429

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "query",
    [
        "city=x",
        "city=Москва&user_id=999",
        "city=Москва&country_code=bad",
        "city=Москва&city=Казань",
    ],
)
def test_weather_api_rejects_invalid_queries(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    fetch = AsyncMock()
    monkeypatch.setattr(weather_module, "get_weather", fetch)

    async def scenario() -> None:
        async with TestClient(
            TestServer(MiniAppServer(TOKEN, UserStateStore()).application())
        ) as client:
            response = await client.get(
                "/api/weather?" + query,
                headers={"X-Telegram-Init-Data": _signed_init_data()},
            )
            assert response.status == 400
        fetch.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error", [ValueError("private detail"), RuntimeError("secret URL")]
)
def test_weather_source_errors_hide_internal_details(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(weather_module, "get_weather", AsyncMock(side_effect=error))

    async def scenario() -> None:
        async with TestClient(
            TestServer(MiniAppServer(TOKEN, UserStateStore()).application())
        ) as client:
            response = await client.get(
                "/api/weather?city=Москва",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
            )
            assert response.status == 502
            assert str(error) not in await response.text()

    asyncio.run(scenario())


def test_weather_assets_and_csp_allow_only_selected_map_origin() -> None:
    async def scenario() -> None:
        async with TestClient(
            TestServer(MiniAppServer(TOKEN, UserStateStore()).application())
        ) as client:
            response = await client.get("/")
            assert (
                "frame-src https://embed.windy.com;"
                in response.headers["Content-Security-Policy"]
            )
            html = await response.text()
            assert 'id="tab-weather"' in html and 'id="weather-panel"' in html
            assert "Прогноз на 10 дней" in html
            script = await client.get("/static/weather.js")
            assert (
                script.status == 200 and script.content_type == "application/javascript"
            )
            assert "X-Telegram-Init-Data" in await script.text()

    asyncio.run(scenario())


def test_weather_cache_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(weather_module, "monotonic", lambda: clock[0])
    fetch = AsyncMock(return_value={"status": "not_found"})
    monkeypatch.setattr(weather_module, "get_weather", fetch)

    async def scenario() -> None:
        async with TestClient(
            TestServer(MiniAppServer(TOKEN, UserStateStore()).application())
        ) as client:
            for user, tick in ((42, 0), (43, 299), (44, 301)):
                clock[0] = tick
                response = await client.get(
                    "/api/weather?city=Город",
                    headers={"X-Telegram-Init-Data": _signed_init_data(user_id=user)},
                )
                assert response.status == 200
            assert fetch.await_count == 2

    asyncio.run(scenario())
