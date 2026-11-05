"""Ключ Foreca, авторизация, отсутствие покрытия и ограничение платных запросов."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp import lightning as module
from protogen_delta.miniapp.server import MiniAppServer


def _headers(user: int = 42) -> dict[str, str]:
    return {"X-Telegram-Init-Data": _signed_init_data(user_id=user)}


def _flash(**changes: object) -> dict[str, object]:
    return {
        "lat": 55.8,
        "lon": 37.6,
        "time": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        **changes,
    }


@pytest.mark.parametrize("key", [None, "synthetic-secret"])
def test_lightning_always_requires_telegram_auth(
    monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    fetch = AsyncMock()
    monkeypatch.setattr(module, "fetch_provider", fetch)

    async def scenario() -> None:
        server = MiniAppServer(TOKEN, UserStateStore(), foreca_api_key=key)
        async with TestClient(TestServer(server.application())) as client:
            response = await client.get(
                "/api/weather/lightning?latitude=55&longitude=37"
            )
            assert response.status == 401
        fetch.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "query",
    [
        "latitude=nan&longitude=37",
        "latitude=55&longitude=inf",
        "latitude=91&longitude=37",
        "latitude=55&longitude=-181",
        "latitude=x&longitude=37",
        "latitude=55&longitude=37&latitude=54",
        "latitude=55&longitude=37&token=unexpected",
        "latitude=55",
    ],
)
def test_lightning_rejects_invalid_queries(query: str) -> None:
    async def scenario() -> None:
        server = MiniAppServer(TOKEN, UserStateStore())
        async with TestClient(TestServer(server.application())) as client:
            response = await client.get(
                "/api/weather/lightning?" + query, headers=_headers()
            )
            assert response.status == 400

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("key", "query", "status"),
    [
        (None, "latitude=55&longitude=37", "disabled"),
        ("synthetic-secret", "latitude=55&longitude=130", "outside_coverage"),
        ("synthetic-secret", "latitude=80&longitude=37", "outside_coverage"),
    ],
)
def test_disabled_and_uncovered_never_call_provider(
    monkeypatch: pytest.MonkeyPatch, key: str | None, query: str, status: str
) -> None:
    fetch = AsyncMock()
    monkeypatch.setattr(module, "fetch_provider", fetch)

    async def scenario() -> None:
        server = MiniAppServer(TOKEN, UserStateStore(), foreca_api_key=key)
        async with TestClient(TestServer(server.application())) as client:
            response = await client.get(
                "/api/weather/lightning?" + query, headers=_headers()
            )
            assert response.status == 200
            assert await response.json() == {"status": status}
        fetch.assert_not_awaited()

    asyncio.run(scenario())


def test_foreca_uses_server_header_and_shared_bounded_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _flash()
    fetch = AsyncMock(return_value=json.dumps({"strikes": [at]}).encode())
    monkeypatch.setattr(module, "fetch_provider", fetch)
    clock = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])

    async def scenario() -> None:
        server = MiniAppServer(
            TOKEN, UserStateStore(), foreca_api_key="synthetic-secret"
        )
        async with TestClient(TestServer(server.application())) as client:
            for user in (42, 43):
                response = await client.get(
                    "/api/weather/lightning?latitude=55.75&longitude=37.61",
                    headers=_headers(user),
                )
                assert response.status == 200
                body = await response.json()
                assert body["strikes"][0]["latitude"] == at["lat"]
                assert body["minutes"] == 30 and body["radius_km"] == 100
                assert "synthetic-secret" not in json.dumps(body)
                assert response.headers["Cache-Control"] == "no-store"
            fetch.assert_awaited_once_with(
                "https://weatherapi.foreca.net/api/v1/lightning/37.61,55.75",
                {"radius": "100", "limit": "1000"},
                headers={"Authorization": "Bearer synthetic-secret"},
                proxy_url=None,
            )
            clock[0] += 600
            assert (
                await client.get(
                    "/api/weather/lightning?latitude=55.75&longitude=37.61",
                    headers=_headers(44),
                )
            ).status == 200
            assert fetch.await_count == 2
            server._lightning_routes._attempts = 144
            assert (
                await client.get(
                    "/api/weather/lightning?latitude=50&longitude=35",
                    headers=_headers(45),
                )
            ).status == 429
            assert fetch.await_count == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"strikes": None},
        {"strikes": [_flash(lat=True)]},
        {"strikes": [_flash(lon=181)]},
        {"strikes": [_flash(time="yesterday")]},
        {
            "strikes": [
                _flash(
                    time=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
                )
            ]
        },
    ],
)
def test_invalid_foreca_data_never_becomes_no_lightning(payload: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        module.LightningRoutes._parse(payload)


def test_empty_covered_response_is_distinct_from_disabled() -> None:
    result = module.LightningRoutes._parse({"strikes": []})
    assert result["status"] == "ok" and result["strikes"] == []
    assert result["possibly_truncated"] is False


def test_provider_errors_hide_secrets_and_back_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = AsyncMock(side_effect=RuntimeError("synthetic-secret"))
    monkeypatch.setattr(module, "fetch_provider", fetch)

    async def scenario() -> None:
        server = MiniAppServer(
            TOKEN, UserStateStore(), foreca_api_key="synthetic-secret"
        )
        async with TestClient(TestServer(server.application())) as client:
            response = await client.get(
                "/api/weather/lightning?latitude=55&longitude=37", headers=_headers()
            )
            assert response.status == 502
            assert "synthetic-secret" not in await response.text()
            response = await client.get(
                "/api/weather/lightning?latitude=55&longitude=37", headers=_headers(43)
            )
            assert response.status == 503
            fetch.assert_awaited_once()

    asyncio.run(scenario())
