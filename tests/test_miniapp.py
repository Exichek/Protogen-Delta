"""Тесты подписи и защищённого API Telegram Mini App."""

import asyncio
import hashlib
import hmac
import json
import socket
from time import time
from typing import cast
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.auth import MiniAppAuthError, validate_init_data
from protogen_delta.miniapp.server import MiniAppServer

TOKEN = "123456:test-token"


def _signed_init_data(
    *,
    user_id: int = 42,
    auth_date: int | None = None,
    first_name: str = "Тест",
    signature: str | None = None,
) -> str:
    values = {
        "auth_date": str(int(time()) if auth_date is None else auth_date),
        "query_id": "query-1",
        "user": json.dumps(
            {"id": user_id, "first_name": first_name, "username": "tester"},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    if signature is not None:
        values["signature"] = signature
    check = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    secret = hmac.new(b"WebAppData", TOKEN.encode(), hashlib.sha256).digest()
    values["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return urlencode(values)


def test_validate_init_data_returns_verified_user() -> None:
    """Корректная Telegram-подпись должна открыть только свой user_id."""
    user = validate_init_data(_signed_init_data(), TOKEN)

    assert user.id == 42
    assert user.first_name == "Тест"
    assert user.username == "tester"


def test_validate_init_data_rejects_tampering_and_expiration() -> None:
    """Изменённые и старые initData нельзя использовать для доступа к профилю."""
    signed = _signed_init_data()
    with pytest.raises(MiniAppAuthError, match="подпись"):
        validate_init_data(signed.replace("tester", "hacker"), TOKEN)
    with pytest.raises(MiniAppAuthError, match="просрочен"):
        validate_init_data(
            _signed_init_data(auth_date=100),
            TOKEN,
            max_age_seconds=60,
            now=1000,
        )


def test_validate_init_data_includes_telegram_signature_in_hmac() -> None:
    """Современный initData содержит signature, которая также защищена HMAC."""
    signed = _signed_init_data(signature="telegram-signature")
    assert validate_init_data(signed, TOKEN).id == 42
    with pytest.raises(MiniAppAuthError, match="подпись"):
        validate_init_data(signed.replace("telegram-signature", "changed"), TOKEN)


def test_validate_init_data_rejects_missing_fields() -> None:
    """Пустые и неполные данные получают понятную ошибку авторизации."""
    with pytest.raises(MiniAppAuthError, match="отсутствует"):
        validate_init_data("", TOKEN)
    with pytest.raises(MiniAppAuthError, match="hash"):
        validate_init_data("auth_date=123", TOKEN)
    with pytest.raises(ValueError, match="max_age_seconds"):
        validate_init_data("anything", TOKEN, max_age_seconds=0)


def test_validate_init_data_rejects_invalid_dates_and_users() -> None:
    """Свежая подпись не делает некорректные Telegram-поля доверенными."""
    with pytest.raises(MiniAppAuthError, match="просрочен"):
        validate_init_data(
            _signed_init_data(auth_date=2000),
            TOKEN,
            max_age_seconds=60,
            now=1000,
        )

    def sign_values(values: dict[str, str]) -> str:
        check = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
        secret = hmac.new(b"WebAppData", TOKEN.encode(), hashlib.sha256).digest()
        values["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
        return urlencode(values)

    invalid_date = sign_values(
        {
            "auth_date": "bad",
            "query_id": "query-1",
            "user": '{"id":42,"first_name":"Test"}',
        }
    )
    with pytest.raises(MiniAppAuthError, match="auth_date"):
        validate_init_data(invalid_date, TOKEN, now=1000)

    for raw_user in ("not-json", '{"id":true,"first_name":"Test"}'):
        signed = sign_values(
            {"auth_date": "1000", "query_id": "query-1", "user": raw_user}
        )
        with pytest.raises(MiniAppAuthError, match="user"):
            validate_init_data(signed, TOKEN, now=1000)


def test_miniapp_profile_api_reads_and_updates_own_state() -> None:
    """Подписанный пользователь может читать и менять разрешённые настройки."""
    states = UserStateStore()
    application = MiniAppServer(TOKEN, states).application()

    async def scenario() -> None:
        async with TestClient(TestServer(application)) as client:
            health = await client.get("/health")
            assert health.status == 200
            index = await client.get("/")
            assert index.status == 200
            assert "Протоген Дельта" in await index.text()

            unauthorized = await client.get("/api/profile")
            assert unauthorized.status == 401

            headers = {"X-Telegram-Init-Data": _signed_init_data()}
            profile = await client.get("/api/profile", headers=headers)
            assert profile.status == 200
            initial = await profile.json()
            assert initial["user"]["id"] == 42
            assert initial["content_mode"] == "unselected"

            updated = await client.patch(
                "/api/profile",
                headers=headers,
                json={
                    "content_mode": "adult",
                    "roleplay_active": True,
                    "roleplay_configuration": "female",
                    "roleplay_character": "Синий дракон",
                    "roleplay_preferences": "Медленная сцена и юмор",
                    "roleplay_boundaries": "Без унижения",
                },
            )
            assert updated.status == 200
            payload = await updated.json()
            assert payload["content_mode"] == "adult"
            assert payload["roleplay_character"] == "Синий дракон"
            assert payload["roleplay_preferences"] == "Медленная сцена и юмор"
            assert payload["roleplay_boundaries"] == "Без унижения"

    asyncio.run(scenario())
    state = states.get(42)
    assert state.content_mode == "adult"
    assert state.roleplay_active is True
    assert state.roleplay_configuration == "female"
    assert state.roleplay_preferences == "Медленная сцена и юмор"
    assert state.roleplay_boundaries == "Без унижения"


def test_miniapp_profile_api_rejects_unknown_or_invalid_settings() -> None:
    """API принимает только ограниченную схему профиля."""
    application = MiniAppServer(TOKEN, UserStateStore()).application()

    async def scenario() -> None:
        async with TestClient(TestServer(application)) as client:
            headers = {"X-Telegram-Init-Data": _signed_init_data()}
            unknown = await client.patch(
                "/api/profile", headers=headers, json={"admin": True}
            )
            assert unknown.status == 400
            invalid = await client.patch(
                "/api/profile", headers=headers, json={"content_mode": "unsafe"}
            )
            assert invalid.status == 400
            too_long = await client.patch(
                "/api/profile",
                headers=headers,
                json={"roleplay_character": "x" * 1001},
            )
            assert too_long.status == 400
            long_preferences = await client.patch(
                "/api/profile",
                headers=headers,
                json={"roleplay_preferences": "x" * 1001},
            )
            assert long_preferences.status == 400
            invalid_rp = await client.patch(
                "/api/profile", headers=headers, json={"roleplay_active": "yes"}
            )
            assert invalid_rp.status == 400
            invalid_configuration = await client.patch(
                "/api/profile",
                headers=headers,
                json={"roleplay_configuration": "robot"},
            )
            assert invalid_configuration.status == 400
            wrong_values: tuple[object, ...] = ([], {})
            for field in ("content_mode", "roleplay_configuration"):
                for value in wrong_values:
                    wrong_type = await client.patch(
                        "/api/profile", headers=headers, json={field: value}
                    )
                    assert wrong_type.status == 400
            invalid_body = await client.patch(
                "/api/profile",
                headers={**headers, "Content-Type": "application/json"},
                data="not-json",
            )
            assert invalid_body.status == 400
            array_body = await client.patch(
                "/api/profile", headers=headers, json=["soft"]
            )
            assert array_body.status == 400

    asyncio.run(scenario())


def test_miniapp_can_clear_only_roleplay_profile() -> None:
    """Удаление RP-профиля не должно стирать режим контента и облик Дельты."""
    states = UserStateStore()
    state = states.get(42)
    state.content_mode = "adult"
    state.delta_appearance = "синий дракон"
    state.roleplay_active = True
    state.roleplay_configuration = "female"
    state.roleplay_character = "лиса"
    state.roleplay_fetishes = ("bondage",)
    state.roleplay_preferences = "медленно"
    state.roleplay_boundaries = "без боли"
    state.emotions.arousal = 0.8
    application = MiniAppServer(TOKEN, states).application()

    async def scenario() -> None:
        async with TestClient(TestServer(application)) as client:
            response = await client.delete(
                "/api/profile/roleplay",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["roleplay_character"] == ""
            assert payload["roleplay_fetishes"] == []
            assert payload["roleplay_preferences"] == ""
            assert payload["roleplay_boundaries"] == ""
            assert payload["content_mode"] == "adult"
            assert payload["delta_appearance"] == "синий дракон"

    asyncio.run(scenario())
    assert state.roleplay_active is False
    assert state.roleplay_configuration == "male"
    assert state.emotions.arousal == 0.0


def test_miniapp_server_validates_bind_configuration() -> None:
    """Невозможные параметры встроенного HTTP-сервера отклоняются сразу."""
    states = UserStateStore()
    with pytest.raises(ValueError, match="host"):
        MiniAppServer(TOKEN, states, host="")
    with pytest.raises(ValueError, match="port"):
        MiniAppServer(TOKEN, states, port=0)
    with pytest.raises(ValueError, match="auth_max_age_seconds"):
        MiniAppServer(TOKEN, states, auth_max_age_seconds=0)


def test_miniapp_server_starts_and_stops_idempotently() -> None:
    """Встроенный сервер должен освобождать порт и терпеть повторные вызовы."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = cast(tuple[str, int], probe.getsockname())[1]
    server = MiniAppServer(TOKEN, UserStateStore(), port=port)

    async def scenario() -> None:
        await server.start()
        await server.start()
        await server.close()
        await server.close()

    asyncio.run(scenario())
