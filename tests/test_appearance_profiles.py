"""Раздельные профили, миграция и анализ собственного персонажа без API."""

import asyncio
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import pytest
from aiohttp.test_utils import TestClient, TestServer
from appearance_fixtures import verified
from test_miniapp import TOKEN, _signed_init_data
from test_miniapp_appearance import _image
from test_response_engine import _create_engine

from protogen_delta.core.appearance_profile import AppearanceProfile
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.appearance_profile_context import (
    appearance_profile_context,
)
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.response_engine import AppearanceAnalysisError


def test_delta_profile_persists_without_changing_user_or_image(tmp_path: Path) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        async with states.use(42) as state:
            state.delta_appearance = "Синяя шерсть Дельты"
            state.delta_appearance_thumbnail = "old-thumbnail"
            state.roleplay_character = "Мой дракон"
            state.roleplay_preferences = "USER_PREF"
            state.history.append(ConversationTurn("history", "unchanged"))
        async with states.use_conversation(42, -100) as state:
            state.delta_appearance = "GROUP"
        server = MiniAppServer(TOKEN, states, response_engine=engine)
        payload = {
            "profile": asdict(
                AppearanceProfile("Упрямый", "Говорит коротко", "Загадки", "Без драк")
            ),
            "expected_appearance": "Синяя шерсть Дельты",
            "expected_profile": asdict(AppearanceProfile()),
        }
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        path = "/api/profile/appearance/behavior"
        async with TestClient(TestServer(server.application())) as client:
            assert (await client.patch(path, json=payload)).status == 401
            response = await client.patch(path, headers=headers, json=payload)
            assert response.status == 200
            saved = await response.json()
            assert saved["delta_appearance_profile"] == payload["profile"]
            assert saved["delta_appearance_thumbnail"] == "old-thumbnail"
            assert saved["roleplay_character"] == "Мой дракон"
            assert saved["roleplay_preferences"] == "USER_PREF"
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 400
            other = await client.get(
                "/api/profile",
                headers={"X-Telegram-Init-Data": _signed_init_data(user_id=7)},
            )
            assert (await other.json())["delta_appearance_profile"] == asdict(
                AppearanceProfile()
            )
        reopened = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with reopened.use(42) as state:
            assert AppearanceProfile.decode(
                state.delta_appearance_profile
            ) == AppearanceProfile.from_payload(payload["profile"])
            assert state.history[-1].user_message == "history"
        async with reopened.use_conversation(42, -100) as state:
            assert (
                state.delta_appearance == "GROUP" and not state.delta_appearance_profile
            )
        model.chat.assert_not_awaited()
        model.analyze_visual_features.assert_not_awaited()

    asyncio.run(scenario())


def test_delta_profile_rejects_invalid_stale_or_busy_writes() -> None:
    async def scenario() -> None:
        engine, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.delta_appearance = "saved"
        server = MiniAppServer(TOKEN, engine._user_states, response_engine=engine)
        payload = {
            "profile": asdict(AppearanceProfile()),
            "expected_profile": asdict(AppearanceProfile()),
            "expected_appearance": "saved",
        }
        path = "/api/profile/appearance/behavior"
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            bad: object
            for bad in (
                [],
                {},
                {**payload, "extra": True},
                {**payload, "expected_appearance": []},
                {**payload, "expected_appearance": ""},
                {**payload, "profile": {}},
                {**payload, "profile": {**asdict(AppearanceProfile()), "behavior": []}},
                {
                    **payload,
                    "profile": {
                        **asdict(AppearanceProfile()),
                        "personality": "x" * 1001,
                    },
                },
            ):
                assert (
                    await client.patch(path, headers=headers, json=bad)
                ).status == 400
                assert not server._appearance_pending
            state.delta_appearance = "changed"
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 400
            state.delta_appearance = "saved"
            engine._delivering_users.add(42)
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 409
            engine._delivering_users.remove(42)
            server._appearance_pending.add(42)
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 409
            server._appearance_pending.clear()
            state.delta_appearance = ""
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 400
        absent = MiniAppServer(TOKEN, UserStateStore())
        async with TestClient(TestServer(absent.application())) as client:
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 503

    asyncio.run(scenario())


def test_delta_profile_follows_identity_and_stays_outside_ordinary_chat() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        profile = AppearanceProfile(
            "DELTA_PERSONALITY", "DELTA_BEHAVIOR", "DELTA_PREF", "DELTA_LIMIT"
        )
        state.delta_appearance = "Синий дракон"
        state.delta_appearance_profile = profile.encode()
        state.roleplay_character = "USER_CHARACTER"
        state.roleplay_preferences = "USER_PREF"
        state.roleplay_boundaries = "USER_LIMIT"
        state.roleplay_active = True
        await engine.respond(42, "*показываю карту леса*")
        call = model.chat.await_args
        assert call is not None
        prompt = call.kwargs["system_prompt"]
        for marker in (
            "DELTA_PERSONALITY",
            "DELTA_BEHAVIOR",
            "DELTA_PREF",
            "DELTA_LIMIT",
            "USER_CHARACTER",
            "USER_PREF",
            "USER_LIMIT",
        ):
            assert marker in prompt
        await engine.disable_roleplay(42)
        await engine.respond(42, "Привет")
        call = model.chat.await_args
        assert (
            call is not None and "DELTA_PERSONALITY" not in call.kwargs["system_prompt"]
        )
        await engine.set_delta_species(
            42,
            "Дракон",
            expected_appearance=state.delta_appearance,
            expected_species=None,
        )
        assert state.delta_appearance_profile == profile.encode()
        await engine.set_delta_appearance_from_text(
            42, "Исправленная внешность", expected_appearance=state.delta_appearance
        )
        assert state.delta_appearance_profile == profile.encode()
        model.chat.return_value = verified("Белая грива", "dragon", "probable")
        await engine.set_delta_appearance_from_image(
            42, ImageInput(b"image", "image/png"), update_existing=True
        )
        assert state.delta_appearance_profile == profile.encode()
        await engine.set_delta_appearance_from_image(
            42, ImageInput(b"image", "image/png")
        )
        assert not state.delta_appearance_profile
        state.delta_appearance_profile = profile.encode()
        await engine.set_delta_appearance_from_text(42, "Новый персонаж")
        assert not state.delta_appearance_profile
        state.delta_appearance = ""
        state.delta_appearance_profile = profile.encode()
        assert not appearance_profile_context(state)

    asyncio.run(scenario())


def test_user_image_and_edit_preserve_delta_and_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        async with states.use(42) as state:
            state.delta_appearance = "DELTA_APPEARANCE"
            state.delta_appearance_profile = AppearanceProfile(
                "DELTA_PERSONALITY"
            ).encode()
            state.delta_appearance_thumbnail = "DELTA_THUMB"
            state.roleplay_character = "OLD_USER"
            state.roleplay_preferences = "USER_PREF"
        async with states.use_conversation(42, -100) as state:
            state.roleplay_character = "GROUP_USER"
        model.chat.return_value = verified("Белая грива", "dragon", "probable")
        server = MiniAppServer(TOKEN, states, response_engine=engine)
        headers = {
            "X-Telegram-Init-Data": _signed_init_data(),
            "X-Appearance-Options": quote(
                json.dumps({"species": "Дракон", "notes": "Один персонаж"})
            ),
        }
        path = "/api/profile/character/appearance"
        async with TestClient(TestServer(server.application())) as client:
            assert (await client.post(path, data=_image())).status == 401
            result = await client.post(path, headers=headers, data=_image())
            assert result.status == 200
            saved = await result.json()
            assert (
                saved["roleplay_character"]
                == "Вид со слов пользователя: Дракон. Белая грива"
            )
            thumb = saved["roleplay_character_thumbnail"]
            assert thumb.startswith("data:image/jpeg;base64,")
            assert not saved["roleplay_active"]
            assert saved["delta_appearance"] == "DELTA_APPEARANCE"
            assert saved["delta_appearance_thumbnail"] == "DELTA_THUMB"
            assert (
                saved["delta_appearance_profile"]["personality"] == "DELTA_PERSONALITY"
            )
            assert saved["roleplay_preferences"] == "USER_PREF"
            call = model.chat.await_args
            assert (
                call is not None
                and "сохранённого RP-персонажа пользователя"
                in call.kwargs["system_prompt"]
            )
            assert "Один персонаж" in call.kwargs["user_message"]
            bad = await client.patch(
                "/api/profile",
                headers=headers,
                json={"roleplay_character": "stale", "expected_character": "OLD_USER"},
            )
            assert bad.status == 409
            assert (
                await client.patch(
                    "/api/profile",
                    headers=headers,
                    json={"roleplay_character": "without-version"},
                )
            ).status == 409
            edited = "Отредактированное описание " + "x" * 1000
            good = await client.patch(
                "/api/profile",
                headers=headers,
                json={
                    "roleplay_character": edited,
                    "expected_character": saved["roleplay_character"],
                },
            )
            assert good.status == 200
            assert (await good.json())["roleplay_character_thumbnail"] == thumb
            other = await client.get(
                "/api/profile",
                headers={"X-Telegram-Init-Data": _signed_init_data(user_id=7)},
            )
            assert not (await other.json())["roleplay_character_thumbnail"]
        restarted = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with restarted.use(42) as state:
            assert (
                state.roleplay_character == edited
                and state.roleplay_character_thumbnail == thumb
            )
            assert state.delta_appearance == "DELTA_APPEARANCE"
        async with restarted.use_conversation(42, -100) as state:
            assert state.roleplay_character == "GROUP_USER"

    asyncio.run(scenario())


def test_user_image_failure_keeps_profile_and_minor_marker_survives_edit() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.roleplay_character = "OLD_USER"
        state.roleplay_character_thumbnail = "OLD_THUMB"
        state.roleplay_active = True
        model.chat.return_value = ""
        with pytest.raises(AppearanceAnalysisError):
            await engine.set_roleplay_character_from_image(
                42, ImageInput(b"image", "image/png")
            )
        assert (
            state.roleplay_character == "OLD_USER"
            and state.roleplay_character_thumbnail == "OLD_THUMB"
        )
        assert not engine._delivering_users
        value = json.loads(verified("Нейтральное описание"))
        value["minor_reference"] = True
        model.chat.return_value = json.dumps(value)
        await engine.set_roleplay_character_from_image(
            42, ImageInput(b"image", "image/png"), thumbnail="NEW_THUMB"
        )
        assert state.roleplay_character_restricted and not state.roleplay_active
        server = MiniAppServer(TOKEN, engine._user_states, response_engine=engine)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            result = await client.patch(
                "/api/profile",
                headers=headers,
                json={
                    "roleplay_character": "Исправленный цвет",
                    "expected_character": state.roleplay_character,
                },
            )
            assert result.status == 200
            assert (await result.json())["roleplay_character_restricted"]
            clear = await client.delete("/api/profile/roleplay", headers=headers)
            saved = await clear.json()
            assert (
                not saved["roleplay_character_thumbnail"]
                and not saved["roleplay_character_restricted"]
            )

    asyncio.run(scenario())


def test_profile_migration_keeps_old_private_and_group_rows(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = UserStateStore(persistence=UserStateRepository(tmp_path))
        for key in (42, (-100, 42)):
            async with store.use(key) as state:
                state.delta_appearance = "old appearance"
                state.roleplay_character = "old character"
                state.delta_appearance_thumbnail = "old thumbnail"
        with sqlite3.connect(tmp_path / "user_states.db") as db:
            for table in ("user_states", "conversation_states"):
                for column in (
                    "delta_appearance_profile",
                    "roleplay_character_thumbnail",
                    "roleplay_character_restricted",
                ):
                    db.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        reopened = UserStateStore(persistence=UserStateRepository(tmp_path))
        for key in (42, (-100, 42)):
            async with reopened.use(key) as state:
                assert (
                    state.delta_appearance == "old appearance"
                    and state.roleplay_character == "old character"
                )
                assert state.delta_appearance_thumbnail == "old thumbnail"
                assert (
                    not state.delta_appearance_profile
                    and not state.roleplay_character_thumbnail
                )
                assert not state.roleplay_character_restricted

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["{", "[]", '{"unexpected":"field"}', "x" * 16385])
def test_invalid_stored_profile_is_not_context(value: str) -> None:
    assert AppearanceProfile.decode(value) is None
