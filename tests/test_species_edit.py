"""Ручное уточнение вида сохраняет облик и не запускает vision."""

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.core.appearance_species import AppearanceSpecies
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.user_state import UserStateRepository


@pytest.mark.parametrize("source", ["none", "vision", "user"])
def test_save_species_preserves_image_details_profile_and_restart(
    tmp_path: Path, source: str
) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        old = (
            None
            if source == "none"
            else (
                AppearanceSpecies("Псовый", "canine", "vision", "probable")
                if source == "vision"
                else AppearanceSpecies("Гибрид", None, "user", "declared")
            )
        )
        header = (
            ""
            if old is None
            else (
                "Вероятно, Псовый. "
                if source == "vision"
                else "Вид со слов пользователя: Гибрид. "
            )
        )
        original = header + "Светлые волосы и полосатые чулки."
        async with states.use(42) as state:
            state.delta_appearance = original
            state.delta_species = old.encode() if old else ""
            state.delta_appearance_thumbnail = "existing-thumbnail"
            state.roleplay_character = "Персонаж пользователя"
            state.history.append(ConversationTurn("HISTORY", "unchanged"))
            state.age_restricted = True
            state.delta_reference_restricted = True
        async with states.use_conversation(42, -100) as state:
            state.delta_appearance = "Групповой облик"
        server = MiniAppServer(TOKEN, states, response_engine=engine)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        payload = {
            "species": "Акула",
            "expected_appearance": original,
            "expected_species": json.loads(old.encode()) if old else None,
        }
        async with TestClient(TestServer(server.application())) as client:
            path = "/api/profile/appearance/species"
            assert (await client.patch(path, json=payload)).status == 401
            result = await client.patch(path, headers=headers, json=payload)
            assert result.status == 200
            saved = await result.json()
            assert (
                saved["delta_appearance"]
                == "Вид со слов пользователя: Акула. Светлые волосы и полосатые чулки."
            )
            assert saved["delta_species"]["species_id"] == "shark"
            assert saved["delta_species"]["source"] == "user"
            assert saved["delta_species"]["visual_check"] == "unconfirmed"
            assert saved["delta_appearance_thumbnail"] == "existing-thumbnail"
            assert saved["roleplay_character"] == "Персонаж пользователя"
            assert saved["age_restricted"] and saved["delta_reference_restricted"]
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 400
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with restored.use(42) as state:
            assert state.delta_appearance == saved["delta_appearance"]
            assert state.delta_appearance_thumbnail == "existing-thumbnail"
            assert state.history[-1].user_message == "HISTORY"
        async with restored.use_conversation(42, -100) as state:
            assert state.delta_appearance == "Групповой облик"
        model.chat.assert_not_awaited()
        model.analyze_visual_features.assert_not_awaited()

    asyncio.run(scenario())


def test_species_edit_rejects_invalid_payloads_and_stale_metadata() -> None:
    async def scenario() -> None:
        engine, _, _, *_ = _create_engine()
        engine._user_states.get(42).delta_appearance = "Сохранённый облик"
        server = MiniAppServer(TOKEN, engine._user_states, response_engine=engine)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        payload = {
            "species": "Акула",
            "expected_appearance": "Сохранённый облик",
            "expected_species": None,
        }
        path = "/api/profile/appearance/species"
        async with TestClient(TestServer(server.application())) as client:
            bad: object
            for bad in (
                [],
                {},
                {**payload, "user_id": 99},
                {**payload, "species": []},
                {**payload, "species": ""},
                {**payload, "species": "x" * 81},
                {**payload, "expected_appearance": ""},
                {**payload, "expected_appearance": "x" * 2001},
                {**payload, "expected_appearance": []},
                {**payload, "expected_species": []},
                {**payload, "expected_species": {}},
            ):
                assert (
                    await client.patch(path, headers=headers, json=bad)
                ).status == 400
                assert not server._appearance_pending
            assert (
                await client.patch(
                    path,
                    headers={**headers, "Content-Type": "application/json"},
                    data="{",
                )
            ).status == 400
            state = engine._user_states.get(42)
            state.delta_species = AppearanceSpecies(
                "Сергал", "sergal", "user", "declared"
            ).encode()
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 400
            assert state.delta_appearance == "Сохранённый облик"
            payload["expected_species"] = json.loads(state.delta_species)
            engine._delivering_users.add(42)
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 409
            engine._delivering_users.remove(42)
            server._appearance_pending.add(42)
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 409
            server._appearance_pending.remove(42)
            state.delta_appearance = ""
            assert (
                await client.patch(path, headers=headers, json=payload)
            ).status == 400

    asyncio.run(scenario())


def test_species_edit_without_engine_is_unavailable() -> None:
    async def scenario() -> None:
        server = MiniAppServer(TOKEN, UserStateStore())
        async with TestClient(TestServer(server.application())) as client:
            response = await client.patch(
                "/api/profile/appearance/species",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
                json={},
            )
            assert response.status == 503

    asyncio.run(scenario())
