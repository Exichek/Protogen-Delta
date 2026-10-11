"""Распознавание меток ручного поля без генерации ответов модели."""

import asyncio
from dataclasses import asdict
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.core.appearance_profile import AppearanceProfile
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.user_state import UserStateRepository

TRIGGERS = {"topic_a": ["комета", "comet*"], "topic_b": ["загадка"]}
NAMES = {"topic_a": "Астрономия"}
HEADERS = {"X-Telegram-Init-Data": _signed_init_data()}


@pytest.mark.parametrize(
    ("text", "keys"),
    [
        ("", []),
        ("Неизвестная тема", []),
        ("КОМЕТА", ["topic_a"]),
        ("комета и comets", ["topic_a"]),
        ("кометарий", []),
        ("загадка", ["topic_b"]),
        ("загадка и комета", ["topic_a", "topic_b"]),
    ],
)
def test_topics_use_triggers_and_display_names(text: str, keys: list[str]) -> None:
    async def scenario() -> None:
        states = UserStateStore()
        state = states.get(42)
        state.delta_appearance = "Сохранённый облик"
        state.delta_appearance_profile = AppearanceProfile(fetishes=text).encode()
        state.roleplay_fetishes = ("chat_topic",)
        before = state.delta_appearance_profile
        server = MiniAppServer(
            TOKEN, states, fetish_triggers=TRIGGERS, fetish_names=NAMES
        )
        async with TestClient(TestServer(server.application())) as client:
            response = await client.get("/api/profile", headers=HEADERS)
            assert response.status == 200
            profile = await response.json()
            assert profile["delta_appearance_topics"] == [
                {"key": key, "name": NAMES.get(key, key)} for key in keys
            ]
            assert profile["delta_appearance_profile"]["fetishes"] == text
            assert profile["roleplay_fetishes"] == ["chat_topic"]
            assert state.delta_appearance_profile == before

    asyncio.run(scenario())


def test_topics_follow_saved_owner_profile_and_clear_without_model(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        async with states.use(42) as state:
            state.delta_appearance = "Сохранённый облик"
            state.roleplay_fetishes = ("chat_topic",)
        async with states.use_conversation(42, -100) as group:
            group.delta_appearance = "Групповой облик"
            group.delta_appearance_profile = AppearanceProfile(
                fetishes="загадка"
            ).encode()

        saved_profile = AppearanceProfile(personality="Любопытный", fetishes="комета")
        server = MiniAppServer(
            TOKEN,
            states,
            response_engine=engine,
            fetish_triggers=TRIGGERS,
            fetish_names=NAMES,
        )
        path = "/api/profile/appearance/behavior"
        async with TestClient(TestServer(server.application())) as client:
            response = await client.patch(
                path,
                headers=HEADERS,
                json={
                    "profile": asdict(saved_profile),
                    "expected_profile": asdict(AppearanceProfile()),
                    "expected_appearance": "Сохранённый облик",
                },
            )
            assert response.status == 200
            assert (await response.json())["delta_appearance_topics"] == [
                {"key": "topic_a", "name": "Астрономия"}
            ]
            other = await client.get(
                "/api/profile",
                headers={"X-Telegram-Init-Data": _signed_init_data(user_id=7)},
            )
            assert (await other.json())["delta_appearance_topics"] == []

        reopened = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = reopened
        server = MiniAppServer(
            TOKEN,
            reopened,
            response_engine=engine,
            fetish_triggers=TRIGGERS,
            fetish_names=NAMES,
        )
        async with TestClient(TestServer(server.application())) as client:
            response = await client.get("/api/profile", headers=HEADERS)
            assert (await response.json())["delta_appearance_topics"] == [
                {"key": "topic_a", "name": "Астрономия"}
            ]
            response = await client.patch(
                path,
                headers=HEADERS,
                json={
                    "profile": asdict(AppearanceProfile(personality="Любопытный")),
                    "expected_profile": asdict(saved_profile),
                    "expected_appearance": "Сохранённый облик",
                },
            )
            assert response.status == 200
            assert (await response.json())["delta_appearance_topics"] == []
        async with reopened.use(42) as state:
            assert state.roleplay_fetishes == ("chat_topic",)
        async with reopened.use_conversation(42, -100) as group:
            assert AppearanceProfile.decode(
                group.delta_appearance_profile
            ) == AppearanceProfile(fetishes="загадка")
        model.chat.assert_not_awaited()
        model.classify.assert_not_awaited()

    asyncio.run(scenario())


def test_topics_ignore_orphan_profile_when_appearance_is_missing() -> None:
    async def scenario() -> None:
        states = UserStateStore()
        states.get(42).delta_appearance_profile = AppearanceProfile(
            fetishes="комета"
        ).encode()
        server = MiniAppServer(
            TOKEN, states, fetish_triggers=TRIGGERS, fetish_names=NAMES
        )
        async with TestClient(TestServer(server.application())) as client:
            response = await client.get("/api/profile", headers=HEADERS)
            assert (await response.json())["delta_appearance_topics"] == []

    asyncio.run(scenario())
