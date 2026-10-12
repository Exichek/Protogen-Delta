"""Выход из сцены и ручные исправления переживают рестарт и не зовут vision."""

import asyncio
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.deepseek import ImageInput


@pytest.mark.parametrize("chat_id", [None, -100])
@pytest.mark.parametrize("stop", ["command", "text", "mixed"])
def test_stop_closes_scene_for_text_images_and_restart(
    tmp_path: Path, chat_id: int | None, stop: str
) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with engine._user_states.use_conversation(42, chat_id) as state:
            state.roleplay_active = True
            state.roleplay_character = "Сохранённый персонаж"
            state.delta_appearance = "Сохранённый облик"
            for _ in range(6):
                state.history.append(ConversationTurn("*стою у маяка*", "SCENE_MARKER"))
        if stop == "command":
            assert await engine.disable_roleplay(42, chat_id=chat_id)
        else:
            await engine.respond(
                42,
                "стоп рп" if stop == "text" else "стоп рп, расскажи о погоде",
                chat_id=chat_id,
            )
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        await engine.respond(
            42,
            "Что на картинке?",
            chat_id=chat_id,
            images=(ImageInput(b"image", "image/png"),),
        )
        args = model.chat.await_args.kwargs
        assert "SCENE_MARKER" not in args["system_prompt"]
        assert all(t.assistant_message != "SCENE_MARKER" for t in args["history"])
        assert "RP выключен" in args["system_prompt"]
        async with engine._user_states.use_conversation(42, chat_id) as state:
            assert not state.roleplay_active
            assert state.roleplay_character == "Сохранённый персонаж"
            assert state.delta_appearance == "Сохранённый облик"
            assert any(t.context_closed for t in state.history)
        # Повторный /rp off закрывает также ошибочно продолженный ответ.
        assert not await engine.disable_roleplay(42, chat_id=chat_id)
        await engine.respond(42, "Привет", chat_id=chat_id)
        assert model.chat.await_args.kwargs["history"] == ()

    asyncio.run(scenario())


def test_edit_saved_appearance_preserves_image_and_profile(tmp_path: Path) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        async with states.use(42) as state:
            state.delta_appearance = "Белая шерсть на лбу"
            state.delta_appearance_thumbnail = "existing-thumbnail"
            state.roleplay_character = "Персонаж пользователя"
        async with states.use((-100, 42)) as state:
            state.delta_appearance = "Групповой облик"
        server = MiniAppServer(TOKEN, states, response_engine=engine)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        payload = {
            "appearance": "Белая чёлка закрывает один глаз",
            "expected_appearance": "Белая шерсть на лбу",
        }
        async with TestClient(TestServer(server.application())) as client:
            assert (
                await client.patch("/api/profile/appearance", json=payload)
            ).status == 401
            result = await client.patch(
                "/api/profile/appearance", headers=headers, json=payload
            )
            assert result.status == 200
            saved = await result.json()
            assert saved["delta_appearance"] == payload["appearance"]
            assert saved["delta_appearance_thumbnail"] == "existing-thumbnail"
            assert saved["roleplay_character"] == "Персонаж пользователя"
            # Устаревший редактор не может затереть более новый облик.
            result = await client.patch(
                "/api/profile/appearance", headers=headers, json=payload
            )
            assert result.status == 400
            bad: object
            for bad in (
                {},
                [],
                {**payload, "expected_appearance": ""},
                {**payload, "appearance": "x" * 2001},
            ):
                assert (
                    await client.patch(
                        "/api/profile/appearance", headers=headers, json=bad
                    )
                ).status == 400
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with restored.use(42) as state:
            assert state.delta_appearance == payload["appearance"]
            assert state.delta_appearance_thumbnail == "existing-thumbnail"
        async with restored.use((-100, 42)) as state:
            assert state.delta_appearance == "Групповой облик"
        model.chat.assert_not_awaited()
        model.analyze_visual_features.assert_not_awaited()

    asyncio.run(scenario())
