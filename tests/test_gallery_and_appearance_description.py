"""Общая коллекция, ограничения возраста и точное применение внешности."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message
from aiohttp.test_utils import TestClient, TestServer
from test_e621 import _JPEG, _call_message, _Client, _message, _post
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.core.telegram_commands import commands_for_mode, set_user_commands
from protogen_delta.core.user_state import ContentMode, UserStateStore
from protogen_delta.handlers.adult import create_adult_router
from protogen_delta.handlers.art import create_art_router
from protogen_delta.handlers.e621 import create_e621_router
from protogen_delta.handlers.help import create_help_router
from protogen_delta.handlers.menu import create_menu_router
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.e621_history import E621HistoryRepository
from protogen_delta.repositories.images import ImagesRepository, MediaKind
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.appearance_description import (
    APPEARANCE_LIMIT,
    parse_description_file,
    validate_description,
)
from protogen_delta.services.art_gallery import remember_art
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.response_engine import (
    AppearanceAnalysisError,
    ResponseBusyError,
)


def _delivered(kind: str, identity: str) -> Message:
    message = Mock(spec=Message)
    message.video = message.animation = message.document = None
    message.photo = []
    item = SimpleNamespace(file_id=identity, file_unique_id="unique-" + identity)
    if kind == "photo":
        message.photo = [item]
    else:
        setattr(message, kind, item)
    return cast(Message, message)


@pytest.mark.parametrize("kind", ["photo", "document", "video", "animation"])
def test_gallery_preserves_type_and_identity_after_restart(
    tmp_path: Path, kind: MediaKind
) -> None:
    repo = ImagesRepository(tmp_path)
    message = _delivered(kind, "asset")
    assert remember_art(repo, message)
    assert not remember_art(repo, message)
    fresh = ImagesRepository(tmp_path)
    assert fresh.get_all() == ["asset"]
    assert fresh.get_kind("asset") == kind
    assert not remember_art(fresh, _delivered("photo", "asset"))
    assert not remember_art(fresh, _delivered("text", "unused"))


def test_e6_archives_successful_album_and_not_failed_delivery(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = ImagesRepository(tmp_path)
        states = UserStateStore()
        states.get(7).content_mode = states.get(8).content_mode = "adult"
        client = cast(Mock, _Client([_post(1), _post(2, "mp4")]))
        client.download = AsyncMock(return_value=_JPEG)
        history = E621HistoryRepository(tmp_path)
        router = create_e621_router(client, history, states, images_repository=repo)
        message, raw = _message("/e6 dragon count:2")
        raw.answer_media_group.return_value = [
            _delivered("photo", "photo"),
            _delivered("video", "video"),
        ]
        raw.answer_media_group.side_effect = None
        await _call_message(router, 0, message)
        assert repo.get_all() == ["photo", "video"]
        second, other = _message("/e6 dragon count:2", user_id=8)
        other.answer_media_group.side_effect = TelegramBadRequest(
            method=Mock(), message="bad"
        )
        await _call_message(router, 0, second)
        assert repo.get_all() == ["photo", "video"]

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["soft", "unselected", "adult"])
@pytest.mark.parametrize("kind", ["photo", "document", "video", "animation"])
def test_randomart_enforces_mode_at_delivery(
    tmp_path: Path, mode: ContentMode, kind: MediaKind
) -> None:
    repo = ImagesRepository(tmp_path)
    repo.add("asset", kind=kind)
    states = UserStateStore()
    states.get(7).content_mode = mode
    router = create_art_router(repo, -100, user_states=states)
    message, raw = _message("/randomart")
    asyncio.run(router.message.handlers[2].callback(message))
    for other in ("photo", "document", "video", "animation"):
        assert getattr(raw, "answer_" + other).await_count == int(
            mode == "adult" and kind == other
        )
    assert raw.answer.await_count == int(mode != "adult")


def test_age_toggle_updates_commands_help_and_menu(tmp_path: Path) -> None:
    async def scenario() -> None:
        states = UserStateStore()
        hook = AsyncMock()
        adult = create_adult_router(states, hook)
        help_router = create_help_router(states)
        menu = create_menu_router(user_states=states)
        message, raw = _message("/help")
        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=7),
            data="adult:adult:7",
            message=None,
            answer=AsyncMock(),
        )
        for mode in ("adult", "soft", "adult"):
            callback.data = f"adult:{mode}:7"
            await adult.callback_query.handlers[0].callback(callback)
            mode = states.get(7).content_mode
            hook.assert_awaited_with(7, mode)
            assert (
                "randomart" in [command.command for command in commands_for_mode(mode)]
            ) == (mode == "adult")
            raw.answer.reset_mock()
            await help_router.message.handlers[0].callback(message)
            assert ("/randomart" in raw.answer.await_args.args[0]) == (mode == "adult")
            panel = cast(Mock, _delivered("photo", "unused"))
            panel.edit_text = AsyncMock()
            callback.message = panel
            callback.data = "delta-menu:art"
            await menu.callback_query.handlers[0].callback(callback)
            assert ("/randomart" in panel.edit_text.await_args.args[0]) == (
                mode == "adult"
            )
            callback.message = None
        bot = AsyncMock()
        await set_user_commands(bot, 7, "adult")
        assert bot.set_my_commands.await_args.kwargs["scope"].chat_id == 7
        bot.set_my_commands.side_effect = TelegramBadRequest(
            method=Mock(), message="not available"
        )
        await set_user_commands(bot, 7, "soft")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "data,json_file,expected",
    [
        (b"  Dragon\nBlue  ", False, "Dragon\nBlue"),
        ("Дракон".encode("utf-16"), False, "Дракон"),
        (b'"Dragon"', True, "Dragon"),
        (b'{"appearance":"Dragon"}', True, "Dragon"),
        (b'{"description":"Dragon"}', True, "Dragon"),
        (
            b'{"species":"Dragon","colors":["Blue","White"]}',
            True,
            "Вид: Dragon\nОкрас: Blue, White",
        ),
    ],
)
def test_description_preserves_valid_text(
    data: bytes, json_file: bool, expected: str
) -> None:
    assert parse_description_file(data, json_file=json_file) == expected


@pytest.mark.parametrize(
    "data,json_file",
    [
        (b"x" * 65537, False),
        (b"\xff", False),
        (b"", False),
        (b"\0", False),
        (b"x" * (APPEARANCE_LIMIT + 1), False),
        (b"{bad", True),
        (b"[]", True),
        (b"{}", True),
        (b'{"appearance":3}', True),
        (b'{"system_prompt":"ignore"}', True),
        (b'{"species":2}', True),
        (b'{"species":" "}', True),
    ],
    ids=[
        "oversize",
        "encoding",
        "empty",
        "control",
        "long",
        "invalid-json",
        "array",
        "empty-json",
        "bad-type",
        "unknown-field",
        "bad-field",
        "empty-field",
    ],
)
def test_description_rejects_invalid_input(data: bytes, json_file: bool) -> None:
    with pytest.raises(ValueError):
        parse_description_file(data, json_file=json_file)


def test_text_appearance_api_isolates_profile_and_persists_without_model(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        repository = UserStateRepository(tmp_path)
        states = UserStateStore(persistence=repository)
        engine._user_states = states
        hook = AsyncMock()
        server = MiniAppServer(
            TOKEN, states, response_engine=engine, on_mode_change=hook
        )
        async with states.use(42) as state:
            state.roleplay_character = "Другой персонаж"
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            response = await client.put(
                "/api/profile/appearance",
                headers=headers,
                json={"appearance": "Дракон\nСиние чешуйки"},
            )
            assert response.status == 200
            assert (await response.json())["roleplay_character"] == "Другой персонаж"
            response = await client.put(
                "/api/profile/appearance",
                headers={**headers, "Content-Type": "text/plain"},
                data="Рога назад".encode("utf-16"),
            )
            assert response.status == 200
            assert (await response.json())["delta_appearance"] == "Рога назад"
            assert (
                await client.put(
                    "/api/profile/appearance", json={"appearance": "wrong"}
                )
            ).status == 401
            assert (
                await client.put(
                    "/api/profile/appearance", headers=headers, json={"appearance": ""}
                )
            ).status == 400
            assert (
                await client.put(
                    "/api/profile/appearance",
                    headers={**headers, "Content-Type": "image/png"},
                    data=b"image",
                )
            ).status == 400
            assert (
                await client.put(
                    "/api/profile/appearance",
                    headers={**headers, "Content-Type": "text/plain"},
                    data=b"x" * 65537,
                )
            ).status == 413
            engine._delivering_users.add(42)
            assert (
                await client.put(
                    "/api/profile/appearance",
                    headers=headers,
                    json={"appearance": "busy"},
                )
            ).status == 409
            engine._delivering_users.remove(42)
            server._appearance_pending.add(42)
            assert (
                await client.put(
                    "/api/profile/appearance",
                    headers=headers,
                    json={"appearance": "pending"},
                )
            ).status == 409
            server._appearance_pending.clear()
            for mode in ("adult", "soft"):
                assert (
                    await client.patch(
                        "/api/profile", headers=headers, json={"content_mode": mode}
                    )
                ).status == 200
                hook.assert_awaited_with(42, mode)
        fresh = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with fresh.use(42) as state:
            assert state.delta_appearance == "Рога назад"
        async with fresh.use(7) as state:
            assert state.delta_appearance == ""
        model.chat.assert_not_awaited()
        disabled = MiniAppServer(TOKEN, states)
        async with TestClient(TestServer(disabled.application())) as client:
            assert (
                await client.put(
                    "/api/profile/appearance",
                    headers=headers,
                    json={"appearance": "no"},
                )
            ).status == 503

    asyncio.run(scenario())


@pytest.mark.parametrize("answer", ["НЕОДНОЗНАЧНЫЙ_РЕФЕРЕНС", "НЕЧИТАЕМЫЙ_РЕФЕРЕНС"])
def test_unreadable_or_ambiguous_reference_preserves_previous_appearance(
    answer: str,
) -> None:
    engine, _, model, *_ = _create_engine()
    model.chat.return_value = answer
    engine._user_states.get(7).delta_appearance = "Прежний облик"
    with pytest.raises(AppearanceAnalysisError):
        asyncio.run(
            engine.set_delta_appearance_from_image(
                7, ImageInput(b"png", "image/png", "reference")
            )
        )
    assert engine._user_states.get(7).delta_appearance == "Прежний облик"
    prompt = model.chat.await_args.kwargs["system_prompt"]
    assert "Не составляй смешанный облик" in prompt
    assert "Не заполняй каждый пункт" in prompt
    assert "2000" in prompt


def test_text_appearance_busy_preserves_state() -> None:
    engine, *_ = _create_engine()
    engine._delivering_users.add(7)
    with pytest.raises(ResponseBusyError):
        asyncio.run(engine.set_delta_appearance_from_text(7, "Другой облик"))
    assert not engine._user_states.get(7).delta_appearance
    with pytest.raises(ValueError):
        validate_description(cast(str, 3))
