"""Загрузка референса, изоляция профиля и восстановление базовой внешности."""

import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.appearance_image import (
    MAX_APPEARANCE_BYTES,
    prepare_appearance_image,
)
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.deepseek import DeepSeekConnectionError, ImageInput
from protogen_delta.services.response_engine import (
    AppearanceAnalysisError,
    ResponseBusyError,
)


def _image(format: str = "PNG", *, size: tuple[int, int] = (100, 200)) -> bytes:
    image = Image.new("RGBA", size, (20, 140, 230, 100))
    output = BytesIO()
    image.convert("RGB").save(output, format)
    return output.getvalue()


@pytest.mark.parametrize("format", ["JPEG", "PNG", "WEBP"])
def test_appearance_image_preserves_whole_frame_and_removes_metadata(
    format: str,
) -> None:
    result = prepare_appearance_image(_image(format, size=(2000, 1000)))
    assert result.mime_type == "image/jpeg"
    with Image.open(BytesIO(result.data)) as image:
        assert image.size == (1536, 768)
        assert not image.getexif()


def test_appearance_image_rotates_exif_and_flattens_transparency() -> None:
    image = Image.new("RGBA", (30, 60), (20, 140, 230, 0))
    output = BytesIO()
    image.save(output, "PNG")
    result = prepare_appearance_image(output.getvalue())
    with Image.open(BytesIO(result.data)) as flattened:
        assert flattened.getpixel((10, 10)) == (255, 255, 255)
    exif = Image.Exif()
    exif[274] = 6
    output = BytesIO()
    image.convert("RGB").save(output, "JPEG", exif=exif)
    result = prepare_appearance_image(output.getvalue())
    with Image.open(BytesIO(result.data)) as rotated:
        assert rotated.size == (60, 30)
        assert not rotated.getexif()


@pytest.mark.parametrize(
    "data",
    [b"", b"bad", _image("GIF"), b"x" * (MAX_APPEARANCE_BYTES + 1)],
    ids=["empty", "corrupt", "gif", "oversize"],
)
def test_appearance_image_rejects_invalid_and_oversized_files(data: bytes) -> None:
    with pytest.raises(ValueError):
        prepare_appearance_image(data)


def test_appearance_image_rejects_excessive_pixels_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SimpleNamespace(format="PNG", width=8000, height=6000)
    opened = MagicMock()
    opened.__enter__.return_value = source
    monkeypatch.setattr(
        "protogen_delta.services.appearance_image.Image.open", lambda _: opened
    )
    with pytest.raises(ValueError, match="мегапикселей"):
        prepare_appearance_image(b"header")


def test_upload_and_reset_persist_only_own_appearance(tmp_path: Path) -> None:
    engine, _, model, *_ = _create_engine()
    repository = UserStateRepository(tmp_path)
    states = UserStateStore(persistence=repository)
    engine._user_states = states
    model.chat.return_value = "Бирюзовый дракон с белой гривой."
    server = MiniAppServer(TOKEN, states, response_engine=engine)

    async def scenario() -> None:
        async with states.use(42) as state:
            state.delta_appearance = "Прежний облик"
            state.roleplay_character = "Мой персонаж"
            state.roleplay_preferences = "Юмор"
            state.roleplay_boundaries = "Без унижения"
            state.roleplay_configuration = "female"
        async with states.use(7) as other:
            other.delta_appearance = "Другой пользователь"
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            response = await client.post(
                "/api/profile/appearance", headers=headers, data=_image()
            )
            assert response.status == 200
            profile = await response.json()
            assert profile["delta_appearance"] == model.chat.return_value
            assert profile["appearance_upload_enabled"]
            assert not profile["roleplay_active"]
            assert profile["roleplay_configuration"] == "female"
            assert profile["roleplay_character"] == "Мой персонаж"
            assert states.get(7).delta_appearance == "Другой пользователь"
            model.chat.assert_awaited_once()
            call = model.chat.await_args
            assert call is not None
            assert call.kwargs["tool_names"] == frozenset()
            assert len(call.kwargs["images"]) == 1
            assert not states.get(42).history
            restored = UserStateStore(persistence=repository)
            async with restored.use(42) as state:
                assert state.delta_appearance == model.chat.return_value
            reset = await client.delete("/api/profile/appearance", headers=headers)
            assert reset.status == 200
            profile = await reset.json()
            assert not profile["delta_appearance"]
            assert profile["roleplay_character"] == "Мой персонаж"
            assert profile["roleplay_preferences"] == "Юмор"
            assert profile["roleplay_boundaries"] == "Без унижения"
            restored = UserStateStore(persistence=repository)
            async with restored.use(42) as state:
                assert not state.delta_appearance

    asyncio.run(scenario())


@pytest.mark.parametrize("reply", ["", DeepSeekConnectionError("unavailable")])
def test_analysis_failure_keeps_previous_appearance_and_releases_lock(
    reply: object,
) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.delta_appearance = "Прежний облик"
    if isinstance(reply, Exception):
        model.chat.side_effect = reply
    else:
        model.chat.return_value = reply
    with pytest.raises(AppearanceAnalysisError):
        asyncio.run(
            engine.set_delta_appearance_from_image(
                42, ImageInput(b"image", "image/png")
            )
        )
    assert state.delta_appearance == "Прежний облик"
    assert not engine._delivering_users


def test_appearance_change_does_not_overlap_chat_delivery() -> None:
    engine, *_ = _create_engine()
    engine._delivering_users.add(42)
    with pytest.raises(ResponseBusyError):
        asyncio.run(
            engine.set_delta_appearance_from_image(
                42, ImageInput(b"image", "image/png")
            )
        )


def test_appearance_api_authenticates_before_accepting_upload_or_reset() -> None:
    engine, *_ = _create_engine()
    server = MiniAppServer(TOKEN, engine._user_states, response_engine=engine)

    async def scenario() -> None:
        async with TestClient(TestServer(server.application())) as client:
            for method in ("POST", "DELETE"):
                response = await client.request(
                    method, "/api/profile/appearance", data=b"bad"
                )
                assert response.status == 401

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "data",
    [b"", b"<svg/>", b"x" * (MAX_APPEARANCE_BYTES + 1)],
    ids=["empty", "svg", "oversize"],
)
def test_appearance_api_rejects_bad_files_without_model(data: bytes) -> None:
    engine, _, model, *_ = _create_engine()
    server = MiniAppServer(TOKEN, engine._user_states, response_engine=engine)

    async def scenario() -> None:
        async with TestClient(TestServer(server.application())) as client:
            response = await client.post(
                "/api/profile/appearance",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
                data=data,
            )
            assert response.status == (413 if len(data) > MAX_APPEARANCE_BYTES else 400)
            assert not server._appearance_pending

    asyncio.run(scenario())
    model.chat.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [AppearanceAnalysisError(), ResponseBusyError(), TimeoutError()]
)
def test_appearance_api_maps_failures_and_cleans_pending(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, *_ = _create_engine()
    monkeypatch.setattr(
        engine, "set_delta_appearance_from_image", AsyncMock(side_effect=error)
    )
    server = MiniAppServer(TOKEN, engine._user_states, response_engine=engine)

    async def scenario() -> None:
        async with TestClient(TestServer(server.application())) as client:
            response = await client.post(
                "/api/profile/appearance",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
                data=_image(),
            )
            assert response.status == (
                409
                if isinstance(error, ResponseBusyError)
                else 504 if isinstance(error, TimeoutError) else 502
            )
            assert not server._appearance_pending

    asyncio.run(scenario())


def test_missing_analyzer_still_allows_reset() -> None:
    states = UserStateStore()
    states.get(42).delta_appearance = "Прежний облик"
    server = MiniAppServer(TOKEN, states)

    async def scenario() -> None:
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            upload = await client.post(
                "/api/profile/appearance", headers=headers, data=_image()
            )
            assert upload.status == 503
            reset = await client.delete("/api/profile/appearance", headers=headers)
            assert reset.status == 200
            assert not (await reset.json())["delta_appearance"]

    asyncio.run(scenario())


def test_concurrent_uploads_and_rate_limit_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, *_ = _create_engine()

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def analyze(user_id: int, image: ImageInput) -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
            await release.wait()
            return "Описание"

        monkeypatch.setattr(
            engine, "set_delta_appearance_from_image", AsyncMock(side_effect=analyze)
        )
        server = MiniAppServer(
            TOKEN,
            engine._user_states,
            response_engine=engine,
            native_work=BlockingWorkPool(2),
        )
        async with TestClient(TestServer(server.application())) as client:

            def headers(user_id: int) -> dict[str, str]:
                return {"X-Telegram-Init-Data": _signed_init_data(user_id=user_id)}

            first = asyncio.create_task(
                client.post(
                    "/api/profile/appearance", headers=headers(42), data=_image()
                )
            )
            second = asyncio.create_task(
                client.post(
                    "/api/profile/appearance", headers=headers(43), data=_image()
                )
            )
            await asyncio.wait_for(entered.wait(), 2)
            duplicate = await client.post(
                "/api/profile/appearance", headers=headers(42), data=_image()
            )
            assert duplicate.status == 409
            reset = await client.delete("/api/profile/appearance", headers=headers(42))
            assert reset.status == 409
            busy = await client.post(
                "/api/profile/appearance", headers=headers(44), data=_image()
            )
            assert busy.status == 429
            release.set()
            responses = await asyncio.gather(first, second)
            assert all(response.status == 200 for response in responses)
            retry = await client.post(
                "/api/profile/appearance", headers=headers(42), data=_image()
            )
            assert retry.status == 429
            assert calls == 2
            assert not server._appearance_pending

    asyncio.run(scenario())
