"""Инструменты в Mini App: личная доставка, безопасные URL и проверка initData."""

import asyncio
import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
from test_e621 import _post
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.miniapp.tools import MiniAppTools, ToolsBusyError
from protogen_delta.services.e621 import E621Error, E621QueryError
from protogen_delta.services.media_download import MediaDownloadError


def test_id_selection_does_not_lookup_private_numeric_ids() -> None:
    async def scenario() -> None:
        bot = AsyncMock()
        bot.get_me.return_value = SimpleNamespace(id=10, full_name="Delta")
        bot.get_chat.return_value = SimpleNamespace(
            id=-100, title="Group", type="supergroup"
        )
        tools = MiniAppTools(bot, Mock(), Mock())
        assert (await tools.lookup_id(42, "self"))["id"] == 42
        assert (await tools.lookup_id(42, "bot"))["id"] == 10
        assert (await tools.lookup_id(42, "@public_group"))["id"] == -100
        await tools.lookup_id(42, "select")
        assert bot.send_message.await_args.args[0] == 42
        assert bot.send_message.await_args.kwargs["reply_markup"].keyboard
        for invalid in ("123", "-100123", "@bad", "https://t.me/group"):
            with pytest.raises(ValueError):
                await tools.lookup_id(42, invalid)
        bot.get_chat.return_value.type = "private"
        with pytest.raises(ValueError):
            await tools.lookup_id(42, "@person")

    asyncio.run(scenario())


def test_gallery_does_not_return_adult_video_or_foreign_media_urls() -> None:
    async def scenario() -> None:
        client = Mock()
        client.search = AsyncMock(
            return_value=[
                replace(
                    _post(1),
                    rating="s",
                    file_url="https://static1.e621.net/data/art.png",
                    sample_url=None,
                ),
                replace(_post(2), rating="e"),
                replace(_post(3, "mp4"), rating="s"),
                replace(_post(4), rating="s", tags=frozenset({"cub"})),
                replace(
                    _post(5),
                    rating="s",
                    file_url="https://other.local/a.png",
                    sample_url=None,
                ),
                replace(
                    _post(6),
                    rating="s",
                    file_url=None,
                    sample_url=None,
                    preview_url=None,
                ),
            ]
        )
        tools = MiniAppTools(AsyncMock(), client, Mock())
        result = await tools.gallery(42, "dragon", 2)
        assert [item["id"] for item in cast(list[dict[str, Any]], result["items"])] == [
            1
        ]
        assert client.search.await_args.args[0].base_url == "https://e926.net"
        assert "rating:s" in client.search.await_args.args[0].tags
        assert client.search.await_args.kwargs["page"] == 2
        with pytest.raises(ToolsBusyError):
            await tools.gallery(42, "dragon", 2)
        with pytest.raises(E621QueryError):
            await tools.gallery(7, "cub", 1)

    asyncio.run(scenario())


def _gallery_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), "blue").save(output, "PNG")
    return output.getvalue()


def test_gallery_source_tracks_age_and_always_filters_explicit_results() -> None:
    async def scenario() -> None:
        states = UserStateStore()
        states.get(42).content_mode = "adult"
        states.get(43).content_mode = "soft"
        client = Mock()
        client.search = AsyncMock(
            return_value=[
                replace(
                    _post(1),
                    rating="s",
                    file_url="https://static1.e926.net/data/art.jpg",
                    sample_url=None,
                ),
                replace(_post(2), rating="e"),
            ]
        )
        tools = MiniAppTools(AsyncMock(), client, Mock(), user_states=states)
        for user_id, site in ((42, "e621"), (43, "e926")):
            result = await tools.gallery(user_id, "dragon rating:e", 1)
            assert result["site"] == site
            query = client.search.await_args.args[0]
            assert query.base_url == f"https://{site}.net"
            assert "rating:s" in query.tags and "rating:e" not in query.tags
            items = cast(list[dict[str, Any]], result["items"])
            assert [item["id"] for item in items] == [1]
            assert items[0]["page_url"] == f"https://{site}.net/posts/1"

    asyncio.run(scenario())


def test_gallery_image_is_owned_cached_bounded_and_expires(monkeypatch: Any) -> None:
    import protogen_delta.miniapp.tools as tools_module

    now = [100.0]
    monkeypatch.setattr(tools_module, "monotonic", lambda: now[0])

    async def scenario() -> None:
        client = Mock()
        client.search = AsyncMock(
            return_value=[
                replace(
                    _post(1),
                    rating="s",
                    file_url="https://static1.e926.net/data/full.png",
                    sample_url=None,
                    preview_url="https://static1.e926.net/data/preview.png",
                )
            ]
        )
        client.download = AsyncMock(return_value=_gallery_png())
        tools = MiniAppTools(AsyncMock(), client, Mock())
        await tools.gallery(42, "dragon", 1)
        with pytest.raises(ValueError, match="Обнови поиск"):
            await tools.gallery_image(43, 1, full=False)
        with pytest.raises(ValueError):
            await tools.gallery_image(42, 2, full=True)
        client.download.assert_not_awaited()
        data, mime = await tools.gallery_image(42, 1, full=False)
        assert data == _gallery_png() and mime == "image/png"
        client.download.assert_awaited_once_with(
            "https://static1.e926.net/data/preview.png",
            max_bytes=2 * 1024 * 1024,
            allow_redirects=False,
        )
        assert await tools.gallery_image(42, 1, full=False) == (data, mime)
        assert client.download.await_count == 1
        await tools.gallery_image(42, 1, full=True)
        assert client.download.await_args.kwargs == {
            "max_bytes": 20 * 1024 * 1024,
            "allow_redirects": False,
        }
        assert client.download.await_args.args[0].endswith("full.png")
        now[0] += 601
        with pytest.raises(ValueError):
            await tools.gallery_image(42, 1, full=False)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/a.png",
        "https://static1.e926.net:8443/a.png",
        "https://static1.e926.net:invalid/a.png",
        "https://user@static1.e926.net/a.png",
        "https://static1.e926.net.evil.org/a.png",
    ],
)
def test_gallery_rejects_untrusted_preview_or_original(url: str) -> None:
    async def scenario() -> None:
        client = Mock()
        client.search = AsyncMock(
            return_value=[
                replace(
                    _post(1),
                    rating="s",
                    file_url=url,
                    sample_url="https://static1.e926.net/data/sample.png",
                ),
                replace(
                    _post(2),
                    rating="s",
                    file_url="https://static1.e926.net/data/full.png",
                    sample_url=None,
                    preview_url=url,
                ),
            ]
        )
        client.download = AsyncMock()
        tools = MiniAppTools(AsyncMock(), client, Mock())
        assert (await tools.gallery(42, "dragon", 1))["items"] == []
        assert not tools._gallery_images
        client.download.assert_not_awaited()

    asyncio.run(scenario())


def test_gallery_rejects_non_image_bytes() -> None:
    async def scenario() -> None:
        client = Mock()
        client.search = AsyncMock(
            return_value=[
                replace(
                    _post(1),
                    rating="s",
                    file_url="https://static1.e926.net/data/art.jpg",
                    sample_url=None,
                )
            ]
        )
        client.download = AsyncMock(return_value=b"<html>not a picture</html>")
        tools = MiniAppTools(AsyncMock(), client, Mock())
        await tools.gallery(42, "dragon", 1)
        with pytest.raises(ValueError, match="изображение"):
            await tools.gallery_image(42, 1, full=False)
        assert not tools._gallery_cache

    asyncio.run(scenario())


def test_gallery_image_api_authentication_and_binary_response() -> None:
    async def scenario() -> None:
        tools = Mock()
        tools.gallery_image = AsyncMock(return_value=(_gallery_png(), "image/png"))
        server = MiniAppServer(TOKEN, UserStateStore(), tools=tools)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            assert (
                await client.post("/api/tools/gallery-image", json={"post_id": 1})
            ).status == 401
            tools.gallery_image.assert_not_awaited()
            for payload in (
                {"post_id": True},
                {"post_id": "1"},
                {"post_id": 0},
                {"post_id": 1, "full": "false"},
                {},
            ):
                assert (
                    await client.post(
                        "/api/tools/gallery-image", headers=headers, json=payload
                    )
                ).status == 400
            response = await client.post(
                "/api/tools/gallery-image",
                headers=headers,
                json={
                    "post_id": 1,
                    "full": True,
                    "user_id": 7,
                    "url": "https://127.0.0.1/",
                },
            )
            assert response.status == 200 and await response.read() == _gallery_png()
            assert response.content_type == "image/png"
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            tools.gallery_image.assert_awaited_once_with(42, 1, full=True)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "suffix,method", [(".mp4", "send_video"), (".webm", "send_document")]
)
def test_download_uses_own_chat_and_cleans_up(
    tmp_path: Path, suffix: str, method: str
) -> None:
    async def scenario() -> None:
        path = tmp_path / ("download" + suffix)
        path.write_bytes(b"media")
        cleaned = []

        @asynccontextmanager
        async def download(url: str) -> AsyncIterator[SimpleNamespace]:
            assert url == "https://youtu.be/abc"
            try:
                yield SimpleNamespace(path=path, title="Video")
            finally:
                cleaned.append(True)

        downloader = Mock()
        downloader.download = download
        bot = AsyncMock()
        tools = MiniAppTools(bot, Mock(), downloader)
        await tools.download(42, "https://youtu.be/abc")
        assert getattr(bot, method).await_args.args[0] == 42
        assert cleaned == [True]
        assert not tools._pending
        with pytest.raises(ToolsBusyError):
            await tools.download(42, "https://youtu.be/abc")
        tools._pending.add(7)
        with pytest.raises(ToolsBusyError):
            await tools.download(7, "https://youtu.be/abc")
        bot.send_video.side_effect = bot.send_document.side_effect = TelegramBadRequest(
            method=Mock(), message="bad"
        )
        with pytest.raises(ValueError, match="не принял"):
            await tools.download(8, "https://youtu.be/abc")
        assert not tools._pending - {7}
        await tools._slots.acquire()
        await tools._slots.acquire()
        with pytest.raises(ToolsBusyError, match="занят"):
            await tools.download(9, "https://youtu.be/abc")
        assert 9 not in tools._pending
        tools._slots.release()
        tools._slots.release()

    asyncio.run(scenario())


def test_tools_api_checks_signature_inputs_and_handles_failures() -> None:
    async def scenario() -> None:
        tools = Mock()
        tools.lookup_id = AsyncMock(return_value={"id": 42})
        tools.gallery = AsyncMock(return_value={"items": []})
        tools.download = AsyncMock(return_value={"message": "done"})
        server = MiniAppServer(TOKEN, UserStateStore(), tools=tools)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            for tool in ("id", "gallery", "download"):
                assert (await client.post(f"/api/tools/{tool}", json={})).status == 401
            tools.lookup_id.assert_not_awaited()
            assert (
                await client.post("/api/tools/id", headers=headers, json={})
            ).status == 200
            tools.lookup_id.assert_awaited_with(42, "self")
            assert (
                await client.post(
                    "/api/tools/gallery",
                    headers=headers,
                    json={"query": "dragon", "page": 3},
                )
            ).status == 200
            tools.gallery.assert_awaited_with(42, "dragon", 3)
            assert (
                await client.post(
                    "/api/tools/download",
                    headers=headers,
                    json={"url": " https://youtu.be/a "},
                )
            ).status == 200
            tools.download.assert_awaited_with(42, "https://youtu.be/a")
            for tool, payload in [
                ("id", {"target": 2}),
                ("id", []),
                ("gallery", {}),
                ("gallery", {"query": "a", "page": True}),
                ("gallery", {"query": "a", "page": 0}),
                ("download", {}),
            ]:
                assert (
                    await client.post(
                        f"/api/tools/{tool}", headers=headers, json=payload
                    )
                ).status == 400
            assert (
                await client.post("/api/tools/id", headers=headers, data=b"bad")
            ).status == 400
            assert (
                await client.post("/api/tools/unknown", headers=headers, json={})
            ).status == 404
            for error, status in [
                (ToolsBusyError("busy"), 429),
                (ValueError("invalid"), 400),
                (E621QueryError("query"), 400),
                (E621Error("network"), 502),
                (TelegramBadRequest(method=Mock(), message="unknown"), 502),
                (MediaDownloadError("unsupported"), 400),
            ]:
                tools.lookup_id.side_effect = error
                response = await client.post("/api/tools/id", headers=headers, json={})
                assert response.status == status
        disabled = MiniAppServer(TOKEN, UserStateStore())
        async with TestClient(TestServer(disabled.application())) as client:
            assert (
                await client.post("/api/tools/id", headers=headers, json={})
            ).status == 503

    asyncio.run(scenario())
