"""Инструменты в Mini App: личная доставка, безопасные URL и проверка initData."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.miniapp.tools import MiniAppTools, ToolsBusyError
from protogen_delta.services.media_download import MediaDownloadError


def test_id_selection_does_not_lookup_private_numeric_ids() -> None:
    async def scenario() -> None:
        bot = AsyncMock()
        bot.get_me.return_value = SimpleNamespace(id=10, full_name="Delta")
        bot.get_chat.return_value = SimpleNamespace(
            id=-100, title="Group", type="supergroup"
        )
        tools = MiniAppTools(bot, Mock())
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
                yield SimpleNamespace(
                    path=path, title="Video", compressed=False, animation=False
                )
            finally:
                cleaned.append(True)

        downloader = Mock()
        downloader.download = download
        bot = AsyncMock()
        tools = MiniAppTools(bot, downloader)
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
        tools.download = AsyncMock(return_value={"message": "done"})
        server = MiniAppServer(TOKEN, UserStateStore(), tools=tools)
        headers = {"X-Telegram-Init-Data": _signed_init_data()}
        async with TestClient(TestServer(server.application())) as client:
            for tool in ("id", "download", "gallery", "gallery-image"):
                assert (await client.post(f"/api/tools/{tool}", json={})).status == 401
            tools.lookup_id.assert_not_awaited()
            assert (
                await client.post("/api/tools/id", headers=headers, json={})
            ).status == 200
            tools.lookup_id.assert_awaited_with(42, "self")
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
            for removed in ("gallery", "gallery-image", "unknown"):
                assert (
                    await client.post(f"/api/tools/{removed}", headers=headers, json={})
                ).status == 404
            for error, status in [
                (ToolsBusyError("busy"), 429),
                (ValueError("invalid"), 400),
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
