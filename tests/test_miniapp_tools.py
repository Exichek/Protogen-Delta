"""Инструменты в Mini App: личная доставка, безопасные URL и проверка initData."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.miniapp.tools import MiniAppTools, ToolsBusyError
from protogen_delta.services.media_download import MediaDownloadError


def test_id_selection_and_public_username_lookup() -> None:
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
        for invalid in ("-100123", "@bad", "https://t.me/group"):
            with pytest.raises(ValueError):
                await tools.lookup_id(42, invalid)
        bot.get_chat.return_value.type = "private"
        with pytest.raises(ValueError):
            await tools.lookup_id(43, "@person")

    asyncio.run(scenario())


@pytest.mark.parametrize("username", ["synthetic_user", None])
def test_numeric_id_returns_only_basic_telegram_profile(username: str | None) -> None:
    async def scenario() -> None:
        bot = AsyncMock()
        bot.get_chat.return_value = SimpleNamespace(
            id=9876543210,
            type="private",
            first_name="Тест",
            last_name="Профиль",
            username=username,
            bio="Не возвращать",
            phone_number="Не возвращать",
            pinned_message="Не возвращать",
        )
        result = await MiniAppTools(bot, Mock()).lookup_id(42, " 9876543210 ")
        bot.get_chat.assert_awaited_once_with(9876543210)
        bot.send_message.assert_not_awaited()
        assert set(result) == {"id", "title", "username", "type", "message"}
        assert result["title"] == "Тест Профиль" and result["username"] == username
        assert "9876543210" in str(result["message"])
        assert ("@synthetic_user" if username else "Username недоступен") in str(
            result["message"]
        )
        assert "Не возвращать" not in str(result)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "target",
    ["0", "01", "+123", "-123", "12.3", "1e9", "１２３", str(2**52), "9" * 1000],
)
def test_invalid_user_id_is_rejected_before_telegram(target: str) -> None:
    async def scenario() -> None:
        bot = AsyncMock()
        with pytest.raises(ValueError):
            await MiniAppTools(bot, Mock()).lookup_id(42, target)
        bot.get_chat.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("exception", [TelegramBadRequest, TelegramForbiddenError])
def test_unavailable_numeric_id_does_not_claim_user_does_not_exist(
    exception: type[TelegramBadRequest] | type[TelegramForbiddenError],
) -> None:
    async def scenario() -> None:
        bot = AsyncMock()
        bot.get_chat.side_effect = exception(
            method=Mock(), message="private backend detail"
        )
        with pytest.raises(ValueError, match="может не знать") as error:
            await MiniAppTools(bot, Mock()).lookup_id(42, "123")
        assert "private backend detail" not in str(error.value)
        bot.send_message.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "chat_type,returned_id", [("supergroup", 123), ("private", 124)]
)
def test_numeric_lookup_rejects_inconsistent_telegram_response(
    chat_type: str, returned_id: int
) -> None:
    async def scenario() -> None:
        bot = AsyncMock()
        bot.get_chat.return_value = SimpleNamespace(id=returned_id, type=chat_type)
        with pytest.raises(ValueError, match="не удалось"):
            await MiniAppTools(bot, Mock()).lookup_id(42, "123")

    asyncio.run(scenario())


def test_lookup_rate_limit_does_not_block_own_id_and_expires() -> None:
    async def scenario() -> None:
        now = [1.0]
        bot = AsyncMock()
        bot.get_chat.return_value = SimpleNamespace(
            id=123, type="private", first_name=None, last_name=None, username=None
        )
        tools = MiniAppTools(bot, Mock())
        tools._id_limiter = UserRateLimiter(clock=lambda: now[0])
        assert (await tools.lookup_id(42, "123"))["title"] == "Имя недоступно"
        with pytest.raises(ToolsBusyError, match="пару секунд"):
            await tools.lookup_id(42, "123")
        assert (await tools.lookup_id(42, "self"))["id"] == 42
        now[0] += 2
        await tools.lookup_id(42, "123")
        assert bot.get_chat.await_count == 2

    asyncio.run(scenario())


def test_lookup_caps_parallel_requests_and_releases_slots_on_cancellation() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        bot = AsyncMock()

        async def get_chat(target: int) -> SimpleNamespace:
            await release.wait()
            return SimpleNamespace(
                id=target,
                type="private",
                first_name="Тест",
                last_name=None,
                username=None,
            )

        bot.get_chat.side_effect = get_chat
        tools = MiniAppTools(bot, Mock())
        tasks = [asyncio.create_task(tools.lookup_id(n, "123")) for n in range(4)]
        await asyncio.sleep(0)
        with pytest.raises(ToolsBusyError, match="занят"):
            await tools.lookup_id(10, "123")
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        release.set()
        await asyncio.gather(*tasks[1:])
        assert (await tools.lookup_id(10, "123"))["id"] == 123
        assert tools._id_slots._value == 4

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
                (TimeoutError(), 504),
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
