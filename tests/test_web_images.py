"""Реальная отправка веб-картинок по просьбе владельца, без внешних API в тестах."""

import asyncio
import json
from dataclasses import replace
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from test_response_engine import _create_engine
from test_tool_loop import response

import protogen_delta.services.deepseek as deepseek_module
import protogen_delta.services.web_images as module
from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.web_image_request import is_web_image_request, web_image_limit
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.prompt_composer import PromptComposer
from protogen_delta.services.tools import ToolExecutor, default_registry
from protogen_delta.services.web_images import (
    MAX_IMAGE_BYTES,
    WebImages,
    _image,
    load_web_image,
    remember_web_image_sources,
)

PAGE = "https://example.org/game"
PHOTO = "https://cdn.example.org/game.jpg"


def picture(size: tuple[int, int] = (640, 360), fmt: str = "PNG") -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, "purple").save(stream, fmt)
    return stream.getvalue()


@pytest.mark.parametrize(
    "text",
    [
        "скинь какую нибудь фотку из интернета) по какой-то игре",
        "Пришли картинку из Cyberpunk 2077",
        "Можешь скинуть скриншот игры?",
        "Покажи фотографию Луны",
        "Найди обои по GTA",
    ],
)
def test_explicit_requests_enable_tool(text: str) -> None:
    assert is_web_image_request(text)
    assert "send_web_image" in PromptComposer.select_tools(text)


@pytest.mark.parametrize(
    "text",
    [
        "Не отправляй картинки",
        "Скинь только ссылку на картинку",
        "Как отправить картинки через бота?",
        "Он сказал «скинь фотку»",
        "Нарисуй картинку",
        "Сгенерируй и пришли картинку",
        "На фотке Cyberpunk",
        "> скинь фотку",
        "Расскажи новость об игре",
    ],
)
def test_non_requests_do_not_authorize_sending(text: str) -> None:
    assert not is_web_image_request(text)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "GIF"])
def test_convert_images_to_bounded_jpeg(fmt: str) -> None:
    data, width, height = _image(picture((2200, 1800), fmt))
    assert max(width, height) == 1920 and len(data) < MAX_IMAGE_BYTES
    with Image.open(BytesIO(data)) as decoded:
        assert decoded.format == "JPEG" and not decoded.getexif()
        decoded.load()


@pytest.mark.parametrize(
    "data",
    [b"", b"<html>not a photo</html>", b"bad", b"x" * (MAX_IMAGE_BYTES + 1)],
    ids=["empty", "html", "invalid", "oversize"],
)
def test_invalid_files(data: bytes) -> None:
    with pytest.raises(ValueError):
        _image(data)


@pytest.mark.parametrize("size", [(10, 10), (4096, 128), (5000, 5000)])
def test_invalid_dimensions(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        _image(picture(size))


def test_page_image_priority_relative_url_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = (
        b'<title>Game screenshots</title><meta property="og:image" content="/broken.jpg">'
        b'<img src="/logo.png"><img src="//cdn.example.org/game.jpg">'
    )
    fetch = AsyncMock(
        side_effect=[
            (html, PAGE, "text/html"),
            ValueError("unavailable"),
            (picture(), PHOTO, "image/png"),
        ]
    )
    monkeypatch.setattr(module, "fetch_public_page", fetch)
    data, url, title, width, height = asyncio.run(load_web_image(PAGE))
    assert url == PAGE and title == "Game screenshots" and (width, height) == (640, 360)
    assert fetch.await_args_list[1].args[0] == "https://example.org/broken.jpg"
    assert fetch.await_args_list[2].args[0] == PHOTO
    assert data.startswith(b"\xff\xd8")


@pytest.mark.parametrize(
    "content",
    [
        '<script><img src="https://x.example/a.jpg"></script>',
        '<meta property="og:image" content="data:image/png;base64,AA">',
        '<img src="javascript:alert(1)">',
        '<img src="https://u:p@example.org/p.jpg">',
        '<svg><image href="/x.jpg"></svg>',
    ],
)
def test_unusable_page_images_are_not_requested(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    fetch = AsyncMock(return_value=(content.encode(), PAGE, "text/html"))
    monkeypatch.setattr(module, "fetch_public_page", fetch)
    with pytest.raises(ValueError):
        asyncio.run(load_web_image(PAGE))
    fetch.assert_awaited_once()


def test_direct_image_is_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "fetch_public_page",
        AsyncMock(return_value=(picture(), PHOTO, "image/png")),
    )
    result = asyncio.run(load_web_image(PHOTO))
    assert result[1:] == (PHOTO, PHOTO, 640, 360)


def test_private_image_addresses_use_existing_ssrf_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import protogen_delta.services.tools.http as http

    monkeypatch.setattr(
        http.socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 443))]
    )
    with pytest.raises(ValueError):
        asyncio.run(load_web_image("https://example.org/private.png"))


def test_scope_success_dedup_and_delivery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        loader = AsyncMock(return_value=(picture(), PAGE, "Game", 640, 360))
        monkeypatch.setattr(module, "load_web_image", loader)
        bot = AsyncMock()
        service = WebImages(bot)
        assert (await service.send_tool({"url": PAGE}))["status"] == "not_sent"
        loader.assert_not_awaited()
        with service.turn(10, allowed=False):
            assert (await service.send_tool({"url": PAGE}))["status"] == "not_sent"
        with service.turn(10, allowed=True) as turn:
            result = await service.send_tool({"url": PAGE})
            assert result["status"] == "sent" and len(turn.receipts) == 1
            call = bot.send_photo.await_args
            assert call.args[0] == 10 and call.args[1].data == picture()
            assert (
                "http" not in call.kwargs["caption"]
                and call.kwargs["parse_mode"] is None
            )
            assert (await service.send_tool({"url": PAGE}))["already_sent"]
            bot.send_photo.assert_awaited_once()
        bot.send_photo.side_effect = TimeoutError()
        with service.turn(10, allowed=True) as turn:
            assert (await service.send_tool({"url": PAGE}))[
                "reason"
            ] == "telegram_delivery_unconfirmed"
            assert not turn.receipts
            assert (await service.send_tool({"url": PAGE}))[
                "reason"
            ] == "image_delivery_not_available"
        loader.side_effect = ValueError("private detail")
        with service.turn(10, allowed=True):
            for index in range(3):
                assert (await service.send_tool({"url": PAGE + str(index)}))[
                    "reason"
                ] == "public_image_unavailable"
            assert (await service.send_tool({"url": PAGE}))[
                "reason"
            ] == "image_delivery_not_available"

    asyncio.run(run())


@pytest.mark.parametrize("scope", ["own", "draft", "group", "forward"])
def test_engine_binds_only_current_explicit_private_request(
    monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module,
            "load_web_image",
            AsyncMock(return_value=(picture(), PAGE, "Game", 640, 360)),
        )
        bot = AsyncMock()
        service = WebImages(bot)
        engine, _, model, *_ = _create_engine()
        engine._web_images = service
        outcomes = []

        async def answer(**kwargs: Any) -> str:
            outcomes.append(await service.send_tool({"url": PAGE}))
            return "Вот картинка."

        model.chat.side_effect = answer
        token = current_reply_sources.set(
            (
                ReplySource(
                    "скинь фотку из игры",
                    "скинь фотку из игры",
                    scope != "forward",
                    True,
                ),
            )
        )
        try:
            await engine.respond(
                10,
                "скинь фотку из игры",
                deliver=None if scope == "draft" else AsyncMock(),
                chat_id=-123 if scope == "group" else None,
            )
        finally:
            current_reply_sources.reset(token)
        assert (outcomes[0]["status"] == "sent") == (scope == "own")
        history = engine._user_states.get(10).history
        if scope == "own":
            assert "Подтверждённая отправка фото:" in history[-1].assistant_message
        else:
            bot.send_photo.assert_not_awaited()

    asyncio.run(run())


def test_real_model_tool_loop_search_read_photo_then_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module,
            "load_web_image",
            AsyncMock(return_value=(picture(), PAGE, "Game", 640, 360)),
        )
        bot = AsyncMock()
        sender = WebImages(bot)
        registry = default_registry(web_images=sender, brave_search_api_key="synthetic")
        registry.tools["web_search"] = replace(
            registry.tools["web_search"],
            handler=AsyncMock(
                return_value={"results": [{"url": PAGE, "title": "Game"}]}
            ),
        )
        registry.tools["fetch_web_page"] = replace(
            registry.tools["fetch_web_page"],
            handler=AsyncMock(return_value={"url": PAGE, "text": "Game screenshots"}),
        )

        def tool(name: str, args: dict[str, str]) -> Any:
            result = response(1, json.dumps(args))
            result.choices[0].message.tool_calls[0].function.name = name
            return result

        final = response()
        final.choices[0].message.content = (
            "Не могу приложить картинку. Вот ссылки: " + PAGE
        )
        client = Mock()
        client.chat.completions.create = AsyncMock(
            side_effect=[
                tool("web_search", {"query": "game screenshots"}),
                tool("fetch_web_page", {"url": PAGE}),
                tool(
                    "send_web_image",
                    {"url": PAGE, "description": "Скриншот Cyberpunk 2077."},
                ),
                final,
            ]
        )
        monkeypatch.setattr(deepseek_module, "AsyncOpenAI", Mock(return_value=client))
        backend = DeepSeekService(
            "test",
            "https://test.local",
            "test",
            tools=ToolExecutor(registry),
            verify_web_answers=True,
        )
        engine, *_ = _create_engine()
        engine._deepseek = backend
        engine._web_images = sender
        delivered = AsyncMock()
        result = await engine.respond(
            10, "Скинь фотку из интернета по игре", deliver=delivered
        )
        assert result == "Фото отправлено."
        delivered.assert_awaited_once_with(result)
        assert client.chat.completions.create.await_count == 3
        calls = client.chat.completions.create.await_args_list
        assert [c.kwargs["tool_choice"]["function"]["name"] for c in calls[:3]] == [
            "web_search",
            "fetch_web_page",
            "send_web_image",
        ]
        bot.send_photo.assert_awaited_once()
        assert (
            "Подтверждённая отправка фото:"
            in engine._user_states.get(10).history[-1].assistant_message
        )

    asyncio.run(run())


def test_failed_image_cannot_be_described_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module, "load_web_image", AsyncMock(side_effect=ValueError("unavailable"))
        )
        bot = AsyncMock()
        sender = WebImages(bot)
        engine, _, model, *_ = _create_engine()
        engine._web_images = sender

        async def answer(**kwargs: Any) -> str:
            await sender.send_tool({"url": PAGE})
            return "Отправил картинку, смотри!"

        model.chat.side_effect = answer
        result = await engine.respond(10, "Скинь фотку из игры", deliver=AsyncMock())
        assert "скачать не получилось" in result
        assert (
            "Подтверждённая отправка фото"
            not in engine._user_states.get(10).history[-1].assistant_message
        )
        bot.send_photo.assert_not_awaited()

    asyncio.run(run())


def test_direct_image_actual_tool_loop_acknowledges_only_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module,
            "load_web_image",
            AsyncMock(return_value=(picture(), PHOTO, PHOTO, 640, 360)),
        )
        sender = WebImages(AsyncMock())
        registry = default_registry(web_images=sender)
        first = response(
            1, json.dumps({"url": PHOTO, "description": "Картинка по твоему запросу."})
        )
        first.choices[0].message.tool_calls[0].function.name = "send_web_image"
        final = response()
        final.choices[0].message.content = (
            "На картинке точно новое дополнение 2027 года!"
        )
        client = Mock()
        client.chat.completions.create = AsyncMock(side_effect=[first, final])
        monkeypatch.setattr(deepseek_module, "AsyncOpenAI", Mock(return_value=client))
        backend = DeepSeekService(
            "key",
            "https://test.local",
            "test",
            tools=ToolExecutor(registry),
            verify_web_answers=True,
        )
        # Прямой image URL не требует текстового чтения страницы.
        with sender.turn(10, allowed=True):
            result = await backend.chat(
                "system", "Скинь картинку " + PHOTO, tool_names={"send_web_image"}
            )
        assert result == "Фото отправлено."
        assert client.chat.completions.create.await_count == 1
        assert (
            client.chat.completions.create.await_args_list[0].kwargs["tool_choice"][
                "function"
            ]["name"]
            == "send_web_image"
        )

    asyncio.run(run())


def test_cancelled_delivery_is_finished_once(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module,
            "load_web_image",
            AsyncMock(return_value=(picture(), PAGE, "Game", 640, 360)),
        )
        bot = AsyncMock()
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold(*args: Any, **kwargs: Any) -> None:
            entered.set()
            await release.wait()

        bot.send_photo.side_effect = hold
        sender = WebImages(bot)
        with sender.turn(10, allowed=True) as turn:
            task = asyncio.create_task(sender.send_tool({"url": PAGE}))
            await entered.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(turn.receipts) == 1
            assert (await sender.send_tool({"url": PAGE}))["status"] == "not_sent"
        bot.send_photo.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize(
    "text,limit",
    [
        ("Скинь скриншот из Cyberpunk 2077", 1),
        ("Скинь какую-нибудь фотку по игре", 1),
        ("Пришли две картинки из Cyberpunk", 2),
        ("Скинь 10 скриншотов", 3),
        ("Скинь картинку. Он сказал «пришли три картинки»", 1),
        ("Пришли три скриншота", 3),
        ("Пришли одну картинку", 1),
    ],
)
def test_requested_image_count(text: str, limit: int) -> None:
    assert web_image_limit(text) == limit


def test_fallback_sources_and_single_photo_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        loader = AsyncMock(
            side_effect=[
                ValueError("blocked"),
                (
                    picture(),
                    PAGE + "/alternative",
                    "Cyberpunk 2077 Screenshots",
                    640,
                    360,
                ),
            ]
        )
        monkeypatch.setattr(module, "load_web_image", loader)
        bot = AsyncMock()
        sender = WebImages(bot)
        with sender.turn(10, allowed=True) as turn:
            remember_web_image_sources(
                json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "results": [
                                {"url": PAGE},
                                {"url": PAGE + "/alternative"},
                            ]
                        },
                    }
                )
            )
            result = await sender.send_tool(
                {"url": PAGE, "description": "Скриншот Cyberpunk 2077 — Найт-Сити."}
            )
            assert result["status"] == "sent" and turn.attempts == 2
            assert [call.args[0] for call in loader.await_args_list] == [
                PAGE,
                PAGE + "/alternative",
            ]
            assert (
                bot.send_photo.await_args.kwargs["caption"]
                == "Скриншот Cyberpunk 2077 — Найт-Сити."
            )
            await sender.send_tool({"url": PHOTO})
            bot.send_photo.assert_awaited_once()
        with sender.turn(10, allowed=True) as fresh:
            assert not fresh.candidates and not fresh.tried
        assert len(turn.receipts) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "description",
    [
        "Скриншот [Cyberpunk](https://example.org). https://example.org/article",
        "English article title https://example.org",
        "Скриншот Cyberpunk, example.org/article",
        "",
    ],
)
def test_captions_have_description_without_links(description: str) -> None:
    caption = module._caption(description)
    assert "http" not in caption and "example.org" not in caption
    assert caption and len(caption) <= 200


def test_successful_photo_cannot_be_denied_by_final_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module,
            "load_web_image",
            AsyncMock(return_value=(picture(), PAGE, "English headline", 640, 360)),
        )
        sender = WebImages(AsyncMock())
        engine, _, model, *_ = _create_engine()
        engine._web_images = sender
        engine.supplements = Mock()

        async def answer(**kwargs: Any) -> str:
            await sender.send_tool(
                {"url": PAGE, "description": "Скриншот Cyberpunk 2077."}
            )
            return "Не могу прикрепить картинку. Источник: " + PAGE

        model.chat.side_effect = answer
        delivered = AsyncMock()
        result = await engine.respond(
            10, "Скинь скриншот из Cyberpunk 2077", deliver=delivered
        )
        assert result == "Фото отправлено."
        delivered.assert_awaited_once_with(result)
        history = engine._user_states.get(10).history[-1].assistant_message
        assert "Скриншот Cyberpunk 2077" in history
        assert "Не могу" not in history and "http" not in history
        engine.supplements.schedule.assert_not_called()

    asyncio.run(run())


def test_fallback_stops_after_three_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        loader = AsyncMock(side_effect=ValueError("unavailable"))
        monkeypatch.setattr(module, "load_web_image", loader)
        sender = WebImages(AsyncMock())
        with sender.turn(10, allowed=True) as turn:
            remember_web_image_sources(
                json.dumps(
                    {
                        "ok": True,
                        "data": {"results": [{"url": PAGE + str(i)} for i in range(6)]},
                    }
                )
            )
            assert (await sender.send_tool({"url": PAGE}))["status"] == "not_sent"
            assert turn.attempts == 3 and loader.await_count == 3
            assert (await sender.send_tool({"url": PHOTO}))[
                "reason"
            ] == "image_delivery_not_available"

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload", ["bad", "[]", '{"ok":true}', '{"ok":true,"data":{"results":null}}']
)
def test_invalid_search_does_not_authorize_fallback(payload: str) -> None:
    remember_web_image_sources(payload)
    with WebImages(AsyncMock()).turn(10, allowed=True) as turn:
        remember_web_image_sources(payload)
        assert not turn.candidates


def test_explicit_multiple_images_have_bounded_separate_deliveries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module,
            "load_web_image",
            AsyncMock(
                side_effect=[
                    (picture(), PAGE, "Game", 640, 360),
                    (picture((800, 600)), PHOTO, "Game", 800, 600),
                ]
            ),
        )
        bot = AsyncMock()
        sender = WebImages(bot)
        with sender.turn(10, allowed=True, limit=2) as turn:
            await sender.send_tool({"url": PAGE})
            assert not module.web_image_delivery_complete()
            await sender.send_tool({"url": PHOTO})
            assert module.web_image_delivery_complete()
            assert len(turn.receipts) == bot.send_photo.await_count == 2

    asyncio.run(run())


def test_alternative_image_does_not_inherit_another_game_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            module,
            "load_web_image",
            AsyncMock(
                side_effect=[
                    ValueError("blocked"),
                    (picture(), PHOTO, "Elden Ring Screenshots", 640, 360),
                ]
            ),
        )
        bot = AsyncMock()
        sender = WebImages(bot)
        with sender.turn(10, allowed=True):
            remember_web_image_sources(
                json.dumps({"ok": True, "data": {"results": [{"url": PHOTO}]}})
            )
            await sender.send_tool(
                {"url": PAGE, "description": "Скриншот Cyberpunk 2077."}
            )
        caption = bot.send_photo.await_args.kwargs["caption"]
        assert caption == "Картинка по твоему запросу."
        assert "Cyberpunk" not in caption and "http" not in caption

    asyncio.run(run())
