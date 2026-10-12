"""Только ссылки на похожие публикации, без уверенных догадок."""

import asyncio
import io
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.types import Message
from PIL import Image

import protogen_delta.services.image_source as module
from protogen_delta.handlers.image_source import create_image_source_router
from protogen_delta.services.image_source import (
    ImageSourceError,
    ImageSourceMatch,
    ImageSourceService,
    image_preview,
    parse_matches,
)


def photo_data() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (1500, 200), "red").save(output, "PNG")
    return output.getvalue()


def test_preview_is_small_jpeg_without_metadata() -> None:
    data = image_preview(photo_data())
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "JPEG" and image.width == 1024
        assert not image.getexif()
    with pytest.raises(ImageSourceError):
        image_preview(b"not-image")
    with pytest.raises(ImageSourceError):
        image_preview(b"")


def test_source_results_filter_uncertain_unsafe_and_duplicate_links() -> None:
    def result(score: str, url: str) -> dict[str, Any]:
        return {
            "header": {"similarity": score, "index_name": "Pixiv"},
            "data": {"ext_urls": [url]},
        }

    payload: dict[str, Any] = {
        "header": {"status": 0},
        "results": [
            result("95", "https://www.pixiv.net/artworks/1"),
            result("96", "https://www.pixiv.net/artworks/1"),
            result("30", "https://example.com/weak"),
            result("NaN", "https://example.com/nan"),
            result("99", "javascript:alert(1)"),
            result("98", "https://user:pass@example.com/a"),
            result("90", "https://example.com/valid"),
            {"header": {}, "data": {}},
            None,
        ],
    }
    matches = parse_matches(payload)
    assert [item.url for item in matches] == [
        "https://www.pixiv.net/artworks/1",
        "https://example.com/valid",
    ]
    invalid: Any
    for invalid in (
        None,
        {"header": {}, "results": {}},
        {"header": {"status": -1}},
        {"header": []},
    ):
        with pytest.raises(ImageSourceError):
            parse_matches(invalid)


def test_source_service_upload_and_provider_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps({"header": {"status": 0}, "results": []}).encode()

    async def chunks(size: int) -> Any:
        yield body

    response = SimpleNamespace(status=200, content=SimpleNamespace(iter_chunked=chunks))
    request_context = AsyncMock()
    request_context.__aenter__.return_value = response
    session = Mock()
    session.post.return_value = request_context
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(module, "ClientSession", Mock(return_value=context))
    service = ImageSourceService("test-secret")
    assert asyncio.run(service.search(photo_data())) == ()
    assert session.post.call_args.args[0] == "https://saucenao.com/search.php"
    assert session.post.call_args.kwargs["allow_redirects"] is False
    response.status = 429
    with pytest.raises(ImageSourceError):
        asyncio.run(service.search(photo_data()))
    response.status = 200
    body = b"x" * 1_048_577
    with pytest.raises(ImageSourceError):
        asyncio.run(service.search(photo_data()))
    with pytest.raises(ImageSourceError, match="SAUCENAO"):
        asyncio.run(ImageSourceService("").search(photo_data()))


def test_source_command_help_results_and_no_matches() -> None:
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> None:
        destination.write(photo_data())

    bot.download.side_effect = download
    service = AsyncMock(spec=ImageSourceService)
    service.search.return_value = (
        ImageSourceMatch(95, "Pixiv", "https://example.com/art"),
    )
    router = create_image_source_router(
        cast(Bot, bot), cast(ImageSourceService, service)
    )
    message = Mock(spec=Message)
    message.from_user = SimpleNamespace(id=1)
    message.reply_to_message = None
    message.answer = AsyncMock()
    callback = router.message.handlers[0].callback
    asyncio.run(callback(message))
    assert "уменьшенную" in message.answer.await_args.args[0]
    message.reply_to_message = SimpleNamespace(
        photo=[SimpleNamespace(file_id="photo", file_size=20)]
    )
    asyncio.run(callback(message))
    assert "95.0%" in message.answer.await_args.args[0]
    message.from_user = SimpleNamespace(id=2)
    service.search.return_value = ()
    asyncio.run(callback(message))
    assert "не нашёл" in message.answer.await_args.args[0]
    message.from_user = SimpleNamespace(id=3)
    service.search.side_effect = ImageSourceError("Не настроен")
    asyncio.run(callback(message))
    assert "Не настроен" in message.answer.await_args.args[0]
