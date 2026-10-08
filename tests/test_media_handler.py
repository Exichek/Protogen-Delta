"""Тесты обработки изображений и стикеров."""

import asyncio
import io
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, Mock, call

import pytest
from aiogram import Bot, Router
from aiogram.types import Message

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.media import (
    IMAGE_DOWNLOAD_ERROR_REPLY,
    IMAGE_TOO_LARGE_REPLY,
    MAX_IMAGE_BYTES,
    UNSUPPORTED_IMAGE_REPLY,
    _has_supported_animation_document,
    create_media_router,
)
from protogen_delta.handlers.text import BUSY_REPLY, RATE_LIMIT_REPLY
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine

TEST_USER_ID = 123456
PNG_DATA = b"\x89PNG\r\n\x1a\ncontent"
JPEG_DATA = b"\xff\xd8\xffcontent"
WEBP_DATA = b"RIFF\x08\x00\x00\x00WEBPcontent"


def _message(**values: Any) -> tuple[Message, AsyncMock]:
    """Создать минимальное сообщение Telegram для обработчика медиа."""
    message = Mock(spec=Message)
    message.from_user = SimpleNamespace(id=TEST_USER_ID)
    message.chat = SimpleNamespace(id=777, type="private")
    message.caption = values.pop("caption", None)
    message.media_group_id = values.pop("media_group_id", None)
    message.photo = values.pop("photo", None)
    message.document = values.pop("document", None)
    message.sticker = values.pop("sticker", None)
    message.animation = values.pop("animation", None)
    message.video = values.pop("video", None)
    for key, value in values.items():
        setattr(message, key, value)
    message.answer = AsyncMock()
    return cast(Message, message), message.answer


def _router(data: bytes) -> tuple[Router, AsyncMock, AsyncMock]:
    """Создать роутер с загрузкой заданных байтов и ответом движка."""
    engine = AsyncMock(spec=ResponseEngine)

    async def respond(
        user_id: int,
        text: str,
        deliver: Any,
        *,
        images: Any,
    ) -> None:
        await deliver("Вижу изображение.")

    engine.respond_and_deliver.side_effect = respond
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> io.BytesIO:
        destination.write(data)
        return destination

    bot.download.side_effect = download
    router = create_media_router(
        cast(ResponseEngine, engine),
        cast(Bot, bot),
        native_work=BlockingWorkPool(2),
    )
    return router, engine, bot


def test_photo_is_downloaded_and_sent_to_response_engine() -> None:
    """Фотография должна попадать в vision-запрос вместе с подписью."""
    router, engine, bot = _router(JPEG_DATA)
    photo = SimpleNamespace(file_id="photo-id", file_size=len(JPEG_DATA))
    message, answer = _message(photo=[photo], caption="Что здесь?")

    asyncio.run(router.message.handlers[0].callback(message))

    bot.download.assert_awaited_once_with("photo-id", destination=ANY)
    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert call.args[:2] == (TEST_USER_ID, "Что здесь?")
    assert call.kwargs["images"][0].data == JPEG_DATA
    assert call.kwargs["images"][0].mime_type == "image/jpeg"
    answer.assert_awaited_once_with("Вижу изображение.")


def test_image_document_uses_extension_and_real_signature() -> None:
    """Изображение-файл должно распознаваться и без точного MIME Telegram."""
    router, engine, _ = _router(PNG_DATA)
    document = SimpleNamespace(
        file_id="document-id",
        file_size=len(PNG_DATA),
        mime_type="application/octet-stream",
        file_name="screen.PNG",
    )
    message, _ = _message(document=document)

    asyncio.run(router.message.handlers[1].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert call.args[1] == "[Пользователь отправил изображение-файл без подписи]"
    assert call.kwargs["images"][0].mime_type == "image/png"


def test_static_sticker_passes_emoji_and_webp() -> None:
    """Статический стикер должен передаваться модели как WebP и жест."""
    router, engine, _ = _router(WEBP_DATA)
    sticker = SimpleNamespace(
        file_id="sticker-id",
        file_size=len(WEBP_DATA),
        is_animated=False,
        is_video=False,
        emoji="😼",
        thumbnail=None,
    )
    message, _ = _message(sticker=sticker)

    asyncio.run(router.message.handlers[2].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert "😼" in call.args[1]
    assert call.kwargs["images"][0].mime_type == "image/webp"
    assert call.kwargs["images"][0].label == "статический стикер"


def test_invalid_tgs_sticker_falls_back_to_thumbnail() -> None:
    """Повреждённый TGS должен использовать Telegram-превью как запасной путь."""
    router, engine, bot = _router(JPEG_DATA)
    thumbnail = SimpleNamespace(file_id="thumb-id", file_size=len(JPEG_DATA))
    sticker = SimpleNamespace(
        file_id="animated-id",
        file_size=100,
        is_animated=True,
        is_video=False,
        emoji="🔥",
        thumbnail=thumbnail,
    )
    message, _ = _message(sticker=sticker)

    asyncio.run(router.message.handlers[2].callback(message))

    assert bot.download.await_args_list == [
        call("animated-id", destination=ANY),
        call("thumb-id", destination=ANY),
    ]
    image = engine.respond_and_deliver.await_args.kwargs["images"][0]
    assert image.label == "превью TGS-стикера"


def test_animated_tgs_sticker_uses_rendered_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Валидный TGS должен передавать последовательность кадров без thumbnail."""
    router, engine, bot = _router(b"tgs")
    frames = (
        ImageInput(PNG_DATA, "image/png", "TGS-анимация, кадр 1"),
        ImageInput(PNG_DATA, "image/png", "TGS-анимация, кадр 2"),
    )
    monkeypatch.setattr(
        "protogen_delta.handlers.media.extract_tgs_frames",
        lambda *args, **kwargs: frames,
    )
    sticker = SimpleNamespace(
        file_id="animated-id",
        file_size=100,
        is_animated=True,
        is_video=False,
        emoji="🔥",
        thumbnail=None,
    )
    message, _ = _message(sticker=sticker)

    asyncio.run(router.message.handlers[2].callback(message))

    bot.download.assert_awaited_once_with("animated-id", destination=ANY)
    call_args = engine.respond_and_deliver.await_args
    assert call_args is not None
    assert call_args.kwargs["images"] == frames


def test_video_sticker_uses_actual_file_without_thumbnail() -> None:
    """Видеостикер должен декодировать сам файл, а не требовать превью."""
    router, engine, _ = _router(JPEG_DATA)
    sticker = SimpleNamespace(
        file_id="video-sticker",
        file_size=len(JPEG_DATA),
        is_animated=False,
        is_video=True,
        emoji="🔥",
        thumbnail=None,
    )
    message, answer = _message(sticker=sticker)

    asyncio.run(router.message.handlers[2].callback(message))

    answer.assert_awaited_once_with("Вижу изображение.")
    assert (
        "видеостикер" in engine.respond_and_deliver.await_args.kwargs["images"][0].label
    )


def test_tgs_sticker_without_thumbnail_reports_unsupported() -> None:
    router, engine, _ = _router(JPEG_DATA)
    sticker = SimpleNamespace(
        file_id="tgs",
        file_size=10,
        is_animated=True,
        is_video=False,
        emoji="🔥",
        thumbnail=None,
    )
    message, answer = _message(sticker=sticker)

    asyncio.run(router.message.handlers[2].callback(message))

    answer.assert_awaited_once_with(UNSUPPORTED_IMAGE_REPLY)
    engine.respond_and_deliver.assert_not_awaited()


def test_telegram_animation_uses_actual_file() -> None:
    """Telegram GIF должен декодировать сам файл."""
    router, engine, bot = _router(JPEG_DATA)
    thumbnail = SimpleNamespace(file_id="gif-thumb", file_size=len(JPEG_DATA))
    animation = SimpleNamespace(
        file_id="gif-file",
        file_size=len(JPEG_DATA),
        thumbnail=thumbnail,
    )
    message, _ = _message(animation=animation, caption="Что скажешь?")

    asyncio.run(router.message.handlers[3].callback(message))

    bot.download.assert_awaited_once_with("gif-file", destination=ANY)
    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert call.args[1] == "Что скажешь?"
    assert "GIF-анимация" in call.kwargs["images"][0].label


def test_telegram_animation_without_thumbnail_still_uses_file() -> None:
    """GIF без Telegram-превью всё равно должен обрабатываться по исходнику."""
    router, engine, _ = _router(JPEG_DATA)
    message, answer = _message(
        animation=SimpleNamespace(
            file_id="gif-file",
            file_size=len(JPEG_DATA),
            thumbnail=None,
        )
    )

    asyncio.run(router.message.handlers[3].callback(message))

    answer.assert_awaited_once_with("Вижу изображение.")
    engine.respond_and_deliver.assert_awaited_once()


def test_animation_document_uses_actual_file() -> None:
    """MP4-анимация, отправленная файлом, декодируется по исходнику."""
    router, engine, bot = _router(JPEG_DATA)
    thumbnail = SimpleNamespace(file_id="file-thumb", file_size=len(JPEG_DATA))
    document = SimpleNamespace(
        file_id="animation-file",
        file_size=500,
        mime_type="video/mp4",
        file_name="reaction.mp4",
        thumbnail=thumbnail,
    )
    message, _ = _message(document=document)

    asyncio.run(router.message.handlers[4].callback(message))

    bot.download.assert_awaited_once_with("animation-file", destination=ANY)
    image = engine.respond_and_deliver.await_args.kwargs["images"][0]
    assert "файлом" in image.label


def test_animation_document_filter_requires_supported_video() -> None:
    """Роутер должен перехватывать поддерживаемые видео даже без превью."""
    thumbnail = SimpleNamespace(file_id="thumb", file_size=10)
    message, _ = _message(
        document=SimpleNamespace(
            mime_type="video/mp4",
            file_name="reaction.bin",
            thumbnail=thumbnail,
        )
    )
    assert _has_supported_animation_document(message) is True

    message.document.thumbnail = None  # type: ignore[union-attr]
    assert _has_supported_animation_document(message) is True

    cast(Any, message).document = SimpleNamespace(
        mime_type="application/octet-stream",
        file_name="reaction.webm",
        thumbnail=thumbnail,
    )
    assert _has_supported_animation_document(message) is True


def test_regular_video_uses_multiple_frame_pipeline() -> None:
    router, engine, bot = _router(JPEG_DATA)
    video = SimpleNamespace(file_id="video-file", file_size=len(JPEG_DATA))
    message, _ = _message(video=video, caption="Что происходит?")

    asyncio.run(router.message.handlers[5].callback(message))

    bot.download.assert_awaited_once_with("video-file", destination=ANY)
    call = engine.respond_and_deliver.await_args
    assert call.args[1] == "Что происходит?"
    assert "видео" in call.kwargs["images"][0].label


def test_moving_media_reports_size_format_and_download_errors() -> None:
    large_router, large_engine, large_bot = _router(JPEG_DATA)
    large, large_answer = _message(
        video=SimpleNamespace(file_id="large", file_size=MAX_IMAGE_BYTES + 1)
    )
    asyncio.run(large_router.message.handlers[5].callback(large))
    large_answer.assert_awaited_once_with(IMAGE_TOO_LARGE_REPLY)
    large_bot.download.assert_not_awaited()
    large_engine.respond_and_deliver.assert_not_awaited()

    broken_router, broken_engine, _ = _router(b"not-video")
    broken, broken_answer = _message(
        video=SimpleNamespace(file_id="broken", file_size=9)
    )
    asyncio.run(broken_router.message.handlers[5].callback(broken))
    broken_answer.assert_awaited_once_with(UNSUPPORTED_IMAGE_REPLY)
    broken_engine.respond_and_deliver.assert_not_awaited()

    error_router, error_engine, error_bot = _router(JPEG_DATA)
    error_bot.download.side_effect = OSError("network")
    failed, failed_answer = _message(
        video=SimpleNamespace(file_id="failed", file_size=9)
    )
    asyncio.run(error_router.message.handlers[5].callback(failed))
    failed_answer.assert_awaited_once_with(IMAGE_DOWNLOAD_ERROR_REPLY)
    error_engine.respond_and_deliver.assert_not_awaited()


def test_large_image_is_rejected_before_download() -> None:
    """Заведомо большой файл не должен скачиваться из Telegram."""
    router, engine, bot = _router(JPEG_DATA)
    photo = SimpleNamespace(file_id="large-id", file_size=MAX_IMAGE_BYTES + 1)
    message, answer = _message(photo=[photo])

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_awaited_once_with(IMAGE_TOO_LARGE_REPLY)
    bot.download.assert_not_awaited()
    engine.respond_and_deliver.assert_not_awaited()


def test_invalid_file_signature_is_rejected() -> None:
    """Расширение не должно позволять передать модели произвольный файл."""
    router, engine, _ = _router(b"not an image")
    document = SimpleNamespace(
        file_id="fake-id",
        file_size=12,
        mime_type="image/png",
        file_name="fake.png",
    )
    message, answer = _message(document=document)

    asyncio.run(router.message.handlers[1].callback(message))

    answer.assert_awaited_once_with(UNSUPPORTED_IMAGE_REPLY)
    engine.respond_and_deliver.assert_not_awaited()


def test_download_failure_returns_stable_reply() -> None:
    """Сбой Telegram при загрузке не должен падать наружу."""
    router, engine, bot = _router(JPEG_DATA)
    bot.download.side_effect = OSError("network failed")
    photo = SimpleNamespace(file_id="photo-id", file_size=10)
    message, answer = _message(photo=[photo])

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_awaited_once_with(IMAGE_DOWNLOAD_ERROR_REPLY)
    engine.respond_and_deliver.assert_not_awaited()


def test_busy_engine_returns_busy_reply() -> None:
    """Параллельная обработка пользователя должна получать общий busy-ответ."""
    router, engine, _ = _router(JPEG_DATA)
    engine.respond_and_deliver.side_effect = ResponseBusyError
    photo = SimpleNamespace(file_id="photo-id", file_size=10)
    message, answer = _message(photo=[photo])

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_awaited_once_with(BUSY_REPLY)


def test_media_uses_shared_rate_limit() -> None:
    """Медиа и текст должны подчиняться одному пользовательскому cooldown."""
    engine = AsyncMock(spec=ResponseEngine)
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> io.BytesIO:
        destination.write(JPEG_DATA)
        return destination

    bot.download.side_effect = download
    limiter = UserRateLimiter(cooldown_seconds=2.0, clock=lambda: 100.0)
    router = create_media_router(
        cast(ResponseEngine, engine),
        cast(Bot, bot),
        rate_limiter=limiter,
    )
    assert limiter.allow(TEST_USER_ID)
    photo = SimpleNamespace(file_id="photo-id", file_size=10)
    message, answer = _message(photo=[photo])

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_awaited_once_with(RATE_LIMIT_REPLY)
    bot.download.assert_awaited_once()


def test_non_image_document_is_ignored_for_later_document_router() -> None:
    """PDF не должен ошибочно отправляться в vision как изображение."""
    router, engine, bot = _router(PNG_DATA)
    document = SimpleNamespace(
        file_id="pdf-id",
        file_size=100,
        mime_type="application/pdf",
        file_name="report.pdf",
    )
    message, answer = _message(document=document)

    asyncio.run(router.message.handlers[1].callback(message))

    answer.assert_not_awaited()
    bot.download.assert_not_awaited()
    engine.respond_and_deliver.assert_not_awaited()


def test_media_without_sender_is_ignored() -> None:
    """Сообщение без подтверждённого пользователя не должно обрабатываться."""
    router, engine, bot = _router(JPEG_DATA)
    photo = SimpleNamespace(file_id="photo-id", file_size=10)
    message, answer = _message(photo=[photo])
    message.from_user = None

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_not_awaited()
    bot.download.assert_not_awaited()
    engine.respond_and_deliver.assert_not_awaited()


def test_album_is_combined_into_one_multimodal_turn() -> None:
    """Элементы Telegram-альбома должны вызывать один ответ с общим контекстом."""
    engine = AsyncMock(spec=ResponseEngine)
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> io.BytesIO:
        destination.write(JPEG_DATA)
        return destination

    bot.download.side_effect = download
    router = create_media_router(
        cast(ResponseEngine, engine),
        cast(Bot, bot),
        album_delay_seconds=0.01,
    )
    first_photo = SimpleNamespace(file_id="one", file_size=len(JPEG_DATA))
    second_photo = SimpleNamespace(file_id="two", file_size=len(JPEG_DATA))
    first, _ = _message(
        photo=[first_photo],
        caption="Сравни их",
        media_group_id="album-1",
    )
    second, _ = _message(photo=[second_photo], media_group_id="album-1")

    async def scenario() -> None:
        await router.message.handlers[0].callback(first)
        await router.message.handlers[0].callback(second)
        await asyncio.sleep(0.03)

    asyncio.run(scenario())

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert call.args[1] == "Сравни их"
    assert len(call.kwargs["images"]) == 2
    assert bot.download.await_count == 2


def test_album_counts_slow_last_download_before_reply() -> None:
    """Медленное скачивание не должно отделять последний элемент альбома."""
    engine = AsyncMock(spec=ResponseEngine)
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> io.BytesIO:
        if file_id == "six":
            await asyncio.sleep(0.03)
        destination.write(JPEG_DATA)
        return destination

    bot.download.side_effect = download
    router = create_media_router(
        cast(ResponseEngine, engine),
        cast(Bot, bot),
        album_delay_seconds=0.01,
    )

    async def scenario() -> None:
        for number in range(1, 7):
            photo = SimpleNamespace(
                file_id="six" if number == 6 else str(number),
                file_size=len(JPEG_DATA),
            )
            message, _ = _message(
                photo=[photo],
                caption="Сколько их?" if number == 1 else None,
                media_group_id="album-six",
            )
            await router.message.handlers[0].callback(message)
        await asyncio.sleep(0.06)

    asyncio.run(scenario())

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert len(call.kwargs["images"]) == 6
    assert bot.download.await_count == 6


def test_album_keeps_valid_images_when_one_download_is_invalid() -> None:
    """Один повреждённый элемент не должен ломать весь Telegram-альбом."""
    engine = AsyncMock(spec=ResponseEngine)
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> io.BytesIO:
        destination.write(b"broken" if file_id == "bad" else JPEG_DATA)
        return destination

    bot.download.side_effect = download
    router = create_media_router(
        cast(ResponseEngine, engine),
        cast(Bot, bot),
        album_delay_seconds=0.01,
    )
    valid, _ = _message(
        photo=[SimpleNamespace(file_id="good", file_size=10)],
        media_group_id="partial",
    )
    invalid, _ = _message(
        photo=[SimpleNamespace(file_id="bad", file_size=10)],
        media_group_id="partial",
    )

    async def scenario() -> None:
        await router.message.handlers[0].callback(valid)
        await router.message.handlers[0].callback(invalid)
        await asyncio.sleep(0.03)

    asyncio.run(scenario())

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert len(call.kwargs["images"]) == 1


def test_album_delay_must_be_positive() -> None:
    """Некорректная задержка сборки альбома должна отклоняться сразу."""
    engine = AsyncMock(spec=ResponseEngine)
    bot = AsyncMock(spec=Bot)

    with pytest.raises(ValueError, match="больше нуля"):
        create_media_router(
            cast(ResponseEngine, engine),
            cast(Bot, bot),
            album_delay_seconds=0,
        )
