"""Обработчики изображений и визуальных стикеров."""

import asyncio
import io
import logging
from collections.abc import Sequence
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.delivery import create_reply_delivery, show_typing
from protogen_delta.handlers.text import BUSY_REPLY, RATE_LIMIT_REPLY
from protogen_delta.services.animation_frames import extract_animation_frames
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine
from protogen_delta.services.tgs_frames import MAX_TGS_BYTES, extract_tgs_frames

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_TOO_LARGE_REPLY = "Картинка слишком большая: сейчас могу скачать до 20 МБ."
IMAGE_DOWNLOAD_ERROR_REPLY = "Не смог скачать картинку из Telegram. Попробуй ещё раз."
UNSUPPORTED_IMAGE_REPLY = "Этот формат изображения я пока не умею смотреть."

_EXTENSION_MIME_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_SUPPORTED_MIME_TYPES = frozenset(_EXTENSION_MIME_TYPES.values())
_ANIMATION_MIME_TYPES = frozenset({"video/mp4", "video/webm"})
_ANIMATION_EXTENSIONS = frozenset({".mp4", ".webm"})


def _detected_mime_type(data: bytes) -> str | None:
    """Определить поддерживаемый формат изображения по сигнатуре файла."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _document_mime_type(mime_type: str | None, file_name: str | None) -> str | None:
    """Получить поддерживаемый MIME-тип изображения-документа."""
    normalized = (mime_type or "").lower()
    if normalized in _SUPPORTED_MIME_TYPES:
        return normalized
    if file_name:
        return _EXTENSION_MIME_TYPES.get(Path(file_name).suffix.lower())
    return None


def _has_supported_image_document(message: Message) -> bool:
    """Пропустить в vision только поддерживаемое изображение-документ."""
    document = message.document
    return (
        document is not None
        and _document_mime_type(
            document.mime_type,
            document.file_name,
        )
        is not None
    )


def _has_supported_animation_document(message: Message) -> bool:
    """Распознать поддерживаемую анимацию, отправленную обычным файлом."""
    document = message.document
    if document is None:
        return False
    mime_type = (document.mime_type or "").lower()
    suffix = Path(document.file_name or "").suffix.lower()
    return mime_type in _ANIMATION_MIME_TYPES or suffix in _ANIMATION_EXTENSIONS


async def _download_image(
    bot: Bot,
    file_id: str,
    *,
    expected_mime_type: str,
    file_size: int | None,
    label: str,
) -> ImageInput:
    """Скачать изображение и проверить размер и реальный формат."""
    if file_size is not None and file_size > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")

    destination = io.BytesIO()
    await bot.download(file_id, destination=destination)
    data = destination.getvalue()

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")

    detected = _detected_mime_type(data)
    if detected is None:
        raise ValueError("unsupported_image")

    if expected_mime_type in _SUPPORTED_MIME_TYPES and detected != expected_mime_type:
        logger.info(
            "Telegram MIME %s не совпал с сигнатурой %s",
            expected_mime_type,
            detected,
        )

    return ImageInput(data=data, mime_type=detected, label=label)


async def _download_animation(
    bot: Bot,
    file_id: str,
    *,
    file_size: int | None,
    label: str,
) -> tuple[ImageInput, ...]:
    """Скачать ролик и извлечь до четырёх кадров по временной шкале."""
    if file_size is not None and file_size > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    destination = io.BytesIO()
    await bot.download(file_id, destination=destination)
    data = destination.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    detected = _detected_mime_type(data)
    if detected is not None and detected != "image/gif":
        return (ImageInput(data=data, mime_type=detected, label=f"{label}, кадр"),)
    frames = await asyncio.to_thread(extract_animation_frames, data, label=label)
    if not frames:
        raise ValueError("unsupported_image")
    return frames


def _media_message(message: Message, label: str, emoji: str | None = None) -> str:
    """Собрать текст текущего визуального сообщения для модели и истории."""
    if message.caption and message.caption.strip():
        return message.caption.strip()
    if emoji:
        return f"[Пользователь отправил {label}; эмодзи стикера: {emoji}]"
    return f"[Пользователь отправил {label} без подписи]"


def create_media_router(
    response_engine: ResponseEngine,
    bot: Bot,
    rate_limiter: UserRateLimiter | None = None,
    album_delay_seconds: float = 1.2,
) -> Router:
    """Создать роутер поддерживаемых изображений и стикеров."""
    if album_delay_seconds <= 0:
        raise ValueError("album_delay_seconds должен быть больше нуля")

    router = Router(name=__name__)
    limiter = rate_limiter or UserRateLimiter()
    albums: dict[
        tuple[int, str],
        list[tuple[Message, asyncio.Task[ImageInput]]],
    ] = {}
    album_tasks: dict[tuple[int, str], asyncio.Task[None]] = {}

    async def respond_with_images(
        message: Message,
        text: str,
        images: Sequence[ImageInput],
    ) -> None:
        """Передать один визуальный ход общему движку ответа."""
        if message.from_user is None:
            return
        user_id = message.from_user.id
        if not limiter.allow(user_id):
            await message.answer(RATE_LIMIT_REPLY)
            return
        try:
            async with show_typing(message, bot):
                await response_engine.respond_and_deliver(
                    user_id,
                    text,
                    create_reply_delivery(message, bot),
                    images=images,
                )
        except ResponseBusyError:
            await message.answer(BUSY_REPLY)

    async def flush_album(key: tuple[int, str]) -> None:
        """Дождаться конца альбома и отправить все изображения одним ходом."""
        try:
            await asyncio.sleep(album_delay_seconds)
            entries = albums.pop(key, [])
            album_tasks.pop(key, None)
            if not entries:
                return
            results = await asyncio.gather(
                *(download for _, download in entries),
                return_exceptions=True,
            )
            images = tuple(
                result for result in results if isinstance(result, ImageInput)
            )
            representative = next(
                (message for message, _ in entries if message.caption),
                entries[0][0],
            )
            if not images:
                logger.warning("Не удалось скачать ни одного элемента альбома")
                await representative.answer(IMAGE_DOWNLOAD_ERROR_REPLY)
                return
            failed = len(entries) - len(images)
            if failed:
                logger.warning(
                    "Часть Telegram-альбома не обработана: %d из %d",
                    failed,
                    len(entries),
                )
            label = f"альбом из {len(images)} изображений"
            logger.info("Собран Telegram-альбом: images=%d", len(images))
            await respond_with_images(
                representative,
                _media_message(representative, label),
                images,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            albums.pop(key, None)
            album_tasks.pop(key, None)
            logger.exception("Не удалось обработать Telegram-альбом")

    def queue_album(
        message: Message,
        download: asyncio.Task[ImageInput],
        group_id: str,
    ) -> None:
        """Сразу учесть элемент альбома и параллельно скачать его содержимое."""
        key = (message.chat.id, group_id)
        albums.setdefault(key, []).append((message, download))
        previous = album_tasks.get(key)
        if previous is not None:
            previous.cancel()
        album_tasks[key] = asyncio.create_task(flush_album(key))

    async def handle_image(
        message: Message,
        *,
        file_id: str,
        mime_type: str,
        file_size: int | None,
        label: str,
        emoji: str | None = None,
    ) -> None:
        """Скачать изображение и передать его общему движку ответа."""
        if message.from_user is None:
            return
        media_group_id = message.media_group_id
        if media_group_id:
            queue_album(
                message,
                asyncio.create_task(
                    _download_image(
                        bot,
                        file_id,
                        expected_mime_type=mime_type,
                        file_size=file_size,
                        label=label,
                    )
                ),
                media_group_id,
            )
            return
        try:
            image = await _download_image(
                bot,
                file_id,
                expected_mime_type=mime_type,
                file_size=file_size,
                label=label,
            )
        except ValueError as error:
            reply = (
                IMAGE_TOO_LARGE_REPLY
                if str(error) == "image_too_large"
                else UNSUPPORTED_IMAGE_REPLY
            )
            await message.answer(reply)
            return
        except TelegramAPIError, OSError:
            logger.warning("Не удалось скачать изображение из Telegram", exc_info=True)
            await message.answer(IMAGE_DOWNLOAD_ERROR_REPLY)
            return

        await respond_with_images(
            message,
            _media_message(message, label, emoji),
            (image,),
        )

    async def handle_moving_media(
        message: Message,
        *,
        file_id: str,
        file_size: int | None,
        label: str,
        emoji: str | None = None,
    ) -> None:
        """Передать модели последовательность кадров движущегося медиа."""
        try:
            frames = await _download_animation(
                bot,
                file_id,
                file_size=file_size,
                label=label,
            )
        except ValueError as error:
            reply = (
                IMAGE_TOO_LARGE_REPLY
                if str(error) == "image_too_large"
                else UNSUPPORTED_IMAGE_REPLY
            )
            await message.answer(reply)
            return
        except TelegramAPIError, OSError:
            logger.warning("Не удалось скачать анимацию из Telegram", exc_info=True)
            await message.answer(IMAGE_DOWNLOAD_ERROR_REPLY)
            return
        await respond_with_images(
            message,
            _media_message(message, label, emoji),
            frames,
        )

    async def handle_tgs_sticker(message: Message) -> None:
        """Скачать и отрисовать TGS, используя thumbnail только как запасной путь."""
        sticker = message.sticker
        if sticker is None:
            return
        frames: tuple[ImageInput, ...] = ()
        if sticker.file_size is None or sticker.file_size <= MAX_TGS_BYTES:
            destination = io.BytesIO()
            try:
                await bot.download(sticker.file_id, destination=destination)
                frames = await asyncio.to_thread(
                    extract_tgs_frames,
                    destination.getvalue(),
                    label="TGS-анимация стикера",
                )
            except TelegramAPIError, OSError:
                logger.warning("Не удалось скачать TGS-стикер", exc_info=True)
        if frames:
            await respond_with_images(
                message,
                _media_message(message, "анимированный стикер", sticker.emoji),
                frames,
            )
            return
        thumbnail = sticker.thumbnail
        if thumbnail is not None:
            await handle_image(
                message,
                file_id=thumbnail.file_id,
                mime_type="image/jpeg",
                file_size=thumbnail.file_size,
                label="превью TGS-стикера",
                emoji=sticker.emoji,
            )
            return
        await message.answer(UNSUPPORTED_IMAGE_REPLY)

    @router.message(F.photo)
    async def handle_photo(message: Message) -> None:
        """Передать модели фотографию в максимальном доступном размере."""
        if not message.photo:
            return
        photo = message.photo[-1]
        await handle_image(
            message,
            file_id=photo.file_id,
            mime_type="image/jpeg",
            file_size=photo.file_size,
            label="фотографию",
        )

    @router.message(F.document, _has_supported_image_document)
    async def handle_document(message: Message) -> None:
        """Передать модели поддерживаемое изображение-документ."""
        document = message.document
        if document is None:
            return
        mime_type = _document_mime_type(document.mime_type, document.file_name)
        if mime_type is None:
            return
        await handle_image(
            message,
            file_id=document.file_id,
            mime_type=mime_type,
            file_size=document.file_size,
            label="изображение-файл",
        )

    @router.message(F.sticker)
    async def handle_sticker(message: Message) -> None:
        """Передать статический стикер или превью анимации модели."""
        sticker = message.sticker
        if sticker is None:
            return
        if not sticker.is_animated and not sticker.is_video:
            await handle_image(
                message,
                file_id=sticker.file_id,
                mime_type="image/webp",
                file_size=sticker.file_size,
                label="статический стикер",
                emoji=sticker.emoji,
            )
            return
        if sticker.is_video:
            await handle_moving_media(
                message,
                file_id=sticker.file_id,
                file_size=sticker.file_size,
                label="видеостикер",
                emoji=sticker.emoji,
            )
            return
        await handle_tgs_sticker(message)

    @router.message(F.animation)
    async def handle_animation(message: Message) -> None:
        """Передать модели кадр Telegram GIF-анимации."""
        animation = message.animation
        if animation is None:
            return
        await handle_moving_media(
            message,
            file_id=animation.file_id,
            file_size=animation.file_size,
            label="GIF-анимация",
        )

    @router.message(F.document, _has_supported_animation_document)
    async def handle_animation_document(message: Message) -> None:
        """Передать модели кадр анимации, отправленной обычным файлом."""
        document = message.document
        if document is None:
            return
        await handle_moving_media(
            message,
            file_id=document.file_id,
            file_size=document.file_size,
            label="видео или анимация, отправленная файлом",
        )

    @router.message(F.video)
    async def handle_video(message: Message) -> None:
        """Передать модели несколько кадров обычного Telegram-видео."""
        video = message.video
        if video is None:
            return
        await handle_moving_media(
            message,
            file_id=video.file_id,
            file_size=video.file_size,
            label="видео",
        )

    return router
