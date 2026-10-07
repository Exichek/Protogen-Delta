"""Обработчик текстовых пользовательских документов."""

import asyncio
import io
import logging
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.delivery import create_reply_delivery, show_typing
from protogen_delta.handlers.text import BUSY_REPLY, RATE_LIMIT_REPLY
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.documents import (
    MAX_DOCUMENT_BYTES,
    DocumentReadError,
    DocumentTooLargeError,
    UnsupportedDocumentError,
    extract_document,
)
from protogen_delta.services.music import is_audio_file
from protogen_delta.services.response_engine import (
    ResponseBusyError,
    ResponseEngine,
    personal_fact_options,
)
from protogen_delta.services.stickers import ContextualStickerService

logger = logging.getLogger(__name__)

DOCUMENT_TOO_LARGE_REPLY = "Документ слишком большой: сейчас могу скачать до 20 МБ."
DOCUMENT_DOWNLOAD_ERROR_REPLY = (
    "Не смог скачать документ из Telegram. Попробуй ещё раз."
)
DOCUMENT_READ_ERROR_REPLY = (
    "Не смог прочитать этот документ: возможно, он повреждён или защищён паролем."
)
UNSUPPORTED_DOCUMENT_REPLY = "Этот формат пока не разбираю. Пришли TXT, Markdown, код, JSON, CSV, PDF, DOCX или XLSX."


def _safe_file_name(file_name: str | None) -> str:
    """Оставить для промпта только короткое имя без пользовательского пути."""
    name = Path(file_name or "document").name.strip() or "document"
    return name[:160]


def create_document_router(
    response_engine: ResponseEngine,
    bot: Bot,
    rate_limiter: UserRateLimiter | None = None,
    sticker_service: ContextualStickerService | None = None,
    *,
    ocr_enabled: bool = False,
) -> Router:
    """Создать роутер чтения поддерживаемых документов."""
    router = Router(name=__name__)
    limiter = rate_limiter or UserRateLimiter()

    @router.message(
        F.document,
        lambda message: not is_audio_file(
            message.document.mime_type, message.document.file_name
        ),
    )
    async def handle_document(message: Message) -> None:
        """Скачать документ, извлечь текст и передать его движку ответа."""
        document = message.document
        if document is None or message.from_user is None:
            return
        user_id = message.from_user.id
        if not limiter.allow(user_id):
            await message.answer(RATE_LIMIT_REPLY)
            return
        if document.file_size is not None and document.file_size > MAX_DOCUMENT_BYTES:
            await message.answer(DOCUMENT_TOO_LARGE_REPLY)
            return

        destination = io.BytesIO()
        try:
            await bot.download(document.file_id, destination=destination)
        except TelegramAPIError, OSError:
            logger.warning("Не удалось скачать документ из Telegram", exc_info=True)
            await message.answer(DOCUMENT_DOWNLOAD_ERROR_REPLY)
            return

        file_name = _safe_file_name(document.file_name)
        try:
            arguments = {"ocr_enabled": True} if ocr_enabled else {}
            extracted = await asyncio.to_thread(
                extract_document,
                destination.getvalue(),
                file_name,
                document.mime_type,
                **arguments,
            )
        except DocumentTooLargeError:
            await message.answer(DOCUMENT_TOO_LARGE_REPLY)
            return
        except UnsupportedDocumentError:
            await message.answer(UNSUPPORTED_DOCUMENT_REPLY)
            return
        except DocumentReadError:
            await message.answer(DOCUMENT_READ_ERROR_REPLY)
            return

        request = (message.caption or "").strip()
        if not request:
            request = (
                f'[Пользователь отправил документ "{file_name}" без вопроса. '
                "Коротко расскажи, что в нём.]"
            )
        attachment_text = extracted.text
        if extracted.truncated:
            attachment_text += (
                "\n\n[Документ обработан частично: достигнут лимит страниц, "
                "времени, OCR/vision или 60000 символов. Не считай пропущенное прочитанным.]"
            )

        try:
            async with show_typing(message, bot):
                images = tuple(
                    ImageInput(
                        data=image.data,
                        mime_type=image.mime_type,
                        label=image.label,
                    )
                    for image in extracted.images
                )
                await response_engine.respond_and_deliver(
                    user_id,
                    request,
                    create_reply_delivery(
                        message,
                        bot,
                        sticker_service,
                        user_id=user_id,
                        context_tags=("document",),
                    ),
                    attachment_text=attachment_text,
                    **personal_fact_options(message.chat.type),
                    attachment_name=f"{file_name} ({extracted.kind})",
                    images=images,
                )
        except ResponseBusyError:
            await message.answer(BUSY_REPLY)

    return router
