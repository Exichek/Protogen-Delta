"""Обработчик голосовых сообщений и аудиофайлов с речью."""

import asyncio
import io
import logging

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.delivery import create_reply_delivery, show_typing
from protogen_delta.handlers.text import BUSY_REPLY, RATE_LIMIT_REPLY
from protogen_delta.services.audio_analysis import AudioAnalysisError, analyze_audio
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine
from protogen_delta.services.speech import (
    MAX_AUDIO_BYTES,
    MAX_AUDIO_DURATION_SECONDS,
    SpeechRecognitionError,
    SpeechTranscriber,
)
from protogen_delta.services.stickers import ContextualStickerService

logger = logging.getLogger(__name__)

AUDIO_TOO_LARGE_REPLY = "Аудио слишком большое: максимум 20 МБ и 10 минут."
AUDIO_DOWNLOAD_ERROR_REPLY = "Не смог скачать аудио из Telegram. Попробуй ещё раз."
AUDIO_RECOGNITION_ERROR_REPLY = "Не смог разобрать речь в этом аудио."


def create_voice_router(
    response_engine: ResponseEngine,
    bot: Bot,
    transcriber: SpeechTranscriber,
    rate_limiter: UserRateLimiter | None = None,
    sticker_service: ContextualStickerService | None = None,
) -> Router:
    """Создать роутер распознавания голосовых и обычного аудио."""
    router = Router(name=__name__)
    limiter = rate_limiter or UserRateLimiter()

    @router.message(F.voice | F.audio)
    async def handle_voice(message: Message) -> None:
        """Скачать аудио, распознать речь и ответить на её содержание."""
        media = message.voice or message.audio
        if media is None or message.from_user is None:
            return
        user_id = message.from_user.id
        if not limiter.allow(user_id):
            await message.answer(RATE_LIMIT_REPLY)
            return
        if (
            media.file_size is not None
            and media.file_size > MAX_AUDIO_BYTES
            or media.duration > MAX_AUDIO_DURATION_SECONDS
        ):
            await message.answer(AUDIO_TOO_LARGE_REPLY)
            return

        destination = io.BytesIO()
        try:
            await bot.download(media.file_id, destination=destination)
        except TelegramAPIError, OSError:
            logger.warning("Не удалось скачать аудио из Telegram", exc_info=True)
            await message.answer(AUDIO_DOWNLOAD_ERROR_REPLY)
            return

        async with show_typing(message, bot, initial_delay_seconds=1.2):
            analysis = None
            if message.audio is not None:
                try:
                    analysis = await asyncio.to_thread(
                        analyze_audio,
                        destination.getvalue(),
                    )
                except AudioAnalysisError:
                    logger.warning(
                        "Не удалось измерить характеристики аудио", exc_info=True
                    )
            try:
                transcript = await transcriber.transcribe(destination.getvalue())
            except SpeechRecognitionError:
                transcript = None
                logger.info("Whisper не нашёл разборчивую речь в аудио")
                if analysis is None or message.audio is None:
                    await message.answer(AUDIO_RECOGNITION_ERROR_REPLY)
                    return

            caption = (message.caption or "").strip()
            media_kind = (
                "голосовое сообщение" if message.voice is not None else "аудиофайл"
            )
            if transcript is not None:
                history_text = f"[Расшифровка {media_kind}: {transcript.text[:2000]}]"
                if caption:
                    history_text = f"{caption}\n\n{history_text}"
                model_message = transcript.text
                if caption:
                    model_message = f"{caption}\n\nРасшифровка речи:\n{transcript.text}"
                if transcript.truncated:
                    model_message += "\n\n[Расшифровка обрезана после 30000 символов.]"
                if analysis is not None:
                    model_message += (
                        "\n\nИзмеримые характеристики аудиосигнала:\n"
                        + analysis.summary()
                    )
                trusted_context = (
                    f"Текущее сообщение действительно пришло как {media_kind}. "
                    "Приложение уже распознало речь и передало тебе точную "
                    "расшифровку. Отвечай на её смысл как на сообщение "
                    "пользователя. Не утверждай, что голосового не было, что "
                    "аудио тебе недоступно или что пользователь прислал текст."
                )
            else:
                assert analysis is not None
                metrics = analysis.summary()
                history_text = caption or "[Пользователь отправил аудиофайл]"
                model_message = (
                    (caption + "\n\n" if caption else "Прокомментируй аудиофайл.\n\n")
                    + "Измеримые характеристики аудиосигнала:\n"
                    + metrics
                )
                trusted_context = (
                    "Текущее сообщение действительно содержит аудиофайл без "
                    "разборчивой речи. Приложение измерило только перечисленные "
                    "характеристики сигнала. Можешь объяснить громкость, динамику, "
                    "паузы и спектральный баланс, но не выдумывай жанр, инструменты, "
                    "мелодию, вокал или настроение, которых эти метрики не подтверждают."
                )

            try:
                await response_engine.respond_and_deliver(
                    user_id,
                    history_text,
                    create_reply_delivery(
                        message,
                        bot,
                        sticker_service,
                        user_id=user_id,
                        context_tags=("voice",),
                    ),
                    model_message_override=model_message,
                    trusted_input_context=trusted_context,
                )
            except ResponseBusyError:
                await message.answer(BUSY_REPLY)

    return router
