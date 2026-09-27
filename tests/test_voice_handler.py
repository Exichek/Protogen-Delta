"""Тесты Telegram-обработчика голосовых и аудиофайлов."""

import asyncio
import io
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from aiogram import Bot
from aiogram.types import Message

from protogen_delta.handlers.text import BUSY_REPLY
from protogen_delta.handlers.voice import (
    AUDIO_DOWNLOAD_ERROR_REPLY,
    AUDIO_RECOGNITION_ERROR_REPLY,
    AUDIO_TOO_LARGE_REPLY,
    create_voice_router,
)
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine
from protogen_delta.services.speech import (
    MAX_AUDIO_BYTES,
    SpeechRecognitionError,
    SpeechTranscriber,
    Transcript,
)

TEST_USER_ID = 123456


def _message(
    *,
    voice: Any = None,
    audio: Any = None,
    caption: str | None = None,
) -> tuple[Message, AsyncMock]:
    """Создать входящее аудиосообщение."""
    message = Mock(spec=Message)
    message.voice = voice
    message.audio = audio
    message.caption = caption
    message.from_user = SimpleNamespace(id=TEST_USER_ID)
    message.chat = SimpleNamespace(id=777)
    message.answer = AsyncMock()
    return cast(Message, message), message.answer


def _router(
    transcript: Transcript = Transcript("Привет, как дела?", "ru"),
) -> tuple[Any, AsyncMock, AsyncMock, AsyncMock]:
    """Создать роутер с подменёнными сетью, Whisper и движком."""
    engine = AsyncMock(spec=ResponseEngine)

    async def respond(
        user_id: int,
        text: str,
        deliver: Any,
        **kwargs: Any,
    ) -> None:
        await deliver("Нормально.")

    engine.respond_and_deliver.side_effect = respond
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> io.BytesIO:
        destination.write(b"ogg audio")
        return destination

    bot.download.side_effect = download
    transcriber = AsyncMock(spec=SpeechTranscriber)
    transcriber.transcribe.return_value = transcript
    router = create_voice_router(
        cast(ResponseEngine, engine),
        cast(Bot, bot),
        cast(SpeechTranscriber, transcriber),
    )
    return router, engine, bot, transcriber


def test_voice_is_transcribed_and_sent_as_user_message() -> None:
    """Голосовое должно отвечать на речь и сохранять компактную расшифровку."""
    router, engine, bot, transcriber = _router()
    voice = SimpleNamespace(file_id="voice-id", file_size=100, duration=4)
    message, answer = _message(voice=voice)

    asyncio.run(router.message.handlers[0].callback(message))

    bot.download.assert_awaited_once()
    transcriber.transcribe.assert_awaited_once_with(b"ogg audio")
    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert "Привет, как дела?" in call.args[1]
    assert call.kwargs["model_message_override"] == "Привет, как дела?"
    answer.assert_awaited_once_with("Нормально.")


def test_audio_caption_and_truncation_are_preserved() -> None:
    """Подпись аудиофайла должна дополнять расшифровку и отметку об обрезании."""
    router, engine, _, _ = _router(Transcript("слова песни", "ru", truncated=True))
    audio = SimpleNamespace(file_id="audio-id", file_size=100, duration=30)
    message, _ = _message(audio=audio, caption="Что тут по смыслу?")

    asyncio.run(router.message.handlers[0].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert call.args[1].startswith("Что тут по смыслу?")
    assert "Расшифровка речи" in call.kwargs["model_message_override"]
    assert "обрезана" in call.kwargs["model_message_override"]


def test_large_or_long_audio_is_rejected() -> None:
    """Лимиты должны проверяться до скачивания и inference."""
    for media in (
        SimpleNamespace(file_id="large", file_size=MAX_AUDIO_BYTES + 1, duration=1),
        SimpleNamespace(file_id="long", file_size=1, duration=601),
    ):
        router, engine, bot, transcriber = _router()
        message, answer = _message(voice=media)
        asyncio.run(router.message.handlers[0].callback(message))
        answer.assert_awaited_once_with(AUDIO_TOO_LARGE_REPLY)
        bot.download.assert_not_awaited()
        transcriber.transcribe.assert_not_awaited()
        engine.respond_and_deliver.assert_not_awaited()


def test_download_recognition_and_busy_errors_are_handled() -> None:
    """Сетевые, ASR и конкурентные ошибки должны получать стабильные ответы."""
    voice = SimpleNamespace(file_id="voice-id", file_size=100, duration=4)

    router, _, bot, _ = _router()
    bot.download.side_effect = OSError("network")
    message, answer = _message(voice=voice)
    asyncio.run(router.message.handlers[0].callback(message))
    answer.assert_awaited_once_with(AUDIO_DOWNLOAD_ERROR_REPLY)

    router, _, _, transcriber = _router()
    transcriber.transcribe.side_effect = SpeechRecognitionError("silence")
    message, answer = _message(voice=voice)
    asyncio.run(router.message.handlers[0].callback(message))
    answer.assert_awaited_once_with(AUDIO_RECOGNITION_ERROR_REPLY)

    router, engine, _, _ = _router()
    engine.respond_and_deliver.side_effect = ResponseBusyError
    message, answer = _message(voice=voice)
    asyncio.run(router.message.handlers[0].callback(message))
    answer.assert_awaited_once_with(BUSY_REPLY)
