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
from protogen_delta.services.audio_analysis import AudioAnalysis, AudioAnalysisError
from protogen_delta.services.audio_understanding import AudioUnderstandingService
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine
from protogen_delta.services.speech import (
    MAX_AUDIO_BYTES,
    SpeechRecognitionError,
    SpeechTranscriber,
    Transcript,
)

TEST_USER_ID = 123456


def test_uncertain_words_are_marked_in_context_and_saved_history() -> None:
    router, engine, _, _ = _router(Transcript("Моя любимая", "ru", uncertain=True))
    message, _ = _message(
        voice=SimpleNamespace(file_id="voice", file_size=10, duration=2)
    )
    asyncio.run(router.message.handlers[0].callback(message))
    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert "Неуверенная расшифровка" in call.args[1]
    assert "неуверенное распознавание" in call.kwargs["trusted_input_context"]
    assert "Не достраивай" in call.kwargs["trusted_input_context"]
    assert call.kwargs["model_message_override"] == "Моя любимая"


def test_audio_model_report_is_used_and_failure_falls_back(monkeypatch: Any) -> None:
    import protogen_delta.services.audio_pipeline as voice_module

    monkeypatch.setattr(
        voice_module,
        "analyze_audio",
        Mock(side_effect=AudioAnalysisError("bad")),
    )
    engine = AsyncMock(spec=ResponseEngine)
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> None:
        destination.write(b"audio")

    bot.download.side_effect = download
    transcriber = AsyncMock(spec=SpeechTranscriber)
    transcriber.transcribe.side_effect = SpeechRecognitionError("no speech")
    model = AsyncMock(spec=AudioUnderstandingService)
    model.analyze.return_value = "Вероятно, рок с гитарой"

    def router() -> Any:
        return create_voice_router(
            cast(ResponseEngine, engine),
            cast(Bot, bot),
            cast(SpeechTranscriber, transcriber),
            audio_understanding=cast(AudioUnderstandingService, model),
            native_work=BlockingWorkPool(2),
        )

    audio = SimpleNamespace(file_id="audio", file_size=20, duration=40)
    message, answer = _message(audio=audio)
    asyncio.run(router().message.handlers[0].callback(message))
    arguments = engine.respond_and_deliver.await_args.kwargs
    assert "рок" in arguments["model_message_override"]
    assert "60 секундам" in arguments["trusted_input_context"]
    model.analyze.side_effect = AudioAnalysisError("not available")
    asyncio.run(router().message.handlers[0].callback(message))
    call = answer.await_args
    assert call is not None
    assert AUDIO_RECOGNITION_ERROR_REPLY in call.args[0]


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
    message.chat = SimpleNamespace(id=777, type="private")
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
        native_work=BlockingWorkPool(2),
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
    assert "действительно пришло как голосовое" in call.kwargs["trusted_input_context"]
    assert (
        "Не утверждай, что голосового не было" in call.kwargs["trusted_input_context"]
    )
    answer.assert_awaited_once_with("Нормально.")


def test_audio_caption_and_truncation_are_preserved(monkeypatch: Any) -> None:
    """Подпись аудиофайла должна дополнять расшифровку и отметку об обрезании."""
    router, engine, _, _ = _router(Transcript("слова песни", "ru", truncated=True))
    monkeypatch.setattr(
        "protogen_delta.services.audio_pipeline.analyze_audio",
        lambda data: AudioAnalysis(
            30,
            48_000,
            2,
            -14,
            -1,
            2,
            8,
            1800,
        ),
    )
    audio = SimpleNamespace(file_id="audio-id", file_size=100, duration=30)
    message, _ = _message(audio=audio, caption="Что тут по смыслу?")

    asyncio.run(router.message.handlers[0].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert call.args[1].startswith("Что тут по смыслу?")
    assert "Расшифровка речи" in call.kwargs["model_message_override"]
    assert "обрезана" in call.kwargs["model_message_override"]
    assert "48000 Гц" in call.kwargs["model_message_override"]


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


def test_audio_without_sender_is_ignored() -> None:
    router, engine, bot, transcriber = _router()
    audio = SimpleNamespace(file_id="audio-id", file_size=100, duration=30)
    message, answer = _message(audio=audio)
    message.from_user = None

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_not_awaited()
    bot.download.assert_not_awaited()
    transcriber.transcribe.assert_not_awaited()
    engine.respond_and_deliver.assert_not_awaited()


def test_music_without_recognized_speech_uses_signal_analysis(
    monkeypatch: Any,
) -> None:
    """Музыка без расшифровки должна получить ответ по измеримым метрикам."""
    router, engine, _, transcriber = _router()
    transcriber.transcribe.side_effect = SpeechRecognitionError("no speech")
    monkeypatch.setattr(
        "protogen_delta.services.audio_pipeline.analyze_audio",
        lambda data: AudioAnalysis(
            duration_seconds=30,
            source_sample_rate=48_000,
            channels=2,
            loudness_dbfs=-14,
            peak_dbfs=-1,
            silence_percent=2,
            dynamic_range_db=8,
            spectral_centroid_hz=1800,
        ),
    )
    audio = SimpleNamespace(file_id="music", file_size=100, duration=30)
    message, answer = _message(audio=audio, caption="Как звучит?")

    asyncio.run(router.message.handlers[0].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert "48000 Гц" in call.kwargs["model_message_override"]
    assert "не выдумывай жанр" in call.kwargs["trusted_input_context"]
    answer.assert_awaited_once_with("Нормально.")


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
