"""Метаданные, ограниченное распознавание и музыка, отправленная файлом."""

import asyncio
import io
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Update
from test_audio_understanding import wav_data
from test_voice_handler import _message

import protogen_delta.services.audio_pipeline as voice_module
import protogen_delta.services.music as module
from protogen_delta.handlers.documents import create_document_router
from protogen_delta.handlers.voice import create_voice_router
from protogen_delta.services.audio_analysis import AudioAnalysisError
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.music import (
    MusicRecognitionError,
    MusicRecognitionService,
    audio_tags,
    is_audio_file,
)
from protogen_delta.services.response_engine import ResponseEngine
from protogen_delta.services.speech import SpeechRecognitionError, SpeechTranscriber


def test_music_file_detection_and_metadata_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert is_audio_file("audio/mpeg", None)
    assert is_audio_file("application/octet-stream", "MUSIC.FLAC")
    assert not is_audio_file(None, "a.docx")
    assert not is_audio_file(None, None)
    assert audio_tags(b"") == audio_tags(b"bad") == {}
    assert audio_tags(b"x" * (20 * 1024 * 1024 + 1)) == {}
    assert audio_tags(wav_data()) == {}
    container = MagicMock()
    container.__enter__.return_value = SimpleNamespace(
        metadata={
            "TITLE": "Track",
            "ARTIST": " Artist  Name ",
            "ALBUM": "x" * 400,
            "GENRE": " ",
        },
        streams=SimpleNamespace(audio=[SimpleNamespace(metadata={"DATE": "2020"})]),
    )
    monkeypatch.setattr(module.av, "open", Mock(return_value=container))
    result = audio_tags(b"audio")
    assert result == {
        "title": "Track",
        "artist": "Artist Name",
        "album": "x" * 200,
        "date": "2020",
    }


class _Response:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.status = status
        self.body = body
        self.content = self

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        # Не полагаться на одно read(): JSON может прийти несколькими частями.
        for index in range(0, len(self.body), 7):
            yield self.body[index : index + 7]  # noqa: E203


def test_recognizer_uploads_only_twelve_seconds_and_parses_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        return_value=_Response(
            json.dumps(
                {
                    "status": "success",
                    "result": {
                        "title": "Track",
                        "artist": "Artist",
                        "album": "Album",
                        "release_date": "2020",
                        "ignored": "secret",
                    },
                }
            ).encode()
        )
    )
    session = Mock()
    session.__aenter__ = AsyncMock(return_value=SimpleNamespace(post=post))
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module.aiohttp, "ClientSession", Mock(return_value=session))
    service = MusicRecognitionService("test-token", native_work=BlockingWorkPool(2))
    report = asyncio.run(service.recognize(wav_data(15)))
    assert report is not None
    result = json.loads(report)
    assert result == {
        "title": "Track",
        "artist": "Artist",
        "album": "Album",
        "release_date": "2020",
    }
    assert post.call_args.args == ("https://api.audd.io/",)
    fields = post.call_args.kwargs["data"]._fields
    clip = fields[1][2]
    import wave

    with wave.open(io.BytesIO(clip)) as audio:
        assert audio.getnframes() / audio.getframerate() <= 12
    session.__aexit__.assert_awaited_once()
    assert b"Telegram" not in clip


@pytest.mark.parametrize(
    "body,status,expected",
    [
        (b'{"status":"success","result":null}', 200, None),
        (b"bad", 200, "error"),
        (b"[]", 200, "error"),
        (b'{"status":"error"}', 200, "error"),
        (b'{"status":"success","result":{}}', 200, "error"),
        (b"bad", 500, "error"),
        (b"x" * 131073, 200, "error"),
    ],
    ids=[
        "no-match",
        "invalid-json",
        "wrong-type",
        "provider-error",
        "incomplete",
        "http-error",
        "oversize",
    ],
)
def test_recognizer_handles_no_match_and_errors(
    monkeypatch: pytest.MonkeyPatch, body: bytes, status: int, expected: str | None
) -> None:
    session = Mock()
    session.__aenter__ = AsyncMock(
        return_value=SimpleNamespace(post=Mock(return_value=_Response(body, status)))
    )
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module.aiohttp, "ClientSession", Mock(return_value=session))
    service = MusicRecognitionService("test-token", native_work=BlockingWorkPool(2))
    if expected == "error":
        with pytest.raises(MusicRecognitionError):
            asyncio.run(service.recognize(wav_data()))
    else:
        assert asyncio.run(service.recognize(wav_data())) is None


def test_recognizer_network_and_invalid_audio_fail_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MusicRecognitionService("test-token", native_work=BlockingWorkPool(2))
    with pytest.raises(MusicRecognitionError):
        asyncio.run(service.recognize(b"bad"))
    session = Mock()
    session.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("network"))
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module.aiohttp, "ClientSession", Mock(return_value=session))
    with pytest.raises(MusicRecognitionError):
        asyncio.run(service.recognize(wav_data()))


def test_audio_metadata_and_recognition_are_data_and_preserved_in_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = AsyncMock(spec=ResponseEngine)
        bot = AsyncMock(spec=Bot)

        async def download(file_id: str, *, destination: io.BytesIO) -> None:
            destination.write(b"audio")

        bot.download.side_effect = download
        speech = AsyncMock(spec=SpeechTranscriber)
        speech.transcribe.side_effect = SpeechRecognitionError("none")
        recognizer = AsyncMock(spec=MusicRecognitionService)
        recognizer.recognize.return_value = '{"title":"Recognized","artist":"Someone"}'
        monkeypatch.setattr(
            voice_module, "analyze_audio", Mock(side_effect=AudioAnalysisError("bad"))
        )
        monkeypatch.setattr(voice_module, "audio_tags", lambda _: {"album": "Album"})
        message, answer = _message(
            audio=SimpleNamespace(
                file_id="audio",
                file_size=10,
                duration=30,
                title="Track",
                performer="Artist",
                file_name="a.mp3",
            )
        )
        router = create_voice_router(
            engine,
            bot,
            speech,
            music_recognition=recognizer,
            native_work=BlockingWorkPool(2),
        )
        await router.message.handlers[0].callback(message)
        call = engine.respond_and_deliver.await_args
        assert "Track" in call.args[1] and "Recognized" in call.args[1]
        assert "данные, не подтверждённое" in call.kwargs["model_message_override"]
        assert "12 секундам" in call.kwargs["model_message_override"]
        assert "не изображай прослушивание" in call.kwargs["trusted_input_context"]
        answer.assert_not_awaited()
        recognizer.recognize.side_effect = MusicRecognitionError("offline")
        other = create_voice_router(
            engine,
            bot,
            speech,
            music_recognition=recognizer,
            native_work=BlockingWorkPool(2),
        )
        await other.message.handlers[0].callback(message)
        assert (
            "Recognized"
            not in engine.respond_and_deliver.await_args.kwargs[
                "model_message_override"
            ]
        )

    asyncio.run(scenario())


def test_audio_document_reaches_voice_instead_of_document_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        engine = AsyncMock(spec=ResponseEngine)
        bot = AsyncMock(spec=Bot)
        bot.id = 123

        async def download(file_id: str, *, destination: io.BytesIO) -> None:
            destination.write(b"audio")

        bot.download.side_effect = download
        speech = AsyncMock(spec=SpeechTranscriber)
        speech.transcribe.side_effect = SpeechRecognitionError("none")
        monkeypatch.setattr(
            voice_module, "audio_tags", lambda _: {"title": "Music file"}
        )
        monkeypatch.setattr(
            voice_module, "analyze_audio", Mock(side_effect=AudioAnalysisError("bad"))
        )
        dispatcher = Dispatcher()
        dispatcher.include_router(create_document_router(engine, bot))
        dispatcher.include_router(
            create_voice_router(engine, bot, speech, native_work=BlockingWorkPool(2))
        )
        update = Update.model_validate(
            {
                "update_id": 1,
                "message": {
                    "message_id": 1,
                    "date": 1700000000,
                    "chat": {"id": 42, "type": "private"},
                    "from": {"id": 42, "is_bot": False, "first_name": "Test"},
                    "caption": "Что думаешь о треке?",
                    "document": {
                        "file_id": "file",
                        "file_unique_id": "unique",
                        "file_name": "Music.FLAC",
                        "mime_type": "application/octet-stream",
                        "file_size": 10,
                    },
                },
            }
        )
        await dispatcher.feed_update(bot, update)
        call = engine.respond_and_deliver.await_args
        assert call.args[0] == 42
        assert "Что думаешь о треке?" in call.args[1]
        assert "Music file" in call.kwargs["model_message_override"]

    asyncio.run(scenario())
