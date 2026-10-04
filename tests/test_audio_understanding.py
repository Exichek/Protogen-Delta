"""Фрагмент аудио и отдельный ограниченный API-запрос."""

import asyncio
import io
import wave
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

import protogen_delta.services.audio_understanding as module
from protogen_delta.services.audio_analysis import AudioAnalysisError
from protogen_delta.services.audio_understanding import (
    AudioUnderstandingService,
    audio_excerpt,
)


def wav_data(seconds: int = 2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x01\x01\x00" * 8000 * seconds)
    return output.getvalue()


def test_excerpt_normalizes_and_limits_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "MAX_SEMANTIC_SECONDS", 1)
    with wave.open(io.BytesIO(audio_excerpt(wav_data())), "rb") as result:
        assert result.getnchannels() == 1
        assert result.getframerate() == 16000
        assert result.getnframes() <= 16000


@pytest.mark.parametrize("data", [b"", b"broken"])
def test_excerpt_rejects_unreadable_audio(data: bytes) -> None:
    with pytest.raises(AudioAnalysisError):
        audio_excerpt(data)


def test_model_receives_only_excerpt_and_closes_stream() -> None:
    stream = AsyncMock()
    stream.__aenter__.return_value = stream
    stream.__aiter__.return_value = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Музыка"))],
            usage=None,
        ),
        SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=12)),
    ]
    client = AsyncMock()
    client.chat.completions.create.return_value = stream
    service = AudioUnderstandingService(cast(AsyncOpenAI, client), "audio-model")
    assert asyncio.run(service.analyze(wav_data())) == "Музыка"
    arguments = client.chat.completions.create.await_args.kwargs
    assert arguments["model"] == "audio-model"
    assert arguments["max_tokens"] == 600
    content = arguments["messages"][0]["content"]
    assert content[1]["input_audio"]["format"] == "wav"
    assert content[1]["input_audio"]["data"].startswith("data:;base64,")
    stream.__aexit__.assert_awaited_once()
    asyncio.run(service.close())
    client.close.assert_awaited_once()


def test_model_empty_response_and_timeout_are_readable_errors() -> None:
    stream = AsyncMock()
    stream.__aenter__.return_value = stream
    stream.__aiter__.return_value = []
    client = AsyncMock()
    client.chat.completions.create.return_value = stream
    service = AudioUnderstandingService(cast(AsyncOpenAI, client), "audio-model")
    with pytest.raises(AudioAnalysisError, match="описание"):
        asyncio.run(service.analyze(wav_data()))
    client.chat.completions.create.side_effect = TimeoutError()
    with pytest.raises(AudioAnalysisError, match="недоступна"):
        asyncio.run(service.analyze(wav_data()))


def test_model_report_is_bounded() -> None:
    stream = AsyncMock()
    stream.__aenter__.return_value = stream
    stream.__aiter__.return_value = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="x" * 5000))],
            usage=None,
        )
    ]
    client = AsyncMock()
    client.chat.completions.create.return_value = stream
    report = asyncio.run(
        AudioUnderstandingService(cast(AsyncOpenAI, client), "model").analyze(
            wav_data()
        )
    )
    assert len(report) == module.MAX_REPORT_CHARS
