"""Фрагмент аудио и отдельный ограниченный API-запрос."""

import asyncio
import base64
import io
import json
import wave
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import httpx2 as httpx
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


@pytest.mark.parametrize("input_data_url", [False, True])
def test_model_receives_only_excerpt_and_closes_stream(input_data_url: bool) -> None:
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
    service = AudioUnderstandingService(
        cast(AsyncOpenAI, client), "audio-model", input_data_url=input_data_url
    )
    assert asyncio.run(service.analyze(wav_data())) == "Музыка"
    arguments = client.chat.completions.create.await_args.kwargs
    assert arguments["model"] == "audio-model"
    assert arguments["max_tokens"] == 600
    content = arguments["messages"][0]["content"]
    assert content[1]["input_audio"]["format"] == "wav"
    encoded = content[1]["input_audio"]["data"]
    if input_data_url:
        assert encoded.startswith("data:;base64,")
        encoded = encoded.removeprefix("data:;base64,")
    else:
        assert not encoded.startswith("data:")
    decoded = base64.b64decode(encoded, validate=True)
    with wave.open(io.BytesIO(decoded), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getframerate() == 16000
        assert wav.getnframes() <= 60 * 16000
    assert "messages" in arguments and len(arguments["messages"]) == 1
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


@pytest.mark.parametrize("status", [200, 402])
def test_free_openrouter_request_and_credit_error(status: int) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status != 200:
            return httpx.Response(
                status,
                json={"error": {"message": "Audio requires a balance", "code": 402}},
            )
        chunk = {"choices": [{"index": 0, "delta": {"content": "Музыка"}}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )

    async def scenario() -> None:
        client = AsyncOpenAI(
            api_key="test-audio-key",
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        service = AudioUnderstandingService(client, "nvidia/audio-test:free")
        try:
            if status == 200:
                assert await service.analyze(wav_data()) == "Музыка"
            else:
                with pytest.raises(AudioAnalysisError, match="недоступна"):
                    await service.analyze(wav_data())
        finally:
            await service.close()

    asyncio.run(scenario())
    assert len(requests) == 1
    assert requests[0].url.path == "/api/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["provider"]["max_price"] == {"prompt": 0, "completion": 0}
    assert payload["reasoning"] == {"enabled": False}
    encoded = payload["messages"][0]["content"][1]["input_audio"]["data"]
    assert base64.b64decode(encoded, validate=True).startswith(b"RIFF")
