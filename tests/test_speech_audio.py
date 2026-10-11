"""Сохранение длительности PCM при ресемплинге и ограничение записи."""

import io
import wave

import pytest

from protogen_delta.services.speech_audio import prepare_speech_audio


def _silence(sample_rate: int, channels: int, seconds: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(sample_rate * channels * seconds * 2))
    return buffer.getvalue()


@pytest.mark.parametrize(
    "sample_rate,channels", [(48000, 2), (44100, 1), (16000, 1), (8000, 1)]
)
def test_resampling_preserves_the_last_samples(sample_rate: int, channels: int) -> None:
    data = prepare_speech_audio(_silence(sample_rate, channels, 1))

    with wave.open(io.BytesIO(data), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
            1,
            2,
            16000,
        )
        assert wav.getnframes() == 16000
        assert len(wav.readframes(16000)) == 16000 * 2


@pytest.mark.parametrize("seconds", [600, 601])
def test_resampling_keeps_the_ten_minute_limit(seconds: int) -> None:
    data = prepare_speech_audio(_silence(8000, 1, seconds))

    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnframes() == 600 * 16000
        assert len(wav.readframes(600 * 16000 + 1)) == 600 * 16000 * 2
