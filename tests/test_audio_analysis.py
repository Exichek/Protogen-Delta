"""Тесты объективного анализа аудиосигнала."""

import io
import wave

import av
import numpy as np
import pytest

import protogen_delta.services.audio_analysis as audio_module
from protogen_delta.services.audio_analysis import (
    AudioAnalysisError,
    _spectral_centroid,
    analyze_audio,
)


def _wave_with_tone() -> bytes:
    rate = 16_000
    time = np.arange(rate, dtype=np.float64) / rate
    signal = (0.4 * np.sin(2 * np.pi * 440 * time) * 32767).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(signal.tobytes())
    return output.getvalue()


def test_analyze_audio_measures_tone_without_guessing_content() -> None:
    result = analyze_audio(_wave_with_tone())

    assert result.duration_seconds == pytest.approx(1.0, abs=0.05)
    assert result.source_sample_rate == 16_000
    assert result.channels == 1
    assert -12 < result.loudness_dbfs < -9
    assert result.spectral_centroid_hz == pytest.approx(440, abs=35)
    assert "спектральный центр" in result.summary()


def test_analyze_audio_rejects_empty_and_invalid_files() -> None:
    with pytest.raises(AudioAnalysisError, match="пуст"):
        analyze_audio(b"")
    with pytest.raises(AudioAnalysisError, match="декодировать"):
        analyze_audio(b"not audio")


def test_analyze_audio_caps_decoded_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_module, "MAX_ANALYSIS_SECONDS", 0.1)

    result = analyze_audio(_wave_with_tone())

    assert result.duration_seconds == pytest.approx(0.1, abs=0.01)


def test_analyze_audio_rejects_container_without_audio() -> None:
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame(16, 16, "rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    with pytest.raises(AudioAnalysisError, match="аудиопотока"):
        analyze_audio(output.getvalue())


def test_spectral_centroid_handles_tiny_signal() -> None:
    assert _spectral_centroid(np.zeros(10, dtype=np.float32)) == 0.0
