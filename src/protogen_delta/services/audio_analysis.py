"""Измеримый анализ аудиосигнала без догадок о содержании записи."""

import io
import math
from dataclasses import dataclass

import av
import numpy as np

ANALYSIS_SAMPLE_RATE = 16_000
MAX_ANALYSIS_SECONDS = 600


class AudioAnalysisError(ValueError):
    """Аудиопоток отсутствует, повреждён или не декодируется."""


@dataclass(frozen=True, slots=True)
class AudioAnalysis:
    """Компактный набор объективных характеристик аудиосигнала."""

    duration_seconds: float
    source_sample_rate: int
    channels: int
    loudness_dbfs: float
    peak_dbfs: float
    silence_percent: float
    dynamic_range_db: float
    spectral_centroid_hz: float

    def summary(self) -> str:
        """Подготовить русское описание для модели без субъективных выводов."""
        return (
            f"длительность {self.duration_seconds:.1f} с; "
            f"каналов {self.channels}; исходная частота {self.source_sample_rate} Гц; "
            f"средний уровень {self.loudness_dbfs:.1f} dBFS; "
            f"пиковый уровень {self.peak_dbfs:.1f} dBFS; "
            f"тишина {self.silence_percent:.0f}%; "
            f"динамический диапазон около {self.dynamic_range_db:.1f} дБ; "
            f"спектральный центр около {self.spectral_centroid_hz:.0f} Гц"
        )


def analyze_audio(data: bytes) -> AudioAnalysis:
    """Декодировать аудио в моно и вычислить ограниченный набор метрик."""
    if not data:
        raise AudioAnalysisError("Аудиофайл пуст")
    try:
        with av.open(
            io.BytesIO(data), mode="r", options={"protocol_whitelist": "pipe"}
        ) as container:
            stream = next(iter(container.streams.audio), None)
            if stream is None:
                raise AudioAnalysisError("В файле нет аудиопотока")
            source_rate = int(stream.codec_context.sample_rate or stream.rate or 0)
            channels = int(stream.codec_context.channels or 0)
            resampler = av.AudioResampler(
                format="flt",
                layout="mono",
                rate=ANALYSIS_SAMPLE_RATE,
            )
            chunks: list[np.ndarray] = []
            remaining = int(ANALYSIS_SAMPLE_RATE * MAX_ANALYSIS_SECONDS)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1).astype(np.float32)
                    if len(samples) > remaining:
                        samples = samples[:remaining]
                    if len(samples):
                        chunks.append(samples)
                        remaining -= len(samples)
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
            if remaining > 0:
                for converted in resampler.resample(None):
                    samples = converted.to_ndarray().reshape(-1).astype(np.float32)
                    chunks.append(samples[:remaining])
                    remaining -= min(len(samples), remaining)
                    if remaining <= 0:
                        break
    except AudioAnalysisError:
        raise
    except (av.error.FFmpegError, EOFError, OSError, ValueError) as error:
        raise AudioAnalysisError("Не удалось декодировать аудио") from error
    if not chunks:
        raise AudioAnalysisError("В аудио нет декодируемых сэмплов")
    signal = np.concatenate(chunks)
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal)
    return _metrics(signal, source_rate=source_rate, channels=channels)


def _metrics(
    signal: np.ndarray,
    *,
    source_rate: int,
    channels: int,
) -> AudioAnalysis:
    """Вычислить уровни, паузы, динамику и спектральный центр."""
    absolute = np.abs(signal)
    rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))
    peak = float(np.max(absolute))
    window = max(1, ANALYSIS_SAMPLE_RATE // 4)
    usable = len(signal) - len(signal) % window
    windows = signal[:usable].reshape(-1, window) if usable else signal.reshape(1, -1)
    window_rms = np.sqrt(np.mean(np.square(windows), axis=1, dtype=np.float64))
    window_db = 20 * np.log10(np.maximum(window_rms, 1e-9))
    silence = float(np.mean(window_db < -45.0) * 100)
    active_db = window_db[window_db >= -45.0]
    dynamic = (
        float(np.percentile(active_db, 95) - np.percentile(active_db, 10))
        if len(active_db) >= 2
        else 0.0
    )
    centroid = _spectral_centroid(signal)
    return AudioAnalysis(
        duration_seconds=len(signal) / ANALYSIS_SAMPLE_RATE,
        source_sample_rate=source_rate or ANALYSIS_SAMPLE_RATE,
        channels=max(channels, 1),
        loudness_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(peak),
        silence_percent=silence,
        dynamic_range_db=max(dynamic, 0.0),
        spectral_centroid_hz=centroid,
    )


def _spectral_centroid(signal: np.ndarray) -> float:
    """Оценить центр спектра по нескольким окнам по всей записи."""
    fft_size = min(4096, len(signal))
    if fft_size < 32:
        return 0.0
    starts = np.linspace(
        0, len(signal) - fft_size, num=min(20, len(signal) // fft_size + 1)
    )
    frequencies = np.fft.rfftfreq(fft_size, 1 / ANALYSIS_SAMPLE_RATE)
    centroids: list[float] = []
    hann = np.hanning(fft_size)
    for start in starts.astype(int):
        start_index = int(start)
        end_index = start_index + fft_size
        spectrum = np.abs(np.fft.rfft(signal[start_index:end_index] * hann))
        total = float(np.sum(spectrum))
        if total > 1e-9:
            centroids.append(float(np.sum(frequencies * spectrum) / total))
    return float(np.median(centroids)) if centroids else 0.0


def _dbfs(value: float) -> float:
    """Преобразовать линейную амплитуду в dBFS с устойчивым нижним пределом."""
    return max(20 * math.log10(max(value, 1e-9)), -180.0)
