"""Декодировать недоверенное аудио в ограниченный PCM WAV внутри worker."""

import io
import wave

import av


def prepare_speech_audio(data: bytes) -> bytes:
    samples = bytearray()
    limit = 600 * 16000 * 2
    with av.open(
        io.BytesIO(data), mode="r", options={"protocol_whitelist": "pipe"}
    ) as container:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            raise ValueError("audio_stream")
        stream.codec_context.thread_count = 1
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunk = converted.to_ndarray().tobytes()
                samples.extend(chunk[: limit - len(samples)])
            if len(samples) >= limit:
                break
    if not samples:
        raise ValueError("audio_empty")
    output = io.BytesIO()
    with wave.open(output, "wb") as result:
        result.setnchannels(1)
        result.setsampwidth(2)
        result.setframerate(16000)
        result.writeframes(samples)
    return output.getvalue()
