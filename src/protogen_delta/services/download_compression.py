"""Сжатие большого локального видео для облачного Telegram Bot API."""

import math
import sys
from fractions import Fraction
from pathlib import Path

import av

MAX_UPLOAD_BYTES = 49_000_000
_TARGET_BYTES = 43_000_000
_MAX_INPUT_BYTES = 100 * 1024 * 1024
_MAX_SECONDS = 600


def limit_compression_process() -> None:
    """Установить POSIX-лимиты только в одноразовом дочернем worker."""
    if sys.platform.startswith("linux"):
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def compress_download(source: Path, target: Path) -> None:
    """Перекодировать локальный AV-файл в H.264/AAC с запасом по размеру."""
    if source == target or not 0 < source.stat().st_size <= _MAX_INPUT_BYTES:
        raise ValueError("compression_input_size")
    with av.open(str(source), options={"protocol_whitelist": "file"}) as incoming:
        if not incoming.streams.video:
            raise ValueError("compression_no_video")
        video = incoming.streams.video[0]
        if not 0 < video.width <= 4096 or not 0 < video.height <= 4096:
            raise ValueError("compression_dimensions")
        duration = (incoming.duration or 0) / av.time_base
        if not math.isfinite(duration) or not 0 < duration <= _MAX_SECONDS:
            raise ValueError("compression_duration")
        video.codec_context.thread_count = 2
        video.thread_type = "SLICE"
        audio = incoming.streams.audio[0] if incoming.streams.audio else None
        if audio:
            audio.codec_context.thread_count = 2
        audio_rate = 96000 if audio else 0
        video_rate = min(
            8000000, max(100000, int(_TARGET_BYTES * 8 / duration) - audio_rate)
        )
        # Long clips need less detail at the available bitrate. Short videos
        # keep up to 720p; higher source frame rates are capped at 30 fps.
        max_width, max_height = (854, 480) if video_rate < 900000 else (1280, 720)
        scale = min(1.0, max_width / video.width, max_height / video.height)
        width = max(2, int(video.width * scale) // 2 * 2)
        height = max(2, int(video.height * scale) // 2 * 2)
        rate = min(Fraction(30), video.average_rate or Fraction(30))
        if rate <= 0:
            rate = Fraction(30)
        with av.open(str(target), "w", options={"movflags": "+faststart"}) as outgoing:
            output_video = outgoing.add_stream("libx264", rate=rate)
            output_video.width, output_video.height = width, height
            output_video.pix_fmt = "yuv420p"
            output_video.codec_context.thread_count = 2
            output_video.bit_rate = video_rate
            output_video.options = {
                "preset": "veryfast",
                "maxrate": str(video_rate),
                "bufsize": str(video_rate * 2),
            }
            output_audio = outgoing.add_stream("aac", rate=48000) if audio else None
            if output_audio:
                output_audio.layout = "stereo"
                output_audio.bit_rate = audio_rate
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
            last_time = -1.0
            origin = float(incoming.start_time or 0) / av.time_base
            decoded = 0
            for packet in incoming.demux([video] + ([audio] if audio else [])):
                for frame in packet.decode():
                    if not isinstance(frame, (av.VideoFrame, av.AudioFrame)):
                        continue
                    decoded += 1
                    if decoded > 100000:
                        raise ValueError("compression_frames")
                    if isinstance(frame, av.VideoFrame):
                        raw_time = frame.time
                        timestamp = (
                            float(raw_time) - origin
                            if raw_time is not None
                            else max(0.0, last_time + 1 / float(rate))
                        )
                        if timestamp > _MAX_SECONDS:
                            raise ValueError("compression_duration")
                        if timestamp - last_time < 1 / float(rate) - 0.001:
                            continue
                        last_time = timestamp
                        converted = frame.reformat(
                            width=width, height=height, format="yuv420p"
                        )
                        converted.pts = round(timestamp * 90000)
                        converted.time_base = Fraction(1, 90000)
                        for encoded in output_video.encode(converted):
                            outgoing.mux(encoded)
                    elif output_audio:
                        # Preserve the source timeline relative to the same
                        # stream origin rather than resetting each audio frame.
                        if frame.pts is not None and frame.time_base is not None:
                            frame.pts -= round(origin / frame.time_base)
                            if frame.time is not None and frame.time > _MAX_SECONDS:
                                raise ValueError("compression_duration")
                        for converted_audio in resampler.resample(frame):
                            for encoded in output_audio.encode(converted_audio):
                                outgoing.mux(encoded)
                if target.exists() and target.stat().st_size > MAX_UPLOAD_BYTES:
                    raise ValueError("compression_output_size")
            if last_time < 0:
                raise ValueError("compression_no_frames")
            if output_audio:
                for converted_audio in resampler.resample(None):
                    for encoded in output_audio.encode(converted_audio):
                        outgoing.mux(encoded)
                for encoded in output_audio.encode(None):
                    outgoing.mux(encoded)
            for encoded in output_video.encode(None):
                outgoing.mux(encoded)
    if not 0 < target.stat().st_size <= MAX_UPLOAD_BYTES:
        raise ValueError("compression_output_size")
