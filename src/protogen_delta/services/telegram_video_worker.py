"""Локальный дочерний декодер: H.264/AAC, до 720p, 10 минут, два потока CPU."""

import sys
from fractions import Fraction
from pathlib import Path

import av

from protogen_delta.services.e621 import MAX_VIDEO_BYTES


def transcode(source: Path, target: Path) -> None:
    """Сохранить движение и звук в MP4; никогда не открывать сетевые URL."""
    if not 0 < source.stat().st_size <= MAX_VIDEO_BYTES:
        raise ValueError("input_size")
    with source.open("rb") as header:
        if header.read(4) != b"\x1a\x45\xdf\xa3":
            raise ValueError("input_format")
    with av.open(str(source), options={"protocol_whitelist": "file"}) as incoming:
        video = incoming.streams.video[0]
        video.thread_type = "SLICE"
        video.codec_context.thread_count = 2
        if video.width > 8192 or video.height > 8192:
            raise ValueError("dimensions")
        if incoming.duration and incoming.duration / av.time_base > 600:
            raise ValueError("duration")
        scale = min(1.0, 1280 / video.width, 720 / video.height)
        width, height = max(2, int(video.width * scale) // 2 * 2), max(
            2, int(video.height * scale) // 2 * 2
        )
        with av.open(str(target), "w", options={"movflags": "+faststart"}) as outgoing:
            output_video = outgoing.add_stream("libx264", rate=30)
            output_video.width, output_video.height = width, height
            output_video.pix_fmt = "yuv420p"
            output_video.codec_context.thread_count = 2
            output_video.options = {"preset": "veryfast", "crf": "25"}
            audio = incoming.streams.audio[0] if incoming.streams.audio else None
            output_audio = outgoing.add_stream("aac", rate=48000) if audio else None
            if output_audio:
                output_audio.layout = "stereo"
                output_audio.bit_rate = 128000
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
            last_time = -1.0
            decoded = 0
            for packet in incoming.demux([video] + ([audio] if audio else [])):
                for frame in packet.decode():
                    if not isinstance(frame, (av.VideoFrame, av.AudioFrame)):
                        continue
                    decoded += 1
                    if decoded > 50000 or (frame.time is not None and frame.time > 600):
                        raise ValueError("duration")
                    if isinstance(frame, av.VideoFrame):
                        timestamp = (
                            float(frame.time)
                            if frame.time is not None
                            else (last_time + 1 / 30)
                        )
                        if timestamp - last_time < 1 / 30 - 0.001:
                            continue
                        last_time = timestamp
                        converted = frame.reformat(
                            width=width, height=height, format="yuv420p"
                        )
                        converted.pts = round(timestamp * 90000)
                        converted.time_base = Fraction(1, 90000)
                        for encoded in output_video.encode(converted):
                            outgoing.mux(encoded)
                    elif output_audio and isinstance(frame, av.AudioFrame):
                        for converted_audio in resampler.resample(frame):
                            for encoded in output_audio.encode(converted_audio):
                                outgoing.mux(encoded)
                if target.exists() and target.stat().st_size > MAX_VIDEO_BYTES:
                    raise ValueError("output_size")
            if output_audio:
                for converted_audio in resampler.resample(None):
                    for encoded in output_audio.encode(converted_audio):
                        outgoing.mux(encoded)
                for encoded in output_audio.encode(None):
                    outgoing.mux(encoded)
            for encoded in output_video.encode(None):
                outgoing.mux(encoded)


if __name__ == "__main__":
    try:
        transcode(Path(sys.argv[1]), Path(sys.argv[2]))
    except Exception:
        sys.exit(1)
