"""Изолированная конвертация локального видео и аудио через библиотеки FFmpeg."""

import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np

MAX_INPUT = 100 * 1024 * 1024
MAX_OUTPUT = 150 * 1024 * 1024
MAX_SECONDS = 600
VIDEO_FORMATS = {"mp4": "mp4", "avi": "avi", "mov": "mov", "mkv": "matroska"}
AUDIO_FORMATS = {
    "mp3": ("mp3", "libmp3lame"),
    "wav": ("wav", "pcm_s16le"),
    "flac": ("flac", "flac"),
}


def validate_options(mode: str, extension: str, resolution: str, bitrate: int) -> None:
    if mode not in {"video", "audio"} or extension not in (
        VIDEO_FORMATS if mode == "video" else AUDIO_FORMATS
    ):
        raise ValueError("Выбери поддерживаемый формат видео или аудио.")
    if resolution not in {"original", "480", "720", "1080"}:
        raise ValueError("Выбери исходное разрешение, 480p, 720p или 1080p.")
    if type(bitrate) is not int or bitrate not in {128, 192, 320}:
        raise ValueError("Выбери качество MP3: 128, 192 или 320 кбит/с.")


def convert_media(
    source: Path,
    target: Path,
    *,
    mode: str,
    extension: str,
    resolution: str = "720",
    bitrate: int = 192,
    max_input_bytes: int = MAX_INPUT,
    max_output_bytes: int = MAX_OUTPUT,
) -> dict[str, object]:
    """Сохранить одну видеодорожку и один звуковой поток; URL и плейлисты запрещены."""
    validate_options(mode, extension, resolution, bitrate)
    if source == target or not 0 < source.stat().st_size <= max_input_bytes:
        raise ValueError("Файл должен быть непустым и не больше 100 МБ.")
    with av.open(
        str(source),
        options={
            "protocol_whitelist": "file",
            "format_whitelist": "mov,matroska,avi,mp3,wav,flac,ogg,mpegts,aac",
        },
    ) as incoming:
        duration = (incoming.duration or 0) / av.time_base
        if not math.isfinite(duration) or not 0 < duration <= MAX_SECONDS:
            raise ValueError("Нужен файл с длительностью до 10 минут.")
        audio = incoming.streams.audio[0] if incoming.streams.audio else None
        video = (
            incoming.streams.video[0]
            if incoming.streams.video and mode == "video"
            else None
        )
        if mode == "video" and video is None:
            raise ValueError("В файле нет видеодорожки. Выбери раздел «Аудио».")
        if mode == "audio" and audio is None:
            raise ValueError("В файле нет звука для преобразования.")
        if video and not (0 < video.width <= 4096 and 0 < video.height <= 4096):
            raise ValueError("Слишком большое разрешение исходного видео.")
        for stream in (video, audio):
            if stream:
                stream.codec_context.thread_count = 2
        if video:
            video.thread_type = "SLICE"
        rotation = 0
        if video:
            first_frame = next(incoming.decode(video=0), None)
            if first_frame is None:
                raise ValueError("Не удалось прочитать видеокадры.")
            rotation = first_frame.rotation % 360
            if rotation not in {0, 90, 180, 270}:
                raise ValueError("Нестандартный угол поворота видео.")
            incoming.seek(0)
        origin = Fraction(incoming.start_time or 0, av.time_base)
        output_format = (
            VIDEO_FORMATS[extension] if mode == "video" else AUDIO_FORMATS[extension][0]
        )
        options = {"movflags": "+faststart"} if extension in {"mp4", "mov"} else {}
        with av.open(
            str(target), "w", format=output_format, options=options
        ) as outgoing:
            output_video = None
            if video:
                rate = min(Fraction(30), video.average_rate or Fraction(30))
                if rate <= 0:
                    rate = Fraction(30)
                max_height = 1080 if resolution == "original" else int(resolution)
                input_width, input_height = (
                    (video.height, video.width)
                    if rotation in {90, 270}
                    else (video.width, video.height)
                )
                scale = min(
                    1.0, max_height / input_height, max_height * 16 / 9 / input_width
                )
                output_video = outgoing.add_stream("libx264", rate=rate)
                output_video.width = max(2, int(input_width * scale) // 2 * 2)
                output_video.height = max(2, int(input_height * scale) // 2 * 2)
                output_video.pix_fmt = "yuv420p"
                output_video.codec_context.thread_count = 2
                output_video.options = {
                    "preset": "veryfast",
                    "crf": "23",
                    "maxrate": "8000000",
                    "bufsize": "16000000",
                }
                if max_output_bytes < MAX_OUTPUT:
                    maximum_rate = max(
                        100000,
                        min(
                            8000000,
                            int((max_output_bytes - 3_000_000) * 8 / duration)
                            - bitrate * 1000,
                        ),
                    )
                    output_video.options.update(
                        maxrate=str(maximum_rate), bufsize=str(maximum_rate * 2)
                    )
                if extension == "avi":
                    output_video.options["bf"] = "0"
            output_audio = None
            resampler = None
            if audio:
                lossless = mode == "audio" and extension in {"wav", "flac"}
                sample_rate = min(
                    192000 if lossless else 48000,
                    max(8000, audio.codec_context.sample_rate or 48000),
                )
                codec = (
                    AUDIO_FORMATS[extension][1]
                    if mode == "audio"
                    else ("libmp3lame" if extension == "avi" else "aac")
                )
                source_format = audio.codec_context.format
                sixteen_bit = source_format is not None and source_format.name in {
                    "s16",
                    "s16p",
                }
                if codec == "pcm_s16le" and not sixteen_bit:
                    codec = "pcm_s24le"
                output_audio = cast(
                    av.audio.stream.AudioStream,
                    outgoing.add_stream(codec, rate=sample_rate),
                )
                layout = (
                    audio.codec_context.layout.name
                    if lossless and 1 <= audio.codec_context.channels <= 8
                    else "mono" if audio.codec_context.channels == 1 else "stereo"
                )
                output_audio.layout = layout
                if codec in {"aac", "libmp3lame"}:
                    output_audio.bit_rate = bitrate * 1000
                audio_format = ("s16" if sixteen_bit else "s32") if lossless else "fltp"
                resampler = av.AudioResampler(
                    format=audio_format, layout=layout, rate=sample_rate
                )
            last_video = -1.0
            video_frames = audio_samples = decoded = 0
            audio_cursor: int | None = None
            selected: list[av.stream.Stream] = [
                s for s in (video, audio) if s is not None
            ]
            for packet in incoming.demux(selected):
                for frame in packet.decode():
                    if not isinstance(frame, (av.VideoFrame, av.AudioFrame)):
                        continue
                    decoded += 1
                    if decoded > 100000:
                        raise ValueError("Слишком много кадров в файле.")
                    if (
                        frame.time is not None
                        and float(frame.time) - float(origin) > MAX_SECONDS
                    ):
                        raise ValueError("Файл превышает 10 минут.")
                    if isinstance(frame, av.VideoFrame) and output_video:
                        timestamp = (
                            float(frame.time) - float(origin)
                            if frame.time is not None
                            else max(0.0, last_video + 1 / float(rate))
                        )
                        if timestamp > MAX_SECONDS:
                            raise ValueError("Файл превышает 10 минут.")
                        if timestamp - last_video < 1 / float(rate) - 0.001:
                            continue
                        last_video = timestamp
                        if rotation:
                            frame = av.VideoFrame.from_ndarray(
                                np.ascontiguousarray(
                                    np.rot90(
                                        frame.to_ndarray(format="rgb24"), rotation // 90
                                    )
                                ),
                                format="rgb24",
                            )
                        converted = frame.reformat(
                            width=output_video.width,
                            height=output_video.height,
                            format="yuv420p",
                        )
                        converted.pts, converted.time_base = round(
                            timestamp * 90000
                        ), Fraction(1, 90000)
                        for encoded in output_video.encode(converted):
                            outgoing.mux(encoded)
                        video_frames += 1
                    elif (
                        isinstance(frame, av.AudioFrame) and output_audio and resampler
                    ):
                        if frame.pts is not None and frame.time_base is not None:
                            frame.pts -= round(origin / frame.time_base)
                        for converted_audio in resampler.resample(frame):
                            if audio_cursor is None:
                                audio_cursor = max(
                                    0,
                                    round(
                                        float(converted_audio.time or 0) * sample_rate
                                    ),
                                )
                            converted_audio.pts, converted_audio.time_base = (
                                audio_cursor,
                                Fraction(1, sample_rate),
                            )
                            audio_cursor += converted_audio.samples
                            audio_samples += converted_audio.samples
                            if audio_samples > sample_rate * (MAX_SECONDS + 1):
                                raise ValueError("Файл превышает 10 минут.")
                            for encoded in output_audio.encode(converted_audio):
                                outgoing.mux(encoded)
                if target.exists() and target.stat().st_size > max_output_bytes:
                    raise ValueError(
                        "Результат превышает 150 МБ. Выбери меньший файл или разрешение."
                    )
            if (video and not video_frames) or (mode == "audio" and not audio_samples):
                raise ValueError("Не удалось прочитать кадры или звук файла.")
            if output_audio and resampler:
                for converted_audio in resampler.resample(None):
                    converted_audio.pts, converted_audio.time_base = (
                        audio_cursor or 0,
                        Fraction(1, sample_rate),
                    )
                    audio_cursor = (audio_cursor or 0) + converted_audio.samples
                    for encoded in output_audio.encode(converted_audio):
                        outgoing.mux(encoded)
                for encoded in output_audio.encode(None):
                    outgoing.mux(encoded)
            if output_video:
                for encoded in output_video.encode(None):
                    outgoing.mux(encoded)
    if not 0 < target.stat().st_size <= max_output_bytes:
        raise ValueError("Результат превышает 150 МБ.")
    return {"duration": round(duration, 3), "bytes": target.stat().st_size}


def run(directory: Path, options: dict[str, object]) -> None:
    mode, extension = str(options["mode"]), str(options["format"])
    result = convert_media(
        directory / "input",
        directory / ("result." + extension),
        mode=mode,
        extension=extension,
        resolution=str(options["resolution"]),
        bitrate=cast(int, options["bitrate"]),
    )
    if mode == "video" and extension != "mp4":
        convert_media(
            directory / ("result." + extension),
            directory / "preview.mp4",
            mode="video",
            extension="mp4",
            resolution="480",
            bitrate=128,
            max_input_bytes=MAX_OUTPUT,
            max_output_bytes=49_000_000,
        )
    elif mode == "audio" and extension == "flac":
        convert_media(
            directory / "input",
            directory / "preview.mp3",
            mode="audio",
            extension="mp3",
            bitrate=192,
        )
    (directory / "result.json").write_text(json.dumps(result), "utf-8")


def main() -> None:
    from protogen_delta.services.download_compression import limit_compression_process

    limit_compression_process()

    def audit(event: str, arguments: tuple[object, ...]) -> None:
        if event in {
            "socket.connect",
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
        }:
            raise OSError("network_and_processes_disabled")

    sys.addaudithook(audit)
    directory = Path(sys.argv[1])
    try:
        options = json.loads((directory / "options.json").read_text("utf-8"))
        run(directory, options)
    except Exception:
        (directory / "error.json").write_text(
            json.dumps(
                {
                    "error": "Не удалось преобразовать файл. Проверь формат, длительность и размер."
                }
            ),
            "utf-8",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
