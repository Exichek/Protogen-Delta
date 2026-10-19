"""Извлечение нескольких репрезентативных кадров из анимации или видео."""

import io
import math
from itertools import islice
from typing import cast

import av

from protogen_delta.services.deepseek import ImageInput

MAX_DECODED_FRAMES = 2400


def extract_animation_frames(
    data: bytes,
    *,
    label: str,
    max_frames: int = 4,
) -> tuple[ImageInput, ...]:
    """Выбрать до четырёх кадров по временной шкале и обозначить время."""
    if max_frames <= 0:
        raise ValueError("max_frames должен быть больше нуля")
    budget = [MAX_DECODED_FRAMES]
    try:
        for allow_seek in (True, False):
            opened = av.open(
                io.BytesIO(data), mode="r", options={"protocol_whitelist": "pipe"}
            )
            container = cast(av.container.InputContainer, opened)
            with container:
                stream = next(iter(container.streams.video), None)
                if stream is None:
                    return ()
                stream.codec_context.thread_count = 1
                duration = _duration_seconds(container, stream)
                estimated = stream.frames or duration * float(stream.average_rate or 30)
                if (
                    allow_seek
                    and duration > 0
                    and stream.time_base is not None
                    and estimated > MAX_DECODED_FRAMES
                    and {"mov", "mp4", "matroska", "webm"}.intersection(
                        container.format.name.split(",")
                    )
                ):
                    try:
                        return _seek_frames(
                            container, stream, duration, max_frames, label, budget
                        )
                    except av.error.FFmpegError, EOFError, OSError, ValueError:
                        # Открыть заново для последовательного чтения; бюджет не сбрасывать.
                        continue
                return _sequential_frames(
                    container, stream, duration, max_frames, label, budget
                )
    except av.error.FFmpegError, EOFError, OSError, ValueError:
        return ()
    return ()


def _image(frame: av.VideoFrame, label: str, index: int) -> ImageInput:
    return ImageInput(
        data=_encode_png(frame),
        mime_type="image/png",
        label=f"{label}, кадр {index} из последовательности"
        + (f", время {frame.time:.2f} с" if frame.time is not None else ""),
    )


def _marked(
    selected: list[ImageInput], *, partial: bool = False, unknown: bool = False
) -> tuple[ImageInput, ...]:
    note = (
        " [Выборка неполная: не все участки временной шкалы проверены.]"
        if partial
        else (
            " [Длительность неизвестна: показаны кадры только из начала.]"
            if unknown
            else ""
        )
    )
    return tuple(ImageInput(i.data, i.mime_type, i.label + note) for i in selected)


def _seek_frames(
    container: av.container.InputContainer,
    stream: av.video.VideoStream,
    duration: float,
    count: int,
    label: str,
    budget: list[int],
) -> tuple[ImageInput, ...]:
    """Переходить к ключевым кадрам по всей длине с общим пределом декодера."""
    assert stream.time_base is not None
    origin = float((stream.start_time or 0) * stream.time_base)
    selected: list[ImageInput] = []
    for index in range(count):
        target = origin + duration * index / count
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        segment_budget = budget[0] // (count - index)
        for frame in islice(container.decode(stream), segment_budget):
            budget[0] -= 1
            if frame.time is None or frame.time + 1e-6 < target:
                continue
            selected.append(_image(frame, label, len(selected) + 1))
            break
    return _marked(selected, partial=len(selected) < count)


def _sequential_frames(
    container: av.container.InputContainer,
    stream: av.video.VideoStream,
    duration: float,
    count: int,
    label: str,
    budget: list[int],
) -> tuple[ImageInput, ...]:
    """Короткие ролики и GIF читать подряд; неполную выборку обозначать."""
    targets = [index / count for index in range(count)]
    selected: list[ImageInput] = []
    next_target = 0
    for index, frame in enumerate(islice(container.decode(stream), budget[0])):
        budget[0] -= 1
        progress = _frame_progress(frame, index, stream.frames, duration)
        if progress + 0.015 < targets[next_target]:
            continue
        selected.append(_image(frame, label, len(selected) + 1))
        next_target += 1
        if next_target >= len(targets):
            break
    return _marked(
        selected,
        partial=len(selected) < count and (duration > 0 or stream.frames > 1),
        unknown=duration <= 0 and stream.frames <= 1,
    )


def _duration_seconds(
    container: av.container.InputContainer, stream: av.video.VideoStream
) -> float:
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration = float(container.duration / av.time_base)
    else:
        return 0.0
    return duration if math.isfinite(duration) and duration > 0 else 0.0


def _frame_progress(
    frame: av.VideoFrame,
    index: int,
    frame_count: int,
    duration: float,
) -> float:
    if frame.time is not None and duration > 0:
        return min(max(float(frame.time) / duration, 0.0), 1.0)
    if frame_count > 1:
        return min(index / (frame_count - 1), 1.0)
    return min(index / 45.0, 1.0)


def _encode_png(frame: av.VideoFrame) -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="image2pipe") as container:
        stream = container.add_stream("png", rate=1)
        stream.width = frame.width
        stream.height = frame.height
        stream.pix_fmt = "rgb24"
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()
