"""Извлечение нескольких репрезентативных кадров из анимации или видео."""

import io
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
    try:
        opened = av.open(io.BytesIO(data), mode="r")
        container = cast(av.container.InputContainer, opened)
        with container:
            stream = next(
                (item for item in container.streams.video),
                None,
            )
            if stream is None:
                return ()
            duration = _duration_seconds(container, stream)
            targets = [index / max_frames for index in range(max_frames)]
            selected: list[ImageInput] = []
            next_target = 0
            for index, frame in enumerate(container.decode(stream)):
                if index >= MAX_DECODED_FRAMES or next_target >= len(targets):
                    break
                progress = _frame_progress(frame, index, stream.frames, duration)
                if progress + 0.015 < targets[next_target]:
                    continue
                selected.append(
                    ImageInput(
                        data=_encode_png(frame),
                        mime_type="image/png",
                        label=(
                            f"{label}, кадр {len(selected) + 1} из последовательности"
                            + (
                                f", время {frame.time:.2f} с"
                                if frame.time is not None
                                else ""
                            )
                        ),
                    )
                )
                next_target += 1
            return tuple(selected)
    except av.error.FFmpegError, EOFError, OSError, ValueError:
        return ()


def _duration_seconds(
    container: av.container.InputContainer, stream: av.video.VideoStream
) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    return 0.0


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
