"""Тесты извлечения последовательности кадров из видео."""

import io

import av
import pytest

from protogen_delta.services.animation_frames import (
    _frame_progress,
    extract_animation_frames,
)


def _tiny_video() -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            frame = av.VideoFrame(32, 32, "rgb24")
            frame.planes[0].update(bytes([index * 30, 20, 200]) * (32 * 32))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def _tiny_gif() -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="gif") as container:
        stream = container.add_stream("gif", rate=4)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "rgb8"
        for index in range(8):
            frame = av.VideoFrame(32, 32, "rgb24")
            frame.planes[0].update(bytes([index * 30, 20, 200]) * (32 * 32))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def test_extract_animation_frames_returns_ordered_png_sequence() -> None:
    frames = extract_animation_frames(_tiny_video(), label="видео", max_frames=4)

    assert len(frames) == 4
    assert all(frame.mime_type == "image/png" for frame in frames)
    assert all(frame.data.startswith(b"\x89PNG\r\n\x1a\n") for frame in frames)
    assert [frame.label for frame in frames] == [
        "видео, кадр 1 из последовательности",
        "видео, кадр 2 из последовательности",
        "видео, кадр 3 из последовательности",
        "видео, кадр 4 из последовательности",
    ]


def test_extract_animation_frames_decodes_gif_sequence() -> None:
    frames = extract_animation_frames(_tiny_gif(), label="GIF", max_frames=4)

    assert len(frames) == 4
    assert all(frame.mime_type == "image/png" for frame in frames)


def test_extract_animation_frames_rejects_invalid_data() -> None:
    assert extract_animation_frames(b"not-video", label="видео") == ()
    with pytest.raises(ValueError, match="больше нуля"):
        extract_animation_frames(_tiny_video(), label="видео", max_frames=0)


def test_frame_progress_falls_back_to_frame_count() -> None:
    frame = av.VideoFrame(2, 2, "rgb24")
    assert _frame_progress(frame, 2, 5, 0.0) == 0.5
    assert _frame_progress(frame, 9, 0, 0.0) == 0.2
