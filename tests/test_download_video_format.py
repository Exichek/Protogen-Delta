"""Настоящие контейнеры и кодеки: VK HLS под именем MP4 и подготовка для Telegram."""

from fractions import Fraction
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import av
import numpy as np
import pytest

import protogen_delta.services.media_download_worker as worker
from protogen_delta.services.download_compression import prepare_download
from protogen_delta.services.media_remux import remux_download


def _video(
    path: Path,
    *,
    container_format: str = "mp4",
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
    sound: bool = True,
    origin: int = 0,
) -> None:
    with av.open(str(path), "w", format=container_format) as output:
        video = cast(av.video.stream.VideoStream, output.add_stream(codec, rate=24))
        video.width, video.height, video.pix_fmt = 64, 48, pixel_format
        video.codec_context.thread_count = 2
        audio = (
            cast(av.audio.stream.AudioStream, output.add_stream("aac", rate=48000))
            if sound
            else None
        )
        if audio:
            audio.layout = "stereo"
        for index in range(24):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), index * 8, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = origin * 24 + index, Fraction(1, 24)
            for packet in video.encode(frame):
                output.mux(packet)
            if audio:
                samples = np.sin(np.arange(2000, dtype=np.float32) * 0.08)[None, :]
                aframe = av.AudioFrame.from_ndarray(
                    np.repeat(samples, 2, axis=0), format="fltp", layout="stereo"
                )
                aframe.sample_rate = 48000
                aframe.pts = origin * 48000 + index * 2000
                aframe.time_base = Fraction(1, 48000)
                for packet in audio.encode(aframe):
                    output.mux(packet)
        for packet in video.encode(None):
            output.mux(packet)
        if audio:
            for packet in audio.encode(None):
                output.mux(packet)


@pytest.mark.parametrize(
    "container_format,origin,sound",
    [
        ("mpegts", 0, True),
        ("mpegts", 15, True),
        ("mp4", 0, True),
        ("mp4", 0, False),
    ],
)
def test_small_compatible_video_becomes_real_streamable_mp4_without_quality_loss(
    tmp_path: Path,
    container_format: str,
    origin: int,
    sound: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "video.mp4"
    _video(source, container_format=container_format, origin=origin, sound=sound)
    with av.open(str(source)) as original:
        original_frames = [
            f.to_ndarray(format="rgb24") for f in original.decode(video=0)
        ]
    encode = Mock(side_effect=AssertionError("Compatible video must not be reencoded"))
    monkeypatch.setattr(worker, "compress_download", encode)
    assert worker.prepare_upload(tmp_path) is False
    encode.assert_not_called()
    assert not source.exists()
    target = tmp_path / "telegram.mp4"
    data = target.read_bytes()
    assert data[4:8] == b"ftyp"
    assert data.index(b"moov") < data.index(b"mdat")
    with av.open(str(target)) as result:
        assert "mp4" in result.format.name.split(",")
        assert (
            result.duration is not None
            and 0.95 <= result.duration / av.time_base <= 1.1
        )
        stream = result.streams.video[0]
        assert stream.codec_context.name == "h264"
        frames = list(result.decode(video=0))
        assert len(frames) == len(original_frames) == 24
        assert frames[0].time is not None and 0 <= frames[0].time <= 0.05
        assert frames[0].format.name == "yuv420p"
        for frame, before in zip(frames, original_frames, strict=True):
            np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), before)
        assert bool(result.streams.audio) is sound
    if sound:
        with av.open(str(target)) as result:
            assert result.streams.audio[0].codec_context.name == "aac"
            samples = np.concatenate(
                [f.to_ndarray().reshape(-1) for f in result.decode(audio=0)]
            )
            assert samples.size >= 92000 and np.max(np.abs(samples)) > 0.1


@pytest.mark.parametrize(
    "codec,pixel_format,container_format",
    [
        ("mpeg4", "yuv420p", "mp4"),
        ("libx264", "yuv444p", "mp4"),
        ("libx264", "yuv420p", "matroska"),
    ],
)
def test_small_incompatible_mp4_is_transcoded(
    tmp_path: Path, codec: str, pixel_format: str, container_format: str
) -> None:
    source = tmp_path / "video.mp4"
    _video(
        source,
        codec=codec,
        pixel_format=pixel_format,
        container_format=container_format,
    )
    assert worker.prepare_upload(tmp_path) is True
    with av.open(str(tmp_path / "telegram.mp4")) as result:
        assert result.streams.video[0].codec_context.name == "h264"
        frames = list(result.decode(video=0))
        assert len(frames) == 24 and frames[0].format.name == "yuv420p"
        assert result.streams.audio[0].codec_context.name == "aac"


def test_broken_small_mp4_is_rejected_and_source_kept(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"not an actual video")
    with pytest.raises(av.error.InvalidDataError):
        worker.prepare_upload(tmp_path)
    assert source.read_bytes() == b"not an actual video"


def test_preparation_rejects_same_path_and_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="input_size"):
        prepare_download(source, tmp_path / "out.mp4")
    _video(source)
    with pytest.raises(ValueError, match="input_size"):
        prepare_download(source, source)


def test_remux_enforces_output_limit(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    _video(source)
    with pytest.raises(ValueError, match="output_size"):
        remux_download(source, tmp_path / "out.mp4", 1)


def test_existing_telegram_filename_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "telegram.mp4"
    _video(source, container_format="mpegts")
    assert worker.prepare_upload(tmp_path) is False
    assert not source.exists()
    with av.open(str(tmp_path / "prepared.mp4")) as result:
        assert "mp4" in result.format.name.split(",")
