"""Проверить реальный декодер, сохранение звука и ограничения MP4."""

import asyncio
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from protogen_delta.services.e621 import E621Error
from protogen_delta.services.telegram_video import TelegramVideoConverter
from protogen_delta.services.telegram_video_worker import transcode


def test_real_gif_becomes_playable_mp4_for_album(tmp_path: Path) -> None:
    source = tmp_path / "source.gif"
    Image.new("RGB", (64, 48), "red").save(
        source,
        save_all=True,
        append_images=[Image.new("RGB", (64, 48), "blue")],
        duration=250,
        loop=0,
    )
    result = tmp_path / "converted.mp4"
    result.write_bytes(
        asyncio.run(TelegramVideoConverter().convert(source.read_bytes()))
    )
    with av.open(str(result)) as video:
        assert video.streams.video[0].codec_context.name == "h264"
        frames = list(video.decode(video=0))
        assert len(frames) >= 2
        assert frames[0].to_ndarray(format="rgb24")[0, 0, 0] > 200
        assert frames[-1].to_ndarray(format="rgb24")[0, 0, 2] > 200
        assert (
            video.duration is not None and 0.45 <= video.duration / av.time_base <= 0.55
        )


def _webm(path: Path, sound: bool) -> None:
    with av.open(str(path), "w") as container:
        video = container.add_stream("libvpx-vp9", rate=24)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = container.add_stream("libopus", rate=48000) if sound else None
        if audio:
            audio.layout = "stereo"
        for index in range(24):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), index * 10, dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
            if audio:
                audio_frame = av.AudioFrame.from_ndarray(
                    np.zeros((2, 2000), dtype=np.float32),
                    format="fltp",
                    layout="stereo",
                )
                audio_frame.sample_rate = 48000
                audio_frame.pts = index * 2000
                for packet in audio.encode(audio_frame):
                    container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        if audio:
            for packet in audio.encode(None):
                container.mux(packet)


@pytest.mark.parametrize("sound", [False, True])
def test_real_webm_becomes_h264_mp4_with_sound(tmp_path: Path, sound: bool) -> None:
    source, target = tmp_path / "input.webm", tmp_path / "output.mp4"
    _webm(source, sound)
    transcode(source, target)
    with av.open(str(target)) as result:
        assert result.streams.video[0].codec_context.name == "h264"
        assert bool(result.streams.audio) is sound
        assert len(list(result.decode(video=0))) >= 20
        assert (
            result.duration is not None and 0.8 <= result.duration / av.time_base <= 1.3
        )
    if sound:
        with av.open(str(target)) as result:
            assert result.streams.audio[0].codec_context.name == "aac"
            assert sum(frame.samples for frame in result.decode(audio=0)) >= 47000


def test_converter_rejects_invalid_bytes_and_cleans_up() -> None:
    with pytest.raises(E621Error, match="MP4"):
        asyncio.run(TelegramVideoConverter().convert(b"not a video"))
    with pytest.raises(E621Error, match="лимит"):
        asyncio.run(TelegramVideoConverter().convert(b""))


def test_converter_terminates_worker_on_timeout(tmp_path: Path) -> None:
    source = tmp_path / "input.webm"
    _webm(source, False)
    with pytest.raises(E621Error, match="времени"):
        asyncio.run(TelegramVideoConverter(timeout=0.0001).convert(source.read_bytes()))


def test_converter_process_returns_valid_mp4(tmp_path: Path) -> None:
    source = tmp_path / "source.webm"
    _webm(source, True)
    data = asyncio.run(TelegramVideoConverter().convert(source.read_bytes()))
    result = tmp_path / "converted.mp4"
    result.write_bytes(data)
    with av.open(str(result)) as video:
        assert video.streams.video[0].codec_context.name == "h264"
        assert video.streams.audio[0].codec_context.name == "aac"


def test_worker_rejects_empty_input(tmp_path: Path) -> None:
    source = tmp_path / "empty.webm"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="input_size"):
        transcode(source, tmp_path / "out.mp4")


@pytest.mark.parametrize(
    "width,duration,reason",
    [(9000, 0, "dimensions"), (64, 601 * av.time_base, "duration")],
)
def test_worker_rejects_expensive_metadata_before_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    duration: int,
    reason: str,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    source = tmp_path / "input.webm"
    source.write_bytes(b"\x1a\x45\xdf\xa3metadata")
    video = SimpleNamespace(
        width=width,
        height=48,
        codec_context=SimpleNamespace(thread_count=0),
        thread_type="AUTO",
    )
    container = MagicMock()
    container.__enter__.return_value = SimpleNamespace(
        streams=SimpleNamespace(video=[video]), duration=duration
    )
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)
    with pytest.raises(ValueError, match=reason):
        transcode(source, tmp_path / "out.mp4")
