"""Реальное сжатие, звук, границы загрузки и отправки, подписи и cleanup."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import av
import numpy as np
import pytest
from aiogram.types import Message
from test_media_download import _stop_fake_download

import protogen_delta.services.download_compression as compression
import protogen_delta.services.media_download as download
import protogen_delta.services.media_download_worker as worker
from protogen_delta.handlers.download import create_download_router
from protogen_delta.miniapp.tools import MiniAppTools
from protogen_delta.services.media_download import (
    DownloadedMedia,
    MediaDownloader,
    MediaDownloadError,
)


def _mp4(path: Path, sound: bool, *, rate: int = 24) -> None:
    with av.open(str(path), "w") as container:
        video = container.add_stream("mpeg4", rate=rate)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = container.add_stream("aac", rate=48000) if sound else None
        if audio:
            audio.layout = "stereo"
        for index in range(rate):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), index * 3, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, rate)
            for packet in video.encode(frame):
                container.mux(packet)
            if audio:
                count = 48000 // rate
                samples = np.sin(np.arange(count, dtype=np.float32) * 0.08)[None, :]
                aframe = av.AudioFrame.from_ndarray(
                    np.repeat(samples, 2, axis=0), format="fltp", layout="stereo"
                )
                aframe.sample_rate, aframe.pts, aframe.time_base = (
                    48000,
                    index * count,
                    Fraction(1, 48000),
                )
                for packet in audio.encode(aframe):
                    container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        if audio:
            for packet in audio.encode(None):
                container.mux(packet)


@pytest.mark.parametrize("sound", [False, True])
def test_real_compression_preserves_motion_duration_and_sound(
    tmp_path: Path, sound: bool
) -> None:
    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _mp4(source, sound, rate=60)
    original = source.read_bytes()
    compression.compress_download(source, target)
    assert source.read_bytes() == original
    assert 0 < target.stat().st_size <= compression.MAX_UPLOAD_BYTES
    with av.open(str(target)) as result:
        assert result.streams.video[0].codec_context.name == "h264"
        frames = list(result.decode(video=0))
        assert 28 <= len(frames) <= 31
        assert frames[0].pts is not None and frames[0].time == 0
        assert (
            result.duration is not None
            and 0.95 <= result.duration / av.time_base <= 1.1
        )
        assert bool(result.streams.audio) is sound
    if sound:
        with av.open(str(target)) as result:
            assert result.streams.audio[0].codec_context.name == "aac"
            samples = np.concatenate(
                [f.to_ndarray().reshape(-1) for f in result.decode(audio=0)]
            )
            assert samples.size >= 94000 and np.max(np.abs(samples)) > 0.1


def test_large_download_is_compressed_and_original_removed(tmp_path: Path) -> None:
    source = tmp_path / "video.mp4"
    _mp4(source, True)
    # A valid MP4 free atom creates a large input without expensive footage.
    size = compression.MAX_UPLOAD_BYTES + 1
    with source.open("r+b") as file:
        file.seek(0, os.SEEK_END)
        padding = size - file.tell()
        file.write(padding.to_bytes(4, "big") + b"free")
        file.truncate(size)
    assert worker.prepare_upload(tmp_path) is True
    assert not source.exists()
    target = tmp_path / "telegram.mp4"
    assert 0 < target.stat().st_size < compression.MAX_UPLOAD_BYTES
    with av.open(str(target)) as result:
        assert result.streams.video and result.streams.audio


def test_small_download_is_not_reencoded_and_failed_compression_retains_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    compress = Mock(side_effect=ValueError("fail"))
    monkeypatch.setattr(worker, "compress_download", compress)
    assert worker.prepare_upload(tmp_path) is False
    compress.assert_not_called()
    monkeypatch.setattr(worker, "MAX_UPLOAD_BYTES", 2)
    with pytest.raises(ValueError, match="fail"):
        worker.prepare_upload(tmp_path)
    assert source.read_bytes() == b"video"


def test_download_and_upload_size_guards_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def write(url: str, directory: Path) -> None:
        (directory / "video.mp4").write_bytes(b"x" * 12)
        (directory / "result.json").write_text(
            json.dumps({"title": "Test", "compressed": True}), "utf-8"
        )

    service = MediaDownloader()
    monkeypatch.setattr(service, "_run", write)
    monkeypatch.setattr(download, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(download, "MAX_UPLOAD_BYTES", 5)

    async def scenario() -> None:
        with pytest.raises(MediaDownloadError, match="100"):
            async with service.download("https://youtu.be/abc"):
                pass
        monkeypatch.setattr(download, "MAX_DOWNLOAD_BYTES", 20)
        with pytest.raises(MediaDownloadError, match="сжать"):
            async with service.download("https://youtu.be/abc"):
                pass
        monkeypatch.setattr(download, "MAX_UPLOAD_BYTES", 15)
        async with service.download("https://youtu.be/abc") as media:
            assert media.compressed and media.path.exists()
            path = media.path
        assert not path.exists()

    asyncio.run(scenario())


def test_worker_rejects_bad_download_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="one"):
        worker.prepare_upload(tmp_path)
    (tmp_path / "video.mp4").write_bytes(b"large")
    monkeypatch.setattr(worker, "MAX_DOWNLOAD_BYTES", 2)
    with pytest.raises(ValueError, match="large"):
        worker.prepare_upload(tmp_path)


def test_compressor_rejects_invalid_inputs_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="input_size"):
        compression.compress_download(source, target)
    source.write_bytes(b"123")
    monkeypatch.setattr(compression, "_MAX_INPUT_BYTES", 2)
    with pytest.raises(ValueError, match="input_size"):
        compression.compress_download(source, target)


@pytest.mark.parametrize("reason", ["no_video", "dimensions", "duration"])
def test_compressor_checks_metadata_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"test")
    video = SimpleNamespace(width=5000 if reason == "dimensions" else 64, height=48)
    incoming = SimpleNamespace(
        streams=SimpleNamespace(video=[] if reason == "no_video" else [video]),
        duration=601 * av.time_base,
    )
    context = Mock()
    context.__enter__ = Mock(return_value=incoming)
    context.__exit__ = Mock(return_value=None)
    monkeypatch.setattr(compression.av, "open", Mock(return_value=context))
    with pytest.raises(ValueError, match=reason):
        compression.compress_download(source, tmp_path / "output.mp4")


def test_compressed_caption_in_command_and_miniapp(tmp_path: Path) -> None:
    source = tmp_path / "telegram.mp4"
    source.write_bytes(b"video")

    class Downloader:
        @asynccontextmanager
        async def download(self, url: str) -> Any:
            yield DownloadedMedia(source, "Synthetic", compressed=True)

    service = cast(MediaDownloader, Downloader())
    message = Mock(
        spec=Message,
        from_user=SimpleNamespace(id=1),
        text="/download https://youtu.be/abc",
        answer_video=AsyncMock(),
    )
    bot = AsyncMock()

    async def scenario() -> None:
        await create_download_router(service).message.handlers[0].callback(message)
        await MiniAppTools(bot, service).download(1, "https://youtu.be/abc")

    asyncio.run(scenario())
    assert "Сжато" in message.answer_video.await_args.kwargs["caption"]
    assert "Сжато" in bot.send_video.await_args.kwargs["caption"]


def test_exact_100_mib_is_accepted_and_larger_input_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "video.mp4"

    def write_output(source: Path, target: Path) -> None:
        target.write_bytes(b"compressed")

    compress = Mock(side_effect=write_output)
    monkeypatch.setattr(worker, "compress_download", compress)
    with source.open("wb") as file:
        file.truncate(100 * 1024 * 1024)
    assert worker.prepare_upload(tmp_path)
    compress.assert_called_once()
    (tmp_path / "telegram.mp4").unlink()
    with source.open("wb") as file:
        file.truncate(100 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="large"):
        worker.prepare_upload(tmp_path)
    assert compress.call_count == 1


def test_repeated_cancellation_waits_for_worker_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download, "_stop_worker", _stop_fake_download)

    async def scenario() -> None:
        started, killed, reap = asyncio.Event(), asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(returncode=None, kill=Mock(side_effect=killed.set))

        async def wait() -> int:
            if not killed.is_set():
                started.set()
                await asyncio.Event().wait()
            await reap.wait()
            process.returncode = -9
            return -9

        process.wait = wait
        monkeypatch.setattr(
            download.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
        )
        task = asyncio.create_task(
            MediaDownloader()._run("https://youtu.be/abc", tmp_path)
        )
        await started.wait()
        task.cancel()
        await killed.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        reap.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        process.kill.assert_called_once()

    asyncio.run(scenario())


def test_compression_posix_limits_do_not_modify_test_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    limits = SimpleNamespace(RLIMIT_AS=1, RLIMIT_CPU=2, RLIMIT_CORE=3, setrlimit=Mock())
    monkeypatch.setitem(sys.modules, "resource", limits)
    monkeypatch.setattr(compression.sys, "platform", "linux")
    compression.limit_compression_process()
    assert limits.setrlimit.call_args_list[0].args == (1, (2 * 1024**3, 2 * 1024**3))
    assert limits.setrlimit.call_args_list[1].args == (2, (180, 180))
    assert limits.setrlimit.call_args_list[2].args == (3, (0, 0))
