"""Проверки URL, границ процесса и отправки скачанного файла."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import av
import numpy as np
import pytest
from aiogram.types import Message

import protogen_delta.services.media_download as module
from protogen_delta.handlers.download import create_download_router
from protogen_delta.services.media_download import (
    DownloadedMedia,
    MediaDownloader,
    MediaDownloadError,
    validate_media_url,
)
from protogen_delta.services.media_download_worker import (
    download_one,
    require_public_address,
)


def test_real_remux_and_separate_stream_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yt_dlp  # type: ignore[import-untyped]

    source_video = tmp_path / "source-video.mp4"
    source_audio = tmp_path / "source-audio.m4a"
    with av.open(str(source_video), mode="w") as container:
        video = container.add_stream("mpeg4", rate=4)
        video.width = 32
        video.height = 32
        video.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame(32, 32, "rgb24")
            frame.planes[0].update(bytes([30, 50, 70]) * 1024)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
    with av.open(str(source_audio), mode="w") as container:
        audio = container.add_stream("aac", rate=8000)
        audio_frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 8000), dtype=np.float32), format="fltp", layout="mono"
        )
        audio_frame.sample_rate = 8000
        for packet in audio.encode(audio_frame):
            container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)
    directory = tmp_path / "download"
    directory.mkdir()
    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=None)
    downloader.extract_info.return_value = {
        "duration": 1,
        "title": "AV",
        "requested_formats": [{"format_id": "video"}, {"format_id": "audio"}],
    }

    def dl(path: str, info: dict[str, Any]) -> tuple[bool, bool]:
        source = source_video if info["format_id"] == "video" else source_audio
        Path(path).write_bytes(source.read_bytes())
        return True, True

    downloader.dl.side_effect = dl
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Mock(return_value=downloader))
    download_one(directory, "https://youtu.be/abc")
    with av.open(str(directory / "video.mp4"), mode="r") as result:
        assert len(result.streams.video) == 1 and len(result.streams.audio) == 1
        assert len(list(result.demux())) > 4
    assert not list(directory.glob("*.bin"))
    downloader.dl.side_effect = lambda path, info: (False, False)
    with pytest.raises(ValueError, match="failed"):
        download_one(directory, "https://youtu.be/abc")
    downloader.extract_info.return_value["requested_formats"].append({})
    with pytest.raises(ValueError, match="two"):
        download_one(directory, "https://youtu.be/abc")


@pytest.mark.parametrize(
    "url",
    [
        "http://youtube.com/watch?v=a",
        "https://youtube.com.evil.org/a",
        "https://youtube.com@127.0.0.1/a",
        "https://user@youtube.com/a",
        "https://youtube.com:8443/a",
        "https://youtube.com/",
        "https://youtube.com:bad/a",
        "https://youtube.com/a b",
        "file:///tmp/a",
        "https://127.0.0.1/a",
    ],
)
def test_rejects_unsupported_or_ambiguous_urls(url: str) -> None:
    with pytest.raises(MediaDownloadError):
        validate_media_url(url)


def test_allows_exact_platform_hosts() -> None:
    for url in (
        "https://youtu.be/abc",
        "https://www.instagram.com/reel/abc/",
        "https://vm.tiktok.com/abc/",
    ):
        assert validate_media_url(url) == url


@pytest.mark.parametrize(
    "address",
    [
        ("127.0.0.1", 80),
        ("10.1.2.3", 443),
        ("::1", 80),
        ("::ffff:8.8.8.8", 80),
        ("localhost", 80),
        "local.sock",
    ],
)
def test_worker_denies_non_public_connections(address: Any) -> None:
    with pytest.raises(OSError):
        require_public_address(address)


def test_worker_allows_public_ipv4_and_ipv6() -> None:
    require_public_address(("8.8.8.8", 443))
    require_public_address(("2606:4700:4700::1111", 443, 0, 0))


def test_worker_guard_and_entrypoint_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import protogen_delta.services.media_download_worker as worker

    hook = Mock()
    monkeypatch.setattr(worker.sys, "addaudithook", hook)
    monkeypatch.setattr(worker.socket, "setdefaulttimeout", Mock())
    worker.install_network_guard()
    audit = hook.call_args.args[0]
    audit("socket.connect", (None, ("8.8.8.8", 443)))
    audit("other", ())
    with pytest.raises(OSError):
        audit("socket.connect", (None, ("127.0.0.1", 80)))
    with pytest.raises(OSError):
        audit("subprocess.Popen", ())
    download = Mock()
    monkeypatch.setattr(worker, "download_one", download)
    monkeypatch.setattr(
        worker.sys, "argv", ["worker", str(tmp_path), "https://youtu.be/abc"]
    )
    worker.main()
    download.assert_called_once()
    download.side_effect = ValueError("not a video")
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1


def test_download_file_lifetime_and_invalid_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def write(url: str, directory: Path) -> None:
        (directory / "video.mp4").write_bytes(b"video")
        (directory / "result.json").write_text(json.dumps({"title": "Тест"}), "utf-8")

    service = MediaDownloader()
    monkeypatch.setattr(service, "_run", write)

    async def scenario() -> None:
        async with service.download("https://youtu.be/abc") as result:
            path = result.path
            assert path.exists() and result.title == "Тест"
        assert not path.exists()
        monkeypatch.setattr(module, "MAX_DOWNLOAD_BYTES", 2)
        with pytest.raises(MediaDownloadError, match="45"):
            async with service.download("https://youtu.be/abc"):
                pass

    asyncio.run(scenario())


def test_worker_process_cleanup_and_secret_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = SimpleNamespace(
        returncode=None, wait=AsyncMock(return_value=0), kill=Mock()
    )

    async def wait() -> int:
        process.returncode = 0
        return 0

    process.wait.side_effect = wait
    factory = AsyncMock(return_value=process)
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", factory)
    monkeypatch.setenv("TELEGRAM_TOKEN", "secret-not-for-worker")
    asyncio.run(MediaDownloader()._run("https://youtu.be/abc", tmp_path))
    call = factory.await_args
    assert call is not None and "TELEGRAM_TOKEN" not in call.kwargs["env"]
    process.returncode = 1
    with pytest.raises(MediaDownloadError, match="скачать"):
        asyncio.run(MediaDownloader()._run("https://youtu.be/abc", tmp_path))


def test_worker_timeout_kills_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def wait() -> int:
        await asyncio.sleep(0.1)
        return 0

    process = SimpleNamespace(
        returncode=None, wait=AsyncMock(side_effect=wait), kill=Mock()
    )
    monkeypatch.setattr(
        module.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    monkeypatch.setattr(module, "DOWNLOAD_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(MediaDownloadError, match="минуты"):
        asyncio.run(MediaDownloader()._run("https://youtu.be/abc", tmp_path))
    process.kill.assert_called_once()


def test_worker_limits_one_recorded_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yt_dlp

    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=None)
    info: dict[str, Any] = {"duration": 12, "title": "Тест"}
    downloader.extract_info.return_value = info
    factory = Mock(return_value=downloader)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", factory)
    download_one(tmp_path, "https://youtu.be/abc")
    downloader.process_info.assert_called_once_with(info)
    options = factory.call_args.args[0]
    with pytest.raises(ValueError):
        options["progress_hooks"][0](
            {"downloaded_bytes": module.MAX_DOWNLOAD_BYTES + 1}
        )
    options["progress_hooks"][0]({"downloaded_bytes": 1})
    options["logger"].debug("ignored")
    options["logger"].warning("ignored")
    options["logger"].error("ignored")
    for invalid in (
        {"_type": "playlist"},
        {"is_live": True},
        {"duration": 9999},
        {"duration": float("nan")},
        {},
    ):
        downloader.extract_info.return_value = invalid
        with pytest.raises(ValueError):
            download_one(tmp_path, "https://youtu.be/abc")


def test_download_handler_sends_file_inside_context(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")

    class Downloader:
        @asynccontextmanager
        async def download(self, url: str) -> Any:
            yield DownloadedMedia(path, "Видео")

    router = create_download_router(cast(MediaDownloader, Downloader()))
    message = Mock(spec=Message, text="/download https://youtu.be/abc")
    message.from_user = SimpleNamespace(id=42)
    message.answer = AsyncMock()
    message.answer_video = AsyncMock()
    asyncio.run(router.message.handlers[0].callback(message))
    message.answer_video.assert_awaited_once()
    asyncio.run(router.message.handlers[0].callback(message))
    message.answer.assert_awaited_once()
    message.text = "/download"
    asyncio.run(router.message.handlers[0].callback(message))
    assert "45 МБ" in message.answer.await_args.args[0]
