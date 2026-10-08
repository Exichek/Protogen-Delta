"""GIF без duration в API X, реальные MP4 и отдельная доставка анимаций."""

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import Message
from test_download_compression import _mp4
from test_twitter_download import _status

import protogen_delta.services.media_download_worker as worker
from protogen_delta.handlers.download import create_download_router
from protogen_delta.miniapp.tools import MiniAppTools
from protogen_delta.services.download_compression import compress_download
from protogen_delta.services.media_download import (
    DownloadedMedia,
    MediaDownloader,
    MediaDownloadError,
    validate_media_url,
)


@pytest.mark.parametrize("host", ["fixupx.com", "www.fixupx.com", "fxtwitter.com"])
@pytest.mark.parametrize("path", ["/i/status/123", "/name/status/123/video/2?s=20"])
def test_embed_links_are_normalized_without_contacting_embed_host(
    host: str, path: str
) -> None:
    assert validate_media_url(f"https://{host}:443{path}") == "https://x.com" + path


@pytest.mark.parametrize(
    "url",
    [
        "http://fixupx.com/i/status/123",
        "https://fixupx.com.evil.test/i/status/123",
        "https://fixupx.com@localhost/i/status/123",
        "https://user@fixupx.com/i/status/123",
        "https://fxtwitter.com:8443/name/status/123",
        "https://fixupx.com/name",
        "https://fixupx.com/i/status/not-an-id",
    ],
)
def test_embed_host_does_not_bypass_url_checks(url: str) -> None:
    with pytest.raises(MediaDownloadError):
        validate_media_url(url)


def test_real_x_extractor_downloads_gif_without_api_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yt_dlp  # type: ignore[import-untyped]
    from yt_dlp.extractor.twitter import TwitterIE  # type: ignore[import-untyped]

    source, prepared = tmp_path / "source.mp4", tmp_path / "prepared.mp4"
    _mp4(source, False)
    compress_download(source, prepared)
    directory = tmp_path / "download"
    directory.mkdir()
    status = _status()
    media = status["extended_entities"]["media"][0]
    media["type"] = "animated_gif"
    media["video_info"] = {
        "variants": [
            {
                "url": "https://video.twimg.com/tweet_video/synthetic.mp4",
                "content_type": "video/mp4",
            }
        ]
    }
    monkeypatch.setattr(TwitterIE, "_extract_status", lambda self, twid: status)
    process = Mock(
        side_effect=lambda info: shutil.copyfile(prepared, directory / "video.mp4")
    )
    monkeypatch.setattr(yt_dlp.YoutubeDL, "process_info", process)
    worker.download_one(directory, "https://fixupx.com/i/status/123")
    assert process.call_args.args[0].get("duration") is None
    assert json.loads((directory / "result.json").read_text("utf-8"))["animation"]
    assert not worker.prepare_upload(directory)
    assert (directory / "video.mp4").read_bytes() == prepared.read_bytes()


def test_animation_is_not_inferred_from_missing_duration_or_silent_video() -> None:
    assert not worker._twitter_gif({"duration": None})
    assert not worker._twitter_gif(
        {"url": "https://video.twimg.com/ext_tw_video/1/silent.mp4"}
    )
    assert not worker._twitter_gif(
        {"url": "https://video.twimg.com.evil.test/tweet_video/fake.mp4"}
    )
    assert not worker._twitter_gif({"url": None})


@pytest.mark.parametrize("sound", [False, True])
def test_gif_probe_checks_real_duration_codec_and_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sound: bool
) -> None:
    source = tmp_path / "source.mp4"
    _mp4(source, sound)
    directory = tmp_path / "result"
    directory.mkdir()
    target = directory / "video.mp4"
    # A silent MPEG-4 Part 2 file is not a Telegram H.264 animation.
    shutil.copyfile(source, target)
    with pytest.raises(ValueError, match="format|audio"):
        worker._check_animation(directory)
    target.unlink()
    compress_download(source, target)
    if sound:
        with pytest.raises(ValueError, match="audio"):
            worker._check_animation(directory)
    else:
        worker._check_animation(directory)
        monkeypatch.setattr(worker, "MAX_DOWNLOAD_SECONDS", 0.5)
        with pytest.raises(ValueError, match="duration"):
            worker._check_animation(directory)
    target.unlink()
    with pytest.raises(ValueError, match="one"):
        worker._check_animation(directory)


def test_command_and_miniapp_send_animation_inside_file_lifetime(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "gif.mp4"
        closed: list[bool] = []

        class Downloader:
            @asynccontextmanager
            async def download(self, url: str) -> AsyncIterator[DownloadedMedia]:
                path.write_bytes(b"synthetic")
                try:
                    yield DownloadedMedia(path, "GIF", animation=True)
                finally:
                    path.unlink()
                    closed.append(True)

        async def during_send(*args: object, **kwargs: object) -> None:
            assert path.exists()

        downloader = cast(MediaDownloader, Downloader())
        message = Mock(
            spec=Message, text="/download https://x.com/synthetic/status/123"
        )
        message.from_user = SimpleNamespace(id=42)
        message.answer_animation = AsyncMock(side_effect=during_send)
        message.answer_video = AsyncMock()
        await create_download_router(downloader).message.handlers[0].callback(message)
        message.answer_animation.assert_awaited_once()
        message.answer_video.assert_not_awaited()
        bot = AsyncMock()
        bot.send_animation.side_effect = during_send
        result = await MiniAppTools(bot, downloader).download(
            42, "https://x.com/synthetic/status/123"
        )
        assert bot.send_animation.await_args.args[0] == 42
        assert isinstance(result["message"], str) and "GIF" in result["message"]
        bot.send_video.assert_not_awaited()
        assert len(closed) == 2 and not path.exists()

    asyncio.run(scenario())


def test_parent_preserves_gif_metadata_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def write(url: str, directory: Path) -> None:
        (directory / "video.mp4").write_bytes(b"video")
        (directory / "result.json").write_text(
            json.dumps({"title": "GIF", "animation": True}), "utf-8"
        )

    async def scenario() -> None:
        downloader = MediaDownloader()
        monkeypatch.setattr(downloader, "_run", write)
        async with downloader.download("https://fixupx.com/i/status/123") as media:
            assert media.animation and media.path.exists()
        assert not media.path.exists()

    asyncio.run(scenario())
