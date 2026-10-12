"""Настоящий экстрактор X с синтетическими метаданными, без сетевых запросов."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from protogen_delta.services.media_download_worker import download_one


def _status(count: int = 1) -> dict[str, Any]:
    return {
        "full_text": "Synthetic public post",
        "extended_entities": {
            "media": [
                {
                    "type": "video",
                    "id_str": str(100 + number),
                    "video_info": {
                        "duration_millis": 1000,
                        "variants": [
                            {
                                "bitrate": 100000,
                                "url": (
                                    f"https://video.twimg.com/ext_tw_video/{100 + number}"
                                    "/pu/vid/avc1/320x180/synthetic.mp4"
                                ),
                            }
                        ],
                    },
                }
                for number in range(count)
            ]
        },
    }


@pytest.mark.parametrize("silent", [False, True])
def test_real_twitter_extractor_selects_unknown_codec_mp4_and_silent_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, silent: bool
) -> None:
    import yt_dlp  # type: ignore[import-untyped]
    from yt_dlp.extractor.twitter import TwitterIE  # type: ignore[import-untyped]

    monkeypatch.setattr(TwitterIE, "_extract_status", lambda self, twid: _status())
    original = TwitterIE._extract_variant_formats
    if silent:

        def silent_formats(self: Any, variant: Any, video_id: str) -> Any:
            formats, subtitles = original(self, variant, video_id)
            for item in formats:
                item.update(vcodec="avc1", acodec="none")
            return formats, subtitles

        monkeypatch.setattr(TwitterIE, "_extract_variant_formats", silent_formats)
    process = Mock()
    monkeypatch.setattr(yt_dlp.YoutubeDL, "process_info", process)
    download_one(tmp_path, "https://x.com/synthetic/status/123")
    process.assert_called_once()
    selected = process.call_args.args[0]
    assert selected["id"] == "100" and selected["duration"] == 1
    assert selected["ext"] == "mp4" and selected["protocol"] == "https"
    assert selected.get("acodec") == ("none" if silent else None)
    assert json.loads((tmp_path / "result.json").read_text("utf-8"))["title"]


@pytest.mark.parametrize("path, selected_id", [("", "100"), ("/video/2", "101")])
def test_real_multi_video_post_downloads_one_selected_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, selected_id: str
) -> None:
    import yt_dlp
    from yt_dlp.extractor.twitter import TwitterIE

    monkeypatch.setattr(TwitterIE, "_extract_status", lambda self, twid: _status(2))
    process = Mock()
    monkeypatch.setattr(yt_dlp.YoutubeDL, "process_info", process)
    download_one(tmp_path, "https://x.com/synthetic/status/123" + path)
    process.assert_called_once()
    assert process.call_args.args[0]["id"] == selected_id
    assert process.call_args.args[0].get("_type", "video") == "video"


def test_twitter_selector_excludes_audio_only_and_prefers_direct_https(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yt_dlp

    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=None)
    downloader.extract_info.return_value = {"duration": 1, "title": "Synthetic"}
    factory = Mock(return_value=downloader)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", factory)
    download_one(tmp_path, "https://x.com/synthetic/status/123")
    options = factory.call_args.args[0]
    # Restore the real class to run its selector against metadata without I/O.
    from yt_dlp.YoutubeDL import YoutubeDL  # type: ignore[import-untyped]

    with YoutubeDL({"format": options["format"], "quiet": True}) as real:
        info = real.process_ie_result(
            {
                "id": "synthetic",
                "title": "Synthetic",
                "formats": [
                    {
                        "format_id": "http",
                        "url": "https://video.twimg.com/a.mp4",
                        "ext": "mp4",
                        "height": 360,
                    },
                    {
                        "format_id": "hls",
                        "url": "https://video.twimg.com/a.m3u8",
                        "ext": "mp4",
                        "height": 720,
                        "protocol": "m3u8_native",
                        "vcodec": "avc1",
                        "acodec": "aac",
                    },
                    {
                        "format_id": "audio",
                        "url": "https://video.twimg.com/audio.mp4",
                        "ext": "mp4",
                        "vcodec": "none",
                        "acodec": "aac",
                    },
                ],
            },
            download=False,
        )
    assert info["format_id"] == "http"


def test_unexpected_x_playlist_still_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yt_dlp

    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=None)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Mock(return_value=downloader))
    cases: tuple[Any, ...] = (None, [], [{}, {}], [None])
    for entries in cases:
        downloader.extract_info.return_value = {"_type": "playlist", "entries": entries}
        with pytest.raises(ValueError):
            download_one(tmp_path, "https://x.com/synthetic/status/123")
    downloader.process_info.assert_not_called()
