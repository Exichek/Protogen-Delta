"""Instagram Reels без длительности API проверяются по реальному локальному видео."""

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import av
import pytest
from test_download_video_format import _video

import protogen_delta.services.media_download_worker as worker
from protogen_delta.services.download_errors import DownloadFailure
from protogen_delta.services.media_download import is_instagram_url, validate_media_url


def _info(duration: Any = None) -> dict[str, Any]:
    return {
        "id": "SyntheticA1",
        "title": "Публичный синтетический Reels",
        "duration": duration,
        "formats": [
            {
                "format_id": "audio",
                "url": "https://cdn.example.test/audio.m4a",
                "ext": "m4a",
                "vcodec": "none",
                "acodec": "mp4a.40.2",
            },
            {
                "format_id": "direct",
                "url": "https://cdn.example.test/video.mp4",
                "ext": "mp4",
            },
        ],
    }


@pytest.mark.parametrize("sound", [False, True])
@pytest.mark.parametrize("missing", ["absent", "null"])
def test_instagram_selects_unknown_codec_mp4_and_checks_real_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sound: bool, missing: str
) -> None:
    import yt_dlp  # type: ignore[import-untyped]
    from yt_dlp.extractor.instagram import InstagramIE  # type: ignore[import-untyped]

    source = tmp_path / "source.mp4"
    _video(source, sound=sound)
    output = tmp_path / "download"
    output.mkdir()
    info = _info()
    if missing == "absent":
        del info["duration"]
    monkeypatch.setattr(InstagramIE, "_real_extract", lambda self, url: info)
    process = Mock(
        side_effect=lambda selected: shutil.copyfile(source, output / "video.mp4")
    )
    monkeypatch.setattr(yt_dlp.YoutubeDL, "process_info", process)
    url = "https://www.instagram.com/reel/SyntheticA1/?utm_source=share"
    worker.download_one(output, url)
    selected = process.call_args.args[0]
    assert selected["format_id"] == "direct" and selected["protocol"] == "https"
    assert selected.get("duration") is None
    assert not json.loads((output / "result.json").read_text("utf-8"))["animation"]
    assert not worker.prepare_upload(output)
    with av.open(str(output / "telegram.mp4")) as result:
        assert result.duration is not None
        assert 0 < result.duration / av.time_base <= 600
        assert result.streams.video[0].codec_context.name == "h264"
        assert bool(result.streams.audio) is sound


@pytest.mark.parametrize(
    "duration,code",
    [
        (0, "formats"),
        (-1, "formats"),
        (float("nan"), "formats"),
        (float("inf"), "too_long"),
        (601, "too_long"),
    ],
)
def test_invalid_or_long_known_duration_rejected_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration: float, code: str
) -> None:
    import yt_dlp

    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=None)
    downloader.extract_info.return_value = _info(duration)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Mock(return_value=downloader))
    with pytest.raises(DownloadFailure, match=code):
        worker.download_one(tmp_path, "https://instagram.com/p/SyntheticA1/")
    downloader.process_info.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/synthetic",
        "https://vimeo.com/123",
        "https://www.tiktok.com/@synthetic/video/123",
        "https://x.com/synthetic/status/123",
    ],
)
def test_missing_duration_exception_is_scoped_to_instagram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    import yt_dlp

    downloader = Mock()
    downloader.__enter__ = Mock(return_value=downloader)
    downloader.__exit__ = Mock(return_value=None)
    downloader.extract_info.return_value = _info()
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Mock(return_value=downloader))
    with pytest.raises(DownloadFailure, match="formats"):
        worker.download_one(tmp_path, url)
    downloader.process_info.assert_not_called()


def test_instagram_retains_separate_stream_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если прямого MP4 нет, сохранить загрузку отдельных видео и аудио."""
    import yt_dlp
    from yt_dlp.extractor.instagram import InstagramIE

    video, combined, audio = (
        tmp_path / "video.mp4",
        tmp_path / "combined.mp4",
        tmp_path / "audio.m4a",
    )
    _video(video, sound=False)
    _video(combined)
    with av.open(str(combined)) as source, av.open(str(audio), "w") as target:
        stream = source.streams.audio[0]
        output_stream = target.add_stream_from_template(stream)
        for packet in source.demux(stream):
            if packet.dts is not None:
                packet.stream = output_stream
                target.mux(packet)
    info = _info()
    info["formats"] = [
        info["formats"][0],
        {
            "format_id": "video",
            "url": "https://cdn.example.test/video.mp4",
            "ext": "mp4",
            "vcodec": "avc1.64001F",
            "acodec": "none",
            "height": 640,
        },
    ]
    monkeypatch.setattr(InstagramIE, "_real_extract", lambda self, url: info)
    output = tmp_path / "download"
    output.mkdir()

    def dl(
        self: Any, filename: str, selected: dict[str, Any], **kwargs: Any
    ) -> tuple[bool, bool]:
        shutil.copyfile(audio if selected["format_id"] == "audio" else video, filename)
        return True, True

    monkeypatch.setattr(yt_dlp.YoutubeDL, "dl", dl)
    worker.download_one(output, "https://www.instagram.com/reel/SyntheticA1/")
    assert not worker.prepare_upload(output)
    with av.open(str(output / "telegram.mp4")) as result:
        assert len(result.streams.video) == len(result.streams.audio) == 1


def test_local_probe_enforces_duration_and_size_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "video.mp4"
    _video(source)
    worker._check_instagram_duration(tmp_path)
    monkeypatch.setattr(worker, "MAX_DOWNLOAD_SECONDS", 0.5)
    with pytest.raises(DownloadFailure, match="too_long"):
        worker._check_instagram_duration(tmp_path)
    monkeypatch.setattr(worker, "MAX_DOWNLOAD_SECONDS", 600)
    monkeypatch.setattr(worker, "MAX_DOWNLOAD_BYTES", source.stat().st_size - 1)
    with pytest.raises(DownloadFailure, match="too_large"):
        worker._check_instagram_duration(tmp_path)


@pytest.mark.parametrize(
    "kind", ["missing", "empty", "multiple", "corrupt", "audio", "no_duration"]
)
def test_local_probe_rejects_non_video_or_unverifiable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    if kind in {"empty", "corrupt", "no_duration"}:
        (tmp_path / "video.mp4").write_bytes(b"" if kind == "empty" else b"not media")
    elif kind == "multiple":
        (tmp_path / "first.mp4").write_bytes(b"x")
        (tmp_path / "second.mp4").write_bytes(b"x")
    elif kind == "audio":
        _video(tmp_path / "video.mp4")
        opened = Mock()
        opened.__enter__ = Mock(return_value=opened)
        opened.__exit__ = Mock(return_value=None)
        opened.duration = 1000000
        opened.streams.video = []
        monkeypatch.setattr(worker.av, "open", Mock(return_value=opened))
    if kind == "no_duration":
        opened = Mock()
        opened.__enter__ = Mock(return_value=opened)
        opened.__exit__ = Mock(return_value=None)
        opened.duration = None
        opened.streams.video = [Mock()]
        monkeypatch.setattr(worker.av, "open", Mock(return_value=opened))
    with pytest.raises(DownloadFailure, match="formats"):
        worker._check_instagram_duration(tmp_path)


def test_instagram_host_check_preserves_exact_origin() -> None:
    assert is_instagram_url("https://instagram.com/reel/SyntheticA1/")
    assert is_instagram_url("https://www.instagram.com/p/SyntheticA1/")
    assert not is_instagram_url("https://instagram.com.evil.test/reel/SyntheticA1/")
    with pytest.raises(ValueError):
        validate_media_url("https://instagram.com.evil.test/reel/SyntheticA1/")
