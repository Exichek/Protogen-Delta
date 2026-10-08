"""Одноразовый worker yt-dlp с проверкой адреса каждого соединения."""

import ipaddress
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import av

from protogen_delta.services.download_compression import (
    MAX_UPLOAD_BYTES,
    compress_download,
    limit_compression_process,
)
from protogen_delta.services.media_download import (
    _TWITTER_HOSTS,
    MAX_DOWNLOAD_BYTES,
    MAX_DOWNLOAD_SECONDS,
    validate_media_url,
)
from protogen_delta.services.media_remux import remux_streams


def require_public_address(address: Any) -> None:
    """Блокировать loopback/LAN даже после redirect или повторного DNS lookup."""
    if not isinstance(address, tuple) or len(address) < 2:
        raise OSError("Only public Internet sockets are allowed")
    try:
        ip = ipaddress.ip_address(address[0])
    except ValueError as error:
        raise OSError("An IP address is required at connect time") from error
    if not ip.is_global or getattr(ip, "ipv4_mapped", None) is not None:
        raise OSError("Non-public destination blocked")


def install_network_guard() -> None:
    """Системный audit hook работает только в отдельном worker-процессе."""

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event == "socket.connect":
            require_public_address(arguments[1])
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
            raise OSError("External downloaders are disabled")

    sys.addaudithook(audit)
    socket.setdefaulttimeout(15)


class _SilentLogger:
    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def _twitter_gif(info: dict[str, Any]) -> bool:
    """X exposes GIF media as silent MP4 under its tweet_video path."""
    formats = [info, *(info.get("formats") or [])]
    for item in formats:
        url = item.get("url", "")
        if not isinstance(url, str):
            continue
        parsed = urlsplit(url)
        if (
            parsed.scheme == "https"
            and parsed.hostname == "video.twimg.com"
            and parsed.path.startswith("/tweet_video/")
            and parsed.path.endswith(".mp4")
        ):
            return True
    return False


def _check_animation(directory: Path) -> None:
    """Check missing X duration against the local file before delivery."""
    paths = [p for p in directory.glob("*.mp4") if p.is_file() and not p.is_symlink()]
    if len(paths) != 1 or not 0 < paths[0].stat().st_size <= MAX_DOWNLOAD_BYTES:
        raise ValueError("Expected one bounded animation")
    with av.open(str(paths[0]), options={"protocol_whitelist": "file"}) as video:
        duration = (video.duration or 0) / av.time_base
        if not math.isfinite(duration) or not 0 < duration <= MAX_DOWNLOAD_SECONDS:
            raise ValueError("Animation duration missing or too long")
        if len(video.streams.video) != 1 or video.streams.audio:
            raise ValueError("Animation must have one video and no audio")
        stream = video.streams.video[0]
        if (
            stream.codec_context.name != "h264"
            or not 0 < stream.width <= 4096
            or not 0 < stream.height <= 4096
        ):
            raise ValueError("Unsupported animation format")


def download_one(directory: Path, url: str) -> None:
    """Только один готовый AV-файл; cookies, DRM и внешние процессы не используются."""
    from yt_dlp import YoutubeDL  # type: ignore[import-untyped]

    url = validate_media_url(url)
    twitter = urlsplit(url).hostname in _TWITTER_HOSTS

    def progress(status: dict[str, Any]) -> None:
        total = sum(
            item.stat().st_size for item in directory.rglob("*") if item.is_file()
        )
        if (
            int(status.get("downloaded_bytes") or 0) > MAX_DOWNLOAD_BYTES
            or total > MAX_DOWNLOAD_BYTES
        ):
            raise ValueError("Too large")

    options = {
        "outtmpl": str(directory / "video.%(ext)s"),
        "format": (
            # X's direct MP4 variants often omit codec metadata. Strict
            # acodec/vcodec filters reject these files and silent videos.
            f"best[ext=mp4][protocol=https][filesize<=?{MAX_DOWNLOAD_BYTES}]"
            "[vcodec!=?none]/"
            f"best[ext=mp4][filesize<=?{MAX_DOWNLOAD_BYTES}][vcodec!=?none]"
            if twitter
            else (
                f"best[ext=mp4][filesize<=?{MAX_DOWNLOAD_BYTES}][vcodec!=none][acodec!=none]/"
                f"bestvideo[ext=mp4][height<=720][filesize<=?{MAX_DOWNLOAD_BYTES}]"
                f"+bestaudio[ext=m4a][filesize<=?{MAX_DOWNLOAD_BYTES}]/"
                f"best[filesize<=?{MAX_DOWNLOAD_BYTES}][vcodec!=none][acodec!=none]"
            )
        ),
        "noplaylist": True,
        "playlist_items": "1",
        "max_filesize": MAX_DOWNLOAD_BYTES,
        "socket_timeout": 15,
        "retries": 1,
        "fragment_retries": 1,
        "concurrent_fragment_downloads": 1,
        "hls_prefer_native": True,
        "hls_use_mpegts": False,
        "fixup": "never",
        "quiet": True,
        "no_warnings": True,
        "logger": _SilentLogger(),
        "progress_hooks": [progress],
        "proxy": "",
        "cachedir": False,
        "enable_file_urls": False,
    }
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
        if twitter and info and info.get("_type") == "playlist":
            # A post may contain several videos; playlist_items=1 already
            # selects one. Explicit /video/N links select their own entry.
            entries = info.get("entries")
            if not isinstance(entries, (list, tuple)) or len(entries) != 1:
                raise ValueError("Expected one selected video from the X post")
            info = entries[0]
        if not info or info.get("_type", "video") != "video" or info.get("is_live"):
            raise ValueError("Not one recorded video")
        animation = twitter and _twitter_gif(info)
        duration = float(info.get("duration") or 0)
        missing_gif_duration = animation and info.get("duration") is None
        if not missing_gif_duration and (
            not math.isfinite(duration) or not 0 < duration <= MAX_DOWNLOAD_SECONDS
        ):
            raise ValueError("Duration missing or too long")
        formats = info.get("requested_formats")
        if formats:
            if len(formats) != 2:
                raise ValueError("Expected exactly two AV streams")
            inputs = []
            for index, selected in enumerate(formats):
                stream_info = {**info, **selected}
                stream_info.pop("requested_formats", None)
                path = directory / f"input{index}.bin"
                success, _ = downloader.dl(str(path), stream_info)
                if not success:
                    raise ValueError("Stream download failed")
                inputs.append(path)
            remux_streams(inputs, directory / "video.mp4")
            for path in inputs:
                path.unlink()
        else:
            downloader.process_info(info)
        if animation:
            _check_animation(directory)
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "title": str(info.get("title") or "Видео")[:150],
                    "animation": animation,
                }
            ),
            "utf-8",
        )


def prepare_upload(directory: Path) -> bool:
    """После загрузки сжать только большой файл и оставить один результат."""
    candidates = [
        item
        for item in directory.iterdir()
        if item.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}
        and item.is_file()
        and not item.is_symlink()
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one downloaded video")
    source = candidates[0]
    if not 0 < source.stat().st_size <= MAX_DOWNLOAD_BYTES:
        raise ValueError("Downloaded video too large")
    if source.stat().st_size <= MAX_UPLOAD_BYTES:
        return False
    target = directory / "telegram.mp4"
    compress_download(source, target)
    source.unlink()
    return True


def main() -> None:
    """Worker не выводит URL/секреты в лог; процесс сообщает только exit code."""
    try:
        install_network_guard()
        limit_compression_process()
        directory = Path(sys.argv[1])
        download_one(directory, sys.argv[2])
        compressed = prepare_upload(directory)
        metadata = json.loads((directory / "result.json").read_text("utf-8"))
        metadata["compressed"] = compressed
        (directory / "result.json").write_text(json.dumps(metadata), "utf-8")
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
