"""Одноразовый worker yt-dlp с проверкой адреса каждого соединения."""

import ipaddress
import json
import math
import signal
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
from protogen_delta.services.download_errors import DownloadFailure, classify_failure
from protogen_delta.services.media_download import (
    _TWITTER_HOSTS,
    MAX_DOWNLOAD_BYTES,
    MAX_DOWNLOAD_SECONDS,
    directory_bytes,
    is_youtube_url,
    validate_media_url,
)
from protogen_delta.services.media_remux import remux_streams
from protogen_delta.services.youtube_download import (
    allowed_deno_command,
    deno_binary,
    limit_extraction_process,
    youtube_options,
)


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


def install_network_guard(runtime: str | None = None) -> None:
    """Системный audit hook работает только в отдельном worker-процессе."""

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event == "socket.connect":
            require_public_address(arguments[1])
        if (
            event in {"subprocess.Popen", "os.posix_spawn"}
            and len(arguments) >= 2
            and allowed_deno_command(arguments[0], arguments[1], runtime)
        ):
            return
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
            raise OSError("External downloaders are disabled")

    sys.addaudithook(audit)
    socket.setdefaulttimeout(15)


class _SilentLogger:
    def __init__(self) -> None:
        self.failure_code: str | None = None

    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        code = classify_failure(RuntimeError(message))
        if code != "unknown":
            self.failure_code = code

    def error(self, message: str) -> None:
        self.warning(message)


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


def download_one(
    directory: Path,
    url: str,
    cookies: Path | None = None,
    *,
    limit_resources: bool = False,
) -> None:
    """Один AV-файл; сессия и локальный EJS разрешены только для YouTube."""
    from yt_dlp import YoutubeDL  # type: ignore[import-untyped]

    url = validate_media_url(url)
    twitter = urlsplit(url).hostname in _TWITTER_HOSTS
    youtube = is_youtube_url(url)

    def progress(status: dict[str, Any]) -> None:
        total = directory_bytes(directory)
        if (
            int(status.get("downloaded_bytes") or 0) > MAX_DOWNLOAD_BYTES
            or total > MAX_DOWNLOAD_BYTES
        ):
            raise DownloadFailure("too_large")

    diagnostics = _SilentLogger()
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
        "logger": diagnostics,
        "progress_hooks": [progress],
        "proxy": "",
        "cachedir": False,
        "enable_file_urls": False,
        "ignore_no_formats_error": True,
    }
    if youtube:
        options.update(youtube_options())
        if cookies:
            options["cookiefile"] = str(cookies)
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
            raise DownloadFailure(
                "too_long" if duration > MAX_DOWNLOAD_SECONDS else "formats"
            )
        formats = info.get("requested_formats")
        choices = info.get("formats")
        if isinstance(choices, list) and not formats and not info.get("url"):
            videos = [f for f in choices if f.get("vcodec") not in {"none", "images"}]
            too_large = bool(videos) and all(
                isinstance(f.get("filesize"), (int, float))
                and f["filesize"] > MAX_DOWNLOAD_BYTES
                for f in videos
            )
            if too_large:
                raise DownloadFailure("too_large")
            reported = getattr(diagnostics, "failure_code", None)
            if reported in {"youtube_age", "youtube_bot", "login", "unavailable"}:
                raise DownloadFailure(reported)
            if youtube and (info.get("age_limit") or 0) >= 18:
                raise DownloadFailure("youtube_age")
            raise DownloadFailure("formats")
        if youtube and limit_resources:
            # EJS has finished. Restore strict AS limits before native parsing.
            limit_compression_process()
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
    if source.stat().st_size == 0:
        raise DownloadFailure("formats")
    if source.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise DownloadFailure("too_large")
    if source.stat().st_size <= MAX_UPLOAD_BYTES:
        return False
    target = directory / "telegram.mp4"
    compress_download(source, target)
    source.unlink()
    return True


def main() -> None:
    """Worker не выводит URL/секреты в лог; процесс сообщает только exit code."""
    directory = Path(sys.argv[1])
    try:
        youtube = is_youtube_url(sys.argv[2])
        install_network_guard(deno_binary() if youtube else None)
        if sys.platform != "win32":

            def terminate(signum: int, frame: Any) -> None:
                raise SystemExit(1)

            signal.signal(signal.SIGTERM, terminate)
        if youtube:
            limit_extraction_process()
        else:
            limit_compression_process()
        cookies = Path(sys.argv[3]) if len(sys.argv) == 4 and youtube else None
        download_one(directory, sys.argv[2], cookies, limit_resources=True)
        try:
            compressed = prepare_upload(directory)
        except DownloadFailure:
            raise
        except Exception:
            raise DownloadFailure("compression") from None
        metadata = json.loads((directory / "result.json").read_text("utf-8"))
        metadata["compressed"] = compressed
        (directory / "result.json").write_text(json.dumps(metadata), "utf-8")
    except Exception as error:
        (directory / "error.json").write_text(
            json.dumps({"error": classify_failure(error)}), "utf-8"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
