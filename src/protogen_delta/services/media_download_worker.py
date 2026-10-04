"""Одноразовый worker yt-dlp с проверкой адреса каждого соединения."""

import ipaddress
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any

from protogen_delta.services.media_download import (
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


def download_one(directory: Path, url: str) -> None:
    """Только один готовый AV-файл; cookies, DRM и внешние процессы не используются."""
    from yt_dlp import YoutubeDL  # type: ignore[import-untyped]

    validate_media_url(url)

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
            f"best[ext=mp4][filesize<=?{MAX_DOWNLOAD_BYTES}][vcodec!=none][acodec!=none]/"
            f"bestvideo[ext=mp4][height<=720][filesize<=?{MAX_DOWNLOAD_BYTES}]"
            f"+bestaudio[ext=m4a][filesize<=?{MAX_DOWNLOAD_BYTES}]/"
            f"best[filesize<=?{MAX_DOWNLOAD_BYTES}][vcodec!=none][acodec!=none]"
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
        if not info or info.get("_type", "video") != "video" or info.get("is_live"):
            raise ValueError("Not one recorded video")
        duration = float(info.get("duration") or 0)
        if not math.isfinite(duration) or not 0 < duration <= MAX_DOWNLOAD_SECONDS:
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
        (directory / "result.json").write_text(
            json.dumps({"title": str(info.get("title") or "Видео")[:150]}), "utf-8"
        )


def main() -> None:
    """Worker не выводит URL/секреты в лог; процесс сообщает только exit code."""
    try:
        install_network_guard()
        download_one(Path(sys.argv[1]), sys.argv[2])
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
