"""Изолированная загрузка одного публичного видео по явной команде."""

import asyncio
import json
import logging
import os
import re
import signal
import stat
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.services.download_compression import MAX_UPLOAD_BYTES
from protogen_delta.services.download_errors import ERROR_MESSAGES, read_failure
from protogen_delta.services.youtube_download import (
    snapshot_youtube_cookies,
)

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_DOWNLOAD_SECONDS = 600
DOWNLOAD_TIMEOUT_SECONDS = 240
_VK_HOSTS = frozenset(
    {
        "vk.com",
        "www.vk.com",
        "m.vk.com",
        "new.vk.com",
        "vk.ru",
        "www.vk.ru",
        "m.vk.ru",
        "new.vk.ru",
        "vkvideo.ru",
        "www.vkvideo.ru",
        "m.vkvideo.ru",
        "vksport.vkvideo.ru",
    }
)
_TWITTER_HOSTS = frozenset(
    f"{prefix}{domain}"
    for domain in ("x.com", "twitter.com")
    for prefix in ("", "www.", "m.", "mobile.")
)
_TWITTER_EMBED_HOSTS = frozenset(
    f"{prefix}{domain}"
    for domain in ("fixupx.com", "fxtwitter.com")
    for prefix in ("", "www.")
)
_TWITTER_PATH = re.compile(
    r"/(?:(?:i/web|[A-Za-z0-9_]+)/status|statuses)/\d+(?:/video/\d+)?/?"
)
_HOSTS = (
    frozenset(
        {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "instagram.com",
            "www.instagram.com",
            "tiktok.com",
            "www.tiktok.com",
            "vm.tiktok.com",
            "vt.tiktok.com",
            "vimeo.com",
            "www.vimeo.com",
        }
    )
    | _VK_HOSTS
    | _TWITTER_HOSTS
    | _TWITTER_EMBED_HOSTS
)
_YOUTUBE_HOSTS = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
)


def is_youtube_url(url: str) -> bool:
    return urlsplit(url).hostname in _YOUTUBE_HOSTS


def directory_bytes(directory: Path) -> int:
    total = 0
    for item in directory.rglob("*"):
        if item.name == "youtube-cookies.txt":
            continue
        try:
            metadata = item.stat(follow_symlinks=False)
        except FileNotFoundError:
            # A completed fragment or remux input can disappear during scanning.
            continue
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
    return total


class MediaDownloadError(ValueError):
    """Ссылка или доступное видео не соответствуют ограничениям загрузки."""


async def _stop_worker(process: asyncio.subprocess.Process) -> None:
    if sys.platform != "win32":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), 2.0)
        except TimeoutError:
            pass
        # Kill descendants even if their parent has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.returncode is None:
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
        if process.returncode is None:
            process.kill()
    await process.wait()


@dataclass(frozen=True, slots=True)
class DownloadedMedia:
    """Временный файл, доступный только внутри контекстного менеджера."""

    path: Path
    title: str
    compressed: bool = False
    animation: bool = False


def validate_media_url(url: str) -> str:
    """Принять только точный HTTPS-хост поддерживаемой платформы."""
    try:
        parsed = urlsplit(url)
        valid = (
            len(url) <= 2048
            and not any(char.isspace() for char in url)
            and parsed.scheme == "https"
            and parsed.hostname in _HOSTS
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
            and bool(parsed.path.strip("/"))
            and (
                parsed.hostname not in _TWITTER_HOSTS | _TWITTER_EMBED_HOSTS
                or bool(_TWITTER_PATH.fullmatch(parsed.path))
            )
        )
    except ValueError:
        valid = False
    if not valid:
        raise MediaDownloadError(
            "Нужна HTTPS-ссылка на одно публичное видео YouTube, Instagram, "
            "TikTok, Vimeo, VK/VK Видео или X/Twitter (ссылка на пост)."
        )
    if parsed.hostname in _TWITTER_EMBED_HOSTS:
        # Normalize the post without contacting the embedding service.
        return parsed._replace(netloc="x.com").geturl()
    if parsed.hostname in _VK_HOSTS | _TWITTER_HOSTS:
        # Экстрактор VK не принимает www; стандартный HTTPS-порт можно опустить.
        return parsed._replace(netloc=parsed.hostname.removeprefix("www.")).geturl()
    return url


class MediaDownloader:
    """Ограничить параллелизм, время, размер и время жизни файлов."""

    def __init__(self, youtube_cookies_file: Path | None = None) -> None:
        self._slots = asyncio.Semaphore(2)
        self._youtube_cookies_file = youtube_cookies_file

    @asynccontextmanager
    async def download(self, url: str) -> AsyncIterator[DownloadedMedia]:
        url = validate_media_url(url)
        async with self._slots:
            with TemporaryDirectory(prefix="delta-video-") as directory:
                path = Path(directory)
                if is_youtube_url(url) and self._youtube_cookies_file:
                    from protogen_delta.services.download_errors import DownloadFailure

                    try:
                        await finish_operation(
                            asyncio.to_thread(
                                snapshot_youtube_cookies,
                                self._youtube_cookies_file,
                                path / "youtube-cookies.txt",
                            )
                        )
                    except DownloadFailure as error:
                        raise MediaDownloadError(ERROR_MESSAGES[error.code]) from None
                await self._run(url, path)
                candidates = [
                    item
                    for item in path.iterdir()
                    if item.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}
                    and item.is_file()
                    and not item.is_symlink()
                ]
                if len(candidates) != 1:
                    raise MediaDownloadError("Сайт не отдал один доступный видеофайл.")
                result = candidates[0]
                if result.stat().st_size == 0:
                    raise MediaDownloadError(ERROR_MESSAGES["formats"])
                if result.stat().st_size > MAX_DOWNLOAD_BYTES:
                    raise MediaDownloadError("Видео превышает лимит 100 МБ.")
                if result.stat().st_size > MAX_UPLOAD_BYTES:
                    raise MediaDownloadError(
                        "Не удалось сжать видео для отправки в Telegram."
                    )
                metadata = json.loads((path / "result.json").read_text("utf-8"))
                yield DownloadedMedia(
                    result,
                    str(metadata["title"])[:150],
                    metadata.get("compressed") is True,
                    metadata.get("animation") is True,
                )

    async def _run(self, url: str, directory: Path) -> None:
        # В дочерний процесс не передаются токены бота и провайдеров.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
        }
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        args = [
            sys.executable,
            "-m",
            "protogen_delta.services.media_download_worker",
            str(directory),
            url,
        ]
        if is_youtube_url(url) and (directory / "youtube-cookies.txt").is_file():
            args.append(str(directory / "youtube-cookies.txt"))
        start = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *args,
                env=environment,
                start_new_session=os.name == "posix",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        )
        try:
            process = await finish_operation(start)
        except asyncio.CancelledError:
            if not start.cancelled() and start.exception() is None:
                await finish_operation(_stop_worker(start.result()))
            raise
        try:
            async with asyncio.timeout(DOWNLOAD_TIMEOUT_SECONDS):
                while process.returncode is None:
                    total = directory_bytes(directory)
                    # Во время remux одновременно существуют исходники и результат.
                    if total > 2 * MAX_DOWNLOAD_BYTES + 1024 * 1024:
                        raise MediaDownloadError("Видео превышает лимит 100 МБ.")
                    try:
                        await asyncio.wait_for(process.wait(), timeout=0.2)
                    except TimeoutError:
                        continue
                if process.returncode != 0:
                    code = read_failure(directory)
                    logger.warning("Video download failed | code=%s", code)
                    raise MediaDownloadError(ERROR_MESSAGES[code])
        except TimeoutError as error:
            raise MediaDownloadError(
                "Загрузка или сжатие видео не завершились за четыре минуты."
            ) from error
        finally:
            await finish_operation(_stop_worker(process))
