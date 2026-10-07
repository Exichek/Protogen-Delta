"""Изолированная загрузка одного публичного видео по явной команде."""

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

MAX_DOWNLOAD_BYTES = 45 * 1024 * 1024
MAX_DOWNLOAD_SECONDS = 600
DOWNLOAD_TIMEOUT_SECONDS = 120
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
)


class MediaDownloadError(ValueError):
    """Ссылка или доступное видео не соответствуют ограничениям загрузки."""


@dataclass(frozen=True, slots=True)
class DownloadedMedia:
    """Временный файл, доступный только внутри контекстного менеджера."""

    path: Path
    title: str


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
        )
    except ValueError:
        valid = False
    if not valid:
        raise MediaDownloadError(
            "Нужна HTTPS-ссылка на одно публичное видео YouTube, Instagram, "
            "TikTok, Vimeo или VK/VK Видео."
        )
    if parsed.hostname in _VK_HOSTS:
        # Экстрактор VK не принимает www; стандартный HTTPS-порт можно опустить.
        return parsed._replace(netloc=parsed.hostname.removeprefix("www.")).geturl()
    return url


class MediaDownloader:
    """Ограничить параллелизм, время, размер и время жизни файлов."""

    def __init__(self) -> None:
        self._slots = asyncio.Semaphore(2)

    @asynccontextmanager
    async def download(self, url: str) -> AsyncIterator[DownloadedMedia]:
        url = validate_media_url(url)
        async with self._slots:
            with TemporaryDirectory(prefix="delta-video-") as directory:
                path = Path(directory)
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
                if not 0 < result.stat().st_size <= MAX_DOWNLOAD_BYTES:
                    raise MediaDownloadError("Видео превышает лимит 45 МБ.")
                title = json.loads((path / "result.json").read_text("utf-8"))["title"]
                yield DownloadedMedia(result, str(title)[:150])

    async def _run(self, url: str, directory: Path) -> None:
        # В дочерний процесс не передаются токены бота и провайдеров.
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
        }
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "protogen_delta.services.media_download_worker",
            str(directory),
            url,
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(DOWNLOAD_TIMEOUT_SECONDS):
                while process.returncode is None:
                    total = sum(
                        item.stat().st_size
                        for item in directory.rglob("*")
                        if item.is_file()
                    )
                    # Во время remux одновременно существуют исходники и результат.
                    if total > 2 * MAX_DOWNLOAD_BYTES + 1024 * 1024:
                        raise MediaDownloadError("Видео превышает лимит 45 МБ.")
                    try:
                        await asyncio.wait_for(process.wait(), timeout=0.2)
                    except TimeoutError:
                        continue
                if process.returncode != 0:
                    raise MediaDownloadError(
                        "Не смог скачать видео: оно недоступно, требует входа, "
                        "превышает 10 минут/45 МБ или сайт ограничил загрузку."
                    )
        except TimeoutError as error:
            raise MediaDownloadError(
                "Сайт не успел отдать видео за две минуты."
            ) from error
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
