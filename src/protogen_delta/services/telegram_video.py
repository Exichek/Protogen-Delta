"""Ограниченное преобразование локального WebM в воспроизводимый Telegram MP4."""

import asyncio
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from protogen_delta.services.e621 import MAX_VIDEO_BYTES, E621Error


class TelegramVideoConverter:
    """Изолировать декодер и ограничить нагрузку одним процессом."""

    def __init__(self, timeout: float = 120) -> None:
        self._timeout = timeout
        self._lock = asyncio.Semaphore(1)

    async def convert(self, data: bytes) -> bytes:
        if not data or len(data) > MAX_VIDEO_BYTES:
            raise E621Error("Видео превышает лимит конвертации.")
        async with self._lock:
            with TemporaryDirectory(prefix="delta-video-") as directory:
                source = Path(directory) / "source.webm"
                target = Path(directory) / "video.mp4"
                await asyncio.to_thread(source.write_bytes, data)
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
                    "protogen_delta.services.telegram_video_worker",
                    str(source),
                    str(target),
                    env=environment,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    async with asyncio.timeout(self._timeout):
                        while process.returncode is None:
                            if (
                                target.exists()
                                and target.stat().st_size > MAX_VIDEO_BYTES
                            ):
                                raise E621Error("MP4 превышает лимит отправки.")
                            try:
                                await asyncio.wait_for(process.wait(), 0.2)
                            except TimeoutError:
                                continue
                        if process.returncode != 0 or not target.exists():
                            raise E621Error("Не удалось подготовить MP4.")
                        if not 0 < target.stat().st_size <= MAX_VIDEO_BYTES:
                            raise E621Error("MP4 превышает лимит отправки.")
                        return await asyncio.to_thread(target.read_bytes)
                except TimeoutError as error:
                    raise E621Error(
                        "Конвертация видео заняла слишком много времени."
                    ) from error
                finally:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
