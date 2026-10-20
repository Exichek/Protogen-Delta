"""Конвертер Mini App: ограниченные процессы и временные результаты для просмотра."""

import asyncio
import json
import math
import os
import re
import secrets
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.operation_metrics import OperationMetrics, OperationSnapshot
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.services.media_conversion_worker import (
    MAX_INPUT,
    MAX_OUTPUT,
    validate_options,
)
from protogen_delta.services.native_work import _terminate, worker_environment

RESULT_LIFETIME = 15 * 60
MIME_TYPES = {
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
}


class ConversionError(ValueError):
    """Безопасная ошибка пользовательского файла."""


class ConversionBusyError(ConversionError):
    """Заняты обработчик или ограниченный временный кэш."""


@dataclass
class ConvertedMedia:
    user_id: int
    token: str
    directory: TemporaryDirectory[str]
    expires_at: float
    name: str
    extension: str
    preview: Path
    duration: float
    size: int

    @property
    def path(self) -> Path:
        return Path(self.directory.name) / ("result." + self.extension)


class MediaConversionService:
    def __init__(
        self, *, timeout: float = 240, clock: Callable[[], float] = monotonic
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout должен быть положительным")
        self._timeout, self._clock = timeout, clock
        self._slots = asyncio.Semaphore(1)
        self._pending: set[int] = set()
        self._jobs: dict[str, ConvertedMedia] = {}
        self._limiter = UserRateLimiter(cooldown_seconds=10)
        self._metrics = OperationMetrics(capacity=1)
        self._cleanup_task: asyncio.Task[None] | None = None
        self._active: set[asyncio.Task[Any]] = set()

    def snapshot(self) -> OperationSnapshot:
        return self._metrics.snapshot()

    async def start(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self.prune()

    async def prune(self) -> None:
        expired = [
            token
            for token, job in self._jobs.items()
            if job.expires_at <= self._clock()
        ]
        for token in expired:
            await self.remove(token)

    async def remove(self, token: str, user_id: int | None = None) -> bool:
        job = self._jobs.get(token)
        if job is None or (user_id is not None and job.user_id != user_id):
            return False
        self._jobs.pop(token)
        await finish_operation(asyncio.to_thread(job.directory.cleanup))
        return True

    def get(self, token: str) -> ConvertedMedia | None:
        job = self._jobs.get(token)
        return job if job and job.expires_at > self._clock() else None

    async def close(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
            self._cleanup_task = None
        tasks = list(self._active)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for token in list(self._jobs):
            await self.remove(token)

    async def convert(
        self,
        user_id: int,
        read_file: Callable[[], Awaitable[bytes]],
        *,
        name: str,
        mode: str,
        extension: str,
        resolution: str = "720",
        bitrate: int = 192,
    ) -> ConvertedMedia:
        validate_options(mode, extension, resolution, bitrate)
        if user_id in self._pending:
            raise ConversionBusyError("Предыдущий файл ещё преобразуется.")
        self._pending.add(user_id)
        acquired = False
        task = asyncio.current_task()
        if task:
            self._active.add(task)
        directory: TemporaryDirectory[str] | None = None
        try:
            with self._metrics.measure(queued=True) as measurement:
                try:
                    await asyncio.wait_for(self._slots.acquire(), timeout=0.1)
                    acquired = True
                except TimeoutError as error:
                    raise ConversionBusyError(
                        "Конвертер занят. Попробуй чуть позже."
                    ) from error
                if not self._limiter.allow(user_id):
                    raise ConversionBusyError(
                        "Подожди 10 секунд перед следующим файлом."
                    )
                await self.prune()
                for token, job in list(self._jobs.items()):
                    if job.user_id == user_id:
                        await self.remove(token)
                if len(self._jobs) >= 4:
                    raise ConversionBusyError(
                        "Временное хранилище занято. Попробуй через несколько минут."
                    )
                measurement.start()
                async with asyncio.timeout(self._timeout):
                    data = await read_file()
                    if not 0 < len(data) <= MAX_INPUT:
                        raise ConversionError(
                            "Файл должен быть непустым и не больше 100 МБ."
                        )
                    directory = TemporaryDirectory(prefix="delta-converter-")
                    root = Path(directory.name)
                    await finish_operation(
                        asyncio.to_thread((root / "input").write_bytes, data)
                    )
                    del data
                    options = {
                        "mode": mode,
                        "format": extension,
                        "resolution": resolution,
                        "bitrate": bitrate,
                    }
                    (root / "options.json").write_text(json.dumps(options), "utf-8")
                    await self._execute(root)
                    metadata = json.loads((root / "result.json").read_text("utf-8"))
                    (root / "input").unlink()
                    (root / "options.json").unlink()
                    preview_extension = "mp4" if mode == "video" else "mp3"
                    preview = root / ("preview." + preview_extension)
                    if not preview.exists():
                        preview = root / ("result." + extension)
                    base = (
                        re.sub(
                            r"[^\w .-]",
                            "_",
                            name.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0],
                        )[:60].strip(" .")
                        or "результат"
                    )
                    token = secrets.token_urlsafe(32)
                    job = ConvertedMedia(
                        user_id,
                        token,
                        directory,
                        self._clock() + RESULT_LIFETIME,
                        base + "." + extension,
                        extension,
                        preview,
                        float(metadata["duration"]),
                        (root / ("result." + extension)).stat().st_size,
                    )
                    self._jobs[token] = job
                    directory = None
                    return job
        except TimeoutError as error:
            raise ConversionError(
                "Преобразование заняло больше четырёх минут. Выбери меньший файл или разрешение."
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise ConversionError("Не удалось обработать файл.") from error
        finally:
            self._pending.discard(user_id)
            if task:
                self._active.discard(task)
            if acquired:
                self._slots.release()
            if directory:
                await finish_operation(asyncio.to_thread(directory.cleanup))

    async def _execute(self, directory: Path) -> None:
        start = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "protogen_delta.services.media_conversion_worker",
                str(directory),
                env=worker_environment(),
                start_new_session=os.name == "posix",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        )
        try:
            process = await finish_operation(start)
        except asyncio.CancelledError:
            if not start.cancelled() and start.exception() is None:
                await finish_operation(_terminate(start.result()))
            raise
        try:
            while process.returncode is None:
                if any(
                    path.stat().st_size > MAX_OUTPUT
                    for path in directory.iterdir()
                    if path.name.startswith(("result.", "preview."))
                ):
                    raise ConversionError(
                        "Результат превышает 150 МБ. Выбери меньший файл."
                    )
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.2)
                except TimeoutError:
                    continue
            result = directory / "result.json"
            if (
                process.returncode != 0
                or not result.is_file()
                or result.stat().st_size > 4096
            ):
                raise ConversionError(
                    "Не удалось преобразовать файл. Проверь формат, длительность и размер."
                )
        finally:
            await finish_operation(_terminate(process))
