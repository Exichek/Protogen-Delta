"""Дочерние парсеры с ограничением времени и остановкой дерева процессов."""

import asyncio
import base64
import json
import math
import os
import signal
import sys
import wave
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.documents import (
    DocumentReadError,
    DocumentTooLargeError,
    ExtractedDocument,
    ExtractedImage,
    UnsupportedDocumentError,
)

MAX_NATIVE_INPUT = 20 * 1024 * 1024
MAX_NATIVE_OUTPUT = 16 * 1024 * 1024
MAX_SPEECH_OUTPUT = 20 * 1024 * 1024
_OPERATIONS = {
    ("protogen_delta.services.documents", "extract_document"): "document",
    (
        "protogen_delta.services.animation_frames",
        "extract_animation_frames",
    ): "animation",
    ("protogen_delta.services.tgs_frames", "extract_tgs_frames"): "tgs",
    ("protogen_delta.services.audio_analysis", "analyze_audio"): "audio",
    ("protogen_delta.services.speech_audio", "prepare_speech_audio"): "speech",
    ("protogen_delta.services.music", "audio_tags"): "tags",
    ("protogen_delta.services.audio_understanding", "audio_excerpt"): "clip",
    (
        "protogen_delta.services.appearance_image",
        "prepare_appearance_image",
    ): "appearance",
}


class NativeWorkError(ValueError):
    """Дочерний обработчик завершился нештатно или превысил бюджет."""


def worker_environment() -> dict[str, str]:
    """Не передавать токены, прокси, пользовательский PYTHONPATH и настройки бота."""
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL"}
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    return environment


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if sys.platform != "win32":
        # OCR запускает Tesseract: убираем и потомков, даже если родитель уже вышел.
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
            try:
                process.kill()
            except ProcessLookupError:
                pass
    await process.wait()


class NativeWorkPool:
    def __init__(self, limit: int = 2, timeout: float = 90.0) -> None:
        if limit < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Нужны положительные limit и timeout")
        self._slots = asyncio.Semaphore(limit)
        self._timeout = timeout

    async def run[T](
        self, function: Callable[..., T], data: bytes, *args: Any, **kwargs: Any
    ) -> T:
        operation = _OPERATIONS.get((function.__module__, function.__name__))
        if operation is None:
            raise ValueError("Неизвестный дочерний обработчик")
        if not data:
            if operation == "document":
                raise DocumentReadError("Документ пуст")
            raise NativeWorkError("Файл пуст.")
        if len(data) > MAX_NATIVE_INPUT:
            if operation == "document":
                raise DocumentTooLargeError("Документ превышает 20 МБ")
            raise NativeWorkError("Файл превышает 20 МБ.")
        try:
            async with asyncio.timeout(self._timeout):
                async with self._slots:
                    value = await self._execute(operation, data, args, kwargs)
                    return cast(T, self._decode(operation, value))
        except TimeoutError as error:
            raise NativeWorkError(
                "Обработка файла заняла слишком много времени. Пришли меньший фрагмент."
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise NativeWorkError("Не удалось обработать этот файл.") from error

    async def _execute(
        self, operation: str, data: bytes, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        with TemporaryDirectory(prefix="delta-native-") as directory:
            maximum_output = (
                MAX_SPEECH_OUTPUT if operation == "speech" else MAX_NATIVE_OUTPUT
            )
            source = Path(directory) / "input"
            target = Path(directory) / "result.json"
            request = Path(directory) / "request.json"
            source.write_bytes(data)
            request.write_text(
                json.dumps({"operation": operation, "args": args, "kwargs": kwargs}),
                "utf-8",
            )
            start = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "protogen_delta.services.native_worker",
                    str(source),
                    str(target),
                    str(request),
                    cwd=directory,
                    env=worker_environment(),
                    start_new_session=os.name == "posix",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            )
            try:
                process = await finish_operation(start)
            except asyncio.CancelledError:
                process = start.result()
                await finish_operation(_terminate(process))
                raise
            try:
                while process.returncode is None:
                    if target.exists() and target.stat().st_size > maximum_output:
                        raise NativeWorkError(
                            "Результат обработки файла слишком большой."
                        )
                    try:
                        await asyncio.wait_for(process.wait(), 0.1)
                    except TimeoutError:
                        continue
                if (
                    process.returncode != 0
                    or not target.exists()
                    or not 0 < target.stat().st_size <= maximum_output
                ):
                    raise NativeWorkError("Не удалось обработать этот файл.")
                if operation == "speech":
                    speech_data = target.read_bytes()
                    if speech_data.startswith(b"RIFF"):
                        return {"speech_data": speech_data}
                value = json.loads(target.read_text("utf-8"))
                if not isinstance(value, dict):
                    raise NativeWorkError("Некорректный ответ обработчика файла.")
                return value
            finally:
                await finish_operation(_terminate(process))

    @staticmethod
    def _decode(operation: str, value: dict[str, Any]) -> Any:
        error = value.get("error")
        if error:
            if not isinstance(error, str):
                raise NativeWorkError("Некорректный ответ обработчика файла.")
            if operation == "document":
                errors = {
                    "too_large": DocumentTooLargeError,
                    "unsupported": UnsupportedDocumentError,
                }
                raise errors.get(error, DocumentReadError)(
                    "Не удалось прочитать документ"
                )
            if operation in {"audio", "tags", "clip", "speech"}:
                from protogen_delta.services.audio_analysis import AudioAnalysisError

                raise AudioAnalysisError("Не удалось разобрать аудио")
            raise NativeWorkError("Не удалось прочитать этот файл.")
        try:
            if operation == "speech":
                import io

                data = value["speech_data"]
                if not isinstance(data, bytes) or len(data) > MAX_SPEECH_OUTPUT:
                    raise ValueError("speech_size")
                with wave.open(io.BytesIO(data), "rb") as wav:
                    if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                        1,
                        2,
                        16000,
                    ) or not 0 < wav.getnframes() <= 600 * 16000:
                        raise ValueError("speech_format")
                    if len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
                        raise ValueError("speech_frames")
                return data
            if operation == "tags":
                tags = value["tags"]
                if (
                    not isinstance(tags, dict)
                    or not set(tags).issubset(
                        {"title", "artist", "album", "genre", "date"}
                    )
                    or not all(
                        isinstance(v, str) and len(v) <= 200 for v in tags.values()
                    )
                ):
                    raise ValueError("tags")
                return tags
            if operation == "clip":
                clip = base64.b64decode(value["clip"], validate=True)
                if not 0 < len(clip) <= 2 * 1024 * 1024:
                    raise ValueError("clip")
                return clip
            if operation == "audio":
                from protogen_delta.services.audio_analysis import AudioAnalysis

                analysis = value["analysis"]
                if not isinstance(analysis, dict) or not all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                    for v in analysis.values()
                ):
                    raise ValueError("analysis")
                return AudioAnalysis(**analysis)
            raw_images = value.get("images", [])
            if not isinstance(raw_images, list) or len(raw_images) > 4:
                raise ValueError("images")
            if any(
                not isinstance(item, dict)
                or not all(
                    isinstance(item.get(key), str)
                    for key in ("data", "mime_type", "label")
                )
                for item in raw_images
            ):
                raise ValueError("image_fields")
            images = tuple(
                ImageInput(
                    base64.b64decode(item["data"], validate=True),
                    item["mime_type"],
                    item["label"],
                )
                for item in raw_images
            )
            if len(images) > 4 or any(
                i.mime_type not in {"image/png", "image/jpeg", "image/webp"}
                or not 0 < len(i.data) <= 4 * 1024 * 1024
                or len(i.label) > 1000
                for i in images
            ):
                raise ValueError("images")
            if operation == "document":
                if (
                    not isinstance(value.get("text"), str)
                    or len(value["text"]) > 60000
                    or not isinstance(value.get("kind"), str)
                    or type(value.get("truncated")) is not bool
                ):
                    raise ValueError("document")
                return ExtractedDocument(
                    value["text"],
                    value["kind"],
                    value["truncated"],
                    tuple(ExtractedImage(i.data, i.mime_type, i.label) for i in images),
                )
            if operation == "appearance":
                if len(images) != 1:
                    raise ValueError("appearance")
                return images[0]
            return images
        except (KeyError, TypeError, ValueError, wave.Error) as error:
            raise NativeWorkError("Некорректный ответ обработчика файла.") from error
