"""Один переиспользуемый Whisper-процесс, остановка до освобождения слота."""

import asyncio
import json
import math
import os
import sys
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.operation_metrics import OperationMetrics, OperationSnapshot
from protogen_delta.services.native_work import _terminate, worker_environment
from protogen_delta.services.speech import (
    MAX_AUDIO_BYTES,
    MAX_TRANSCRIPT_CHARS,
    SpeechRecognitionError,
    Transcript,
)

MAX_WHISPER_REPLY = 128 * 1024


class WhisperProcess:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        *,
        timeout: float = 110,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout должен быть положительным и конечным")
        if any(
            not isinstance(value, str) or not 0 < len(value) <= 1024
            for value in (model_size, device, compute_type)
        ):
            raise ValueError("Некорректные настройки Whisper")
        self._config = {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
        }
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._metrics = OperationMetrics(capacity=1)
        self._process: asyncio.subprocess.Process | None = None
        self._directory: TemporaryDirectory[str] | None = None
        self._sequence = 0
        self._active_task: asyncio.Task[Any] | None = None
        self._closed = False

    def snapshot(self) -> OperationSnapshot:
        return self._metrics.snapshot()

    async def transcribe(self, data: bytes) -> Transcript:
        if not data or len(data) > MAX_AUDIO_BYTES:
            raise SpeechRecognitionError("Некорректный размер PCM-аудио")
        try:
            with self._metrics.measure(queued=True) as measurement:
                async with asyncio.timeout(self._timeout):
                    async with self._lock:
                        measurement.start()
                        if self._closed:
                            raise SpeechRecognitionError("Распознавание остановлено")
                        self._active_task = asyncio.current_task()
                        try:
                            return await self._exchange(data)
                        finally:
                            self._active_task = None
        except TimeoutError as error:
            raise SpeechRecognitionError(
                "Распознавание заняло слишком много времени"
            ) from error
        except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as error:
            raise SpeechRecognitionError(
                "Не удалось связаться с процессом Whisper"
            ) from error

    async def _start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        await finish_operation(self._shutdown())
        self._directory = TemporaryDirectory(prefix="delta-whisper-")
        directory = Path(self._directory.name)
        config = directory / "config.json"
        config.write_text(json.dumps(self._config), "utf-8")
        environment = worker_environment()
        environment["HF_HOME"] = os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        start = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "protogen_delta.services.whisper_worker",
                str(config),
                str(directory),
                cwd=directory,
                env=environment,
                start_new_session=os.name == "posix",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=MAX_WHISPER_REPLY,
            )
        )
        try:
            self._process = await finish_operation(start)
        except asyncio.CancelledError:
            if not start.cancelled() and start.exception() is None:
                self._process = start.result()
            raise

    async def _exchange(self, data: bytes) -> Transcript:
        try:
            await self._start()
            assert self._directory is not None and self._process is not None
            assert self._process.stdin is not None and self._process.stdout is not None
            source = Path(self._directory.name) / "audio.wav"
            source.write_bytes(data)
            self._sequence += 1
            self._process.stdin.write(
                (json.dumps({"sequence": self._sequence}) + "\n").encode()
            )
            await self._process.stdin.drain()
            reply = await self._process.stdout.readline()
            if not reply or len(reply) > MAX_WHISPER_REPLY:
                raise SpeechRecognitionError("Некорректный ответ Whisper")
            value = self._decode(reply, self._sequence)
            source.unlink(missing_ok=True)
        except BaseException:
            await finish_operation(self._shutdown())
            raise
        if isinstance(value, str):
            errors = {
                "load": "Не удалось загрузить модель Whisper",
                "no_speech": "В аудио не удалось распознать речь",
                "recognition": "Не удалось распознать аудио",
            }
            raise SpeechRecognitionError(errors[value])
        return value

    @staticmethod
    def _decode(reply: bytes, sequence: int) -> Transcript | str:
        value = json.loads(reply)
        if (
            not isinstance(value, dict)
            or type(value.get("sequence")) is not int
            or value["sequence"] != sequence
        ):
            raise SpeechRecognitionError("Некорректный ответ Whisper")
        if set(value) == {"sequence", "error"}:
            if not isinstance(value["error"], str) or value["error"] not in {
                "load",
                "no_speech",
                "recognition",
            }:
                raise SpeechRecognitionError("Некорректный ответ Whisper")
            return str(value["error"])
        if (
            set(value) != {"sequence", "text", "language", "truncated", "uncertain"}
            or not isinstance(value["text"], str)
            or not value["text"].strip()
            or len(value["text"]) > MAX_TRANSCRIPT_CHARS
            or type(value["truncated"]) is not bool
            or type(value["uncertain"]) is not bool
            or value["language"] is not None
            and (
                not isinstance(value["language"], str)
                or not 1 <= len(value["language"]) <= 12
            )
        ):
            raise SpeechRecognitionError("Некорректный ответ Whisper")
        return Transcript(
            value["text"], value["language"], value["truncated"], value["uncertain"]
        )

    async def _shutdown(self) -> None:
        process, self._process = self._process, None
        try:
            if process is not None:
                if process.returncode is None:
                    await _terminate(process)
                else:
                    await process.wait()
                if process.stdin is not None:
                    process.stdin.close()
                    with suppress(BrokenPipeError, ConnectionResetError):
                        await process.stdin.wait_closed()
        finally:
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None

    async def close(self) -> None:
        await finish_operation(self._close())

    async def _close(self) -> None:
        self._closed = True
        active = self._active_task
        if active is not None:
            active.cancel()
            await asyncio.gather(active, return_exceptions=True)
        async with self._lock:
            await self._shutdown()
