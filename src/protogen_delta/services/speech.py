"""Локальное распознавание речи через faster-whisper."""

import io
from dataclasses import dataclass
from typing import Any, Protocol

from protogen_delta.core.operation_metrics import OperationSnapshot
from protogen_delta.services.blocking_work import WorkRunner
from protogen_delta.services.native_work import NativeWorkPool
from protogen_delta.services.speech_audio import prepare_speech_audio

MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 600
MAX_TRANSCRIPT_CHARS = 30_000


class SpeechRecognitionError(RuntimeError):
    """Не удалось получить пригодную расшифровку аудио."""


@dataclass(frozen=True, slots=True)
class Transcript:
    """Распознанный текст и определённый моделью язык."""

    text: str
    language: str | None = None
    truncated: bool = False
    uncertain: bool = False


class SpeechInference(Protocol):
    async def transcribe(self, data: bytes) -> Transcript: ...
    def snapshot(self) -> OperationSnapshot: ...
    async def close(self) -> None: ...


class SpeechTranscriber:
    """Лениво загружать Whisper и выполнять распознавание вне event loop."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        native_work: WorkRunner | None = None,
        inference: SpeechInference | None = None,
    ) -> None:
        """Сохранить параметры модели без её немедленной загрузки."""
        if inference is None:
            from protogen_delta.services.whisper_process import WhisperProcess

            inference = WhisperProcess(model_size, device, compute_type)
        self._inference = inference
        self._native_work = native_work or NativeWorkPool()

    def snapshot(self) -> OperationSnapshot:
        """Показывать очередь и фактическую занятость процесса."""
        return self._inference.snapshot()

    async def close(self) -> None:
        await self._inference.close()

    async def transcribe(self, data: bytes) -> Transcript:
        """Распознать речь из Telegram-аудио и вернуть компактный текст."""
        if not data:
            raise SpeechRecognitionError("Аудиофайл пуст")
        if len(data) > MAX_AUDIO_BYTES:
            raise SpeechRecognitionError("Аудиофайл превышает 20 МБ")
        try:
            data = await self._native_work.run(prepare_speech_audio, data)
        except (ValueError, OSError) as error:
            raise SpeechRecognitionError(
                "Не удалось подготовить аудио для Whisper"
            ) from error
        return await self._inference.transcribe(data)

    @staticmethod
    def _transcribe_sync(model: Any, data: bytes) -> Transcript:
        """Выполнить блокирующий inference и собрать генератор сегментов."""
        try:
            segments, info = model.transcribe(
                io.BytesIO(data),
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            pieces: list[str] = []
            uncertain = False
            size = 0
            truncated = False
            for segment in segments:
                piece = segment.text.strip()
                if not piece:
                    continue
                pieces.append(piece)
                size += len(piece) + (1 if len(pieces) > 1 else 0)
                # Это эвристика декодера, а не калиброванная вероятность ошибки.
                score = getattr(segment, "avg_logprob", None)
                silence = getattr(segment, "no_speech_prob", None)
                uncertain |= (isinstance(score, (int, float)) and score < -1.0) or (
                    isinstance(silence, (int, float)) and silence > 0.6
                )
                if size > MAX_TRANSCRIPT_CHARS:
                    truncated = True
                    break
            text = " ".join(pieces).strip()
        except Exception as error:
            raise SpeechRecognitionError("Не удалось распознать аудио") from error
        if not text:
            raise SpeechRecognitionError("В аудио не удалось распознать речь")
        truncated = truncated or len(text) > MAX_TRANSCRIPT_CHARS
        return Transcript(
            text=text[:MAX_TRANSCRIPT_CHARS],
            language=getattr(info, "language", None),
            truncated=truncated,
            uncertain=uncertain,
        )
