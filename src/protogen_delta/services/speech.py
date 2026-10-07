"""Локальное распознавание речи через faster-whisper."""

import asyncio
import io
from dataclasses import dataclass
from typing import Any

from protogen_delta.services.blocking_work import BlockingWorkPool, WorkRunner
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


class SpeechTranscriber:
    """Лениво загружать Whisper и выполнять распознавание вне event loop."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        native_work: WorkRunner | None = None,
    ) -> None:
        """Сохранить параметры модели без её немедленной загрузки."""
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any | None = None
        self._model_lock = asyncio.Lock()
        self._work = BlockingWorkPool()
        self._native_work = native_work

    async def _ensure_model(self) -> Any:
        """Создать единственный экземпляр модели при первом голосовом сообщении."""
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                self._model = await self._work.run(self._load_and_cache_model)
        return self._model

    def _load_and_cache_model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Импортировать faster-whisper и загрузить выбранную модель."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            return WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as error:
            raise SpeechRecognitionError(
                "Не удалось загрузить модель Whisper"
            ) from error

    async def transcribe(self, data: bytes) -> Transcript:
        """Распознать речь из Telegram-аудио и вернуть компактный текст."""
        if not data:
            raise SpeechRecognitionError("Аудиофайл пуст")
        if len(data) > MAX_AUDIO_BYTES:
            raise SpeechRecognitionError("Аудиофайл превышает 20 МБ")
        if self._native_work is not None:
            try:
                data = await self._native_work.run(prepare_speech_audio, data)
            except (ValueError, OSError) as error:
                raise SpeechRecognitionError(
                    "Не удалось подготовить аудио для Whisper"
                ) from error
        model = await self._ensure_model()
        return await self._work.run(self._transcribe_sync, model, data)

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
