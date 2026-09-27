"""Локальное распознавание речи через faster-whisper."""

import asyncio
import io
from dataclasses import dataclass
from typing import Any

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


class SpeechTranscriber:
    """Лениво загружать Whisper и выполнять распознавание вне event loop."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """Сохранить параметры модели без её немедленной загрузки."""
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any | None = None
        self._model_lock = asyncio.Lock()

    async def _ensure_model(self) -> Any:
        """Создать единственный экземпляр модели при первом голосовом сообщении."""
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
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
        model = await self._ensure_model()
        return await asyncio.to_thread(self._transcribe_sync, model, data)

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
            text = " ".join(
                segment.text.strip() for segment in segments if segment.text.strip()
            ).strip()
        except Exception as error:
            raise SpeechRecognitionError("Не удалось распознать аудио") from error
        if not text:
            raise SpeechRecognitionError("В аудио не удалось распознать речь")
        truncated = len(text) > MAX_TRANSCRIPT_CHARS
        return Transcript(
            text=text[:MAX_TRANSCRIPT_CHARS],
            language=getattr(info, "language", None),
            truncated=truncated,
        )
