"""Независимые этапы аудио с отдельными бюджетами и общей остановкой."""

import asyncio
import logging
import math
from collections.abc import Awaitable
from dataclasses import dataclass

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.services.audio_analysis import AudioAnalysis, analyze_audio
from protogen_delta.services.audio_understanding import AudioUnderstandingService
from protogen_delta.services.blocking_work import WorkRunner
from protogen_delta.services.music import MusicRecognitionService, audio_tags
from protogen_delta.services.speech import SpeechTranscriber, Transcript

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioResult:
    transcript: Transcript | None = None
    analysis: AudioAnalysis | None = None
    semantic_report: str | None = None
    music_report: str | None = None
    metadata: dict[str, str] | None = None


class AudioPipeline:
    def __init__(
        self,
        transcriber: SpeechTranscriber,
        work: WorkRunner,
        understanding: AudioUnderstandingService | None = None,
        recognition: MusicRecognitionService | None = None,
        *,
        timeout: float = 130.0,
        speech_timeout: float = 120.0,
    ) -> None:
        if any(
            not math.isfinite(value) or value <= 0
            for value in (timeout, speech_timeout)
        ):
            raise ValueError("Бюджеты аудио должны быть положительными и конечными")
        self._transcriber, self._work = transcriber, work
        self._understanding, self._recognition = understanding, recognition
        self._timeout, self._speech_timeout = timeout, speech_timeout

    @staticmethod
    async def _stage[T](name: str, operation: Awaitable[T], timeout: float) -> T | None:
        try:
            async with asyncio.timeout(timeout):
                return await operation
        except Exception:
            # Сбой одного сервиса не отменяет остальные; содержимое и ключи не логируются.
            logger.info("Audio stage unavailable stage=%s", name)
            return None

    async def collect(self, data: bytes, *, uploaded: bool) -> AudioResult:
        speech = asyncio.create_task(
            self._stage(
                "speech", self._transcriber.transcribe(data), self._speech_timeout
            )
        )
        tags = (
            asyncio.create_task(
                self._stage("tags", self._work.run(audio_tags, data), 10)
            )
            if uploaded
            else None
        )
        metrics = (
            asyncio.create_task(
                self._stage("metrics", self._work.run(analyze_audio, data), 30)
            )
            if uploaded
            else None
        )
        semantic = (
            asyncio.create_task(
                self._stage("understanding", self._understanding.analyze(data), 60)
            )
            if uploaded and self._understanding
            else None
        )
        recognition = (
            asyncio.create_task(
                self._stage("recognition", self._recognition.recognize(data), 40)
            )
            if uploaded and self._recognition
            else None
        )
        tasks = [
            task
            for task in (speech, tags, metrics, semantic, recognition)
            if task is not None
        ]
        try:
            async with asyncio.timeout(self._timeout):
                await asyncio.gather(*tasks)
        except TimeoutError:
            logger.info("Audio pipeline budget exceeded")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await finish_operation(asyncio.gather(*tasks, return_exceptions=True))

        def result[T](task: asyncio.Task[T] | None) -> T | None:
            return task.result() if task and not task.cancelled() else None

        return AudioResult(
            result(speech),
            result(metrics),
            result(semantic),
            result(recognition),
            result(tags),
        )
