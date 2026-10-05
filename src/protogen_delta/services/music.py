"""Метаданные записи и необязательное распознавание трека через AudD."""

import asyncio
import io
import json

import aiohttp
import av

from protogen_delta.services.audio_analysis import AudioAnalysisError
from protogen_delta.services.audio_understanding import audio_excerpt


def is_audio_file(mime_type: str | None, filename: str | None) -> bool:
    """Учитывать отправленную документом музыку до обычного чтения документов."""
    return bool(
        (mime_type and mime_type.startswith("audio/"))
        or (
            filename
            and filename.casefold().endswith(
                (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus")
            )
        )
    )


def audio_tags(data: bytes) -> dict[str, str]:
    """Читать теги из файла, не считать их подтверждённым распознаванием музыки."""
    if not data or len(data) > 20 * 1024 * 1024:
        return {}
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            metadata = dict(container.metadata)
            for stream in container.streams.audio:
                metadata.update(stream.metadata)
            lowered = {key.casefold(): value for key, value in metadata.items()}
            return {
                key: " ".join(lowered[key].split())[:200]
                for key in ("title", "artist", "album", "genre", "date")
                if isinstance(lowered.get(key), str) and lowered[key].strip()
            }
    except Exception:
        return {}


class MusicRecognitionError(RuntimeError):
    """Распознавание недоступно; разбор самого файла может продолжиться."""


class MusicRecognitionService:
    """Отправлять в AudD 12 секунд WAV, без истории чата и Telegram-ссылок."""

    def __init__(self, api_token: str) -> None:
        self._api_token = api_token
        self._slots = asyncio.Semaphore(2)

    async def recognize(self, data: bytes) -> str | None:
        async with self._slots:
            try:
                excerpt = await asyncio.to_thread(audio_excerpt, data, seconds=12)
                form = aiohttp.FormData()
                form.add_field("api_token", self._api_token)
                form.add_field(
                    "file", excerpt, filename="excerpt.wav", content_type="audio/wav"
                )
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as session:
                    async with session.post(
                        "https://api.audd.io/", data=form
                    ) as response:
                        if response.status != 200:
                            raise MusicRecognitionError(
                                "Сервис распознавания недоступен"
                            )
                        body = bytearray()
                        async for chunk in response.content.iter_chunked(16 * 1024):
                            body.extend(chunk)
                            if len(body) > 128 * 1024:
                                raise MusicRecognitionError(
                                    "Ответ распознавания слишком большой"
                                )
                payload = json.loads(body)
                if not isinstance(payload, dict) or payload.get("status") != "success":
                    raise MusicRecognitionError("Сервис не выполнил распознавание")
                result = payload.get("result")
                if result is None:
                    return None
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("artist"), str)
                    or not isinstance(result.get("title"), str)
                ):
                    raise MusicRecognitionError("Неполный ответ распознавания")
                values = {
                    key: result[key][:200]
                    for key in ("title", "artist", "album", "release_date")
                    if isinstance(result.get(key), str)
                }
                return json.dumps(values, ensure_ascii=False)
            except (
                aiohttp.ClientError,
                TimeoutError,
                AudioAnalysisError,
                ValueError,
            ) as error:
                raise MusicRecognitionError("Не удалось распознать трек") from error
