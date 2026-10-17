"""Ограниченное распознавание через закреплённый ShazamIO, без ключа AudD."""

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from shazamio import Shazam
from shazamio.interfaces.client import HTTPClientInterface

from protogen_delta.services.audio_understanding import audio_excerpt
from protogen_delta.services.blocking_work import WorkRunner
from protogen_delta.services.music import (
    MusicRecognitionError,
    MusicRecognitionService,
    MusicRecognizer,
)
from protogen_delta.services.native_work import NativeWorkPool


class _ShazamHTTPClient(HTTPClientInterface):
    """Один запрос без повторов, редиректов и неограниченного ответа SDK."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        target = urlsplit(url)
        if (
            method != "POST"
            or target.scheme != "https"
            or target.hostname not in {"amp.shazam.com", "www.shazam.com"}
            or target.port not in {None, 443}
            or target.username is not None
        ):
            raise MusicRecognitionError("Недопустимый адрес распознавания")
        async with self._session.post(url, allow_redirects=False, **kwargs) as response:
            if response.status != 200:
                raise MusicRecognitionError("Shazam недоступен")
            body = bytearray()
            async for chunk in response.content.iter_chunked(16 * 1024):
                body.extend(chunk)
                if len(body) > 128 * 1024:
                    raise MusicRecognitionError("Ответ Shazam слишком большой")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise MusicRecognitionError("Некорректный ответ Shazam")
        return payload

    async def request_text(self, url: str, *, content_type: str, **kwargs: Any) -> str:
        raise MusicRecognitionError("Текстовый поиск не используется для распознавания")


def shazam_report(payload: dict[str, Any]) -> str | None:
    """Передавать только поля реального совпадения; название не берётся из запроса."""
    matches = payload.get("matches")
    if matches == []:
        return None
    if not isinstance(matches, list) or not matches:
        raise MusicRecognitionError("Ответ Shazam не содержит результата поиска")
    track = payload.get("track")
    if not isinstance(track, dict) or any(
        not isinstance(track.get(name), str) or not track[name].strip()
        for name in ("title", "subtitle")
    ):
        raise MusicRecognitionError("Неполное совпадение Shazam")
    values = {
        "provider": "Shazam",
        "title": " ".join(track["title"].split())[:200],
        "artist": " ".join(track["subtitle"].split())[:200],
    }
    sections = track.get("sections")
    if isinstance(sections, list):
        for section in sections[:8]:
            if not isinstance(section, dict) or section.get("type") != "SONG":
                continue
            metadata = section.get("metadata")
            if not isinstance(metadata, list):
                continue
            for item in metadata[:20]:
                if not isinstance(item, dict):
                    continue
                label = item.get("title")
                key = (
                    {"Album": "album", "Released": "release_date"}.get(label)
                    if isinstance(label, str)
                    else None
                )
                value = item.get("text")
                if key and isinstance(value, str) and value.strip():
                    values[key] = " ".join(value.split())[:200]
    return json.dumps(values, ensure_ascii=False)


class ShazamMusicRecognitionService:
    """Первые 12 секунд WAV, максимум два запроса одновременно и 20 секунд сети."""

    provider_name = "Shazam"

    def __init__(self, *, native_work: WorkRunner | None = None) -> None:
        self._native_work = native_work or NativeWorkPool(2)
        self._slots = asyncio.Semaphore(2)

    async def recognize(self, data: bytes) -> str | None:
        async with self._slots:
            try:
                excerpt = await self._native_work.run(audio_excerpt, data, seconds=12)
                async with asyncio.timeout(20):
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=20)
                    ) as session:
                        async with Shazam(
                            http_client=_ShazamHTTPClient(session)
                        ) as shazam:
                            payload = await shazam.recognize(excerpt, max_skew=1e-3)
                return shazam_report(payload)
            except MusicRecognitionError:
                raise
            except Exception as error:
                raise MusicRecognitionError(
                    "Не удалось распознать трек через Shazam"
                ) from error


def create_music_recognizer(
    provider: str, api_token: str | None, native_work: WorkRunner
) -> MusicRecognizer | None:
    """Не переключаться на платный сервис при сбое бесплатного провайдера."""
    if provider == "shazam":
        return ShazamMusicRecognitionService(native_work=native_work)
    if provider == "audd" and api_token:
        return MusicRecognitionService(api_token, native_work=native_work)
    if provider == "off":
        return None
    raise ValueError(
        "Для AudD нужен MUSIC_AUDD_API_TOKEN; провайдер — shazam, audd или off"
    )
