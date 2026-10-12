"""Поиск публикации арта в SauceNAO, без распознавания личности по лицу."""

import asyncio
import io
import json
import math
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, FormData
from PIL import Image


class ImageSourceError(ValueError):
    """Источник не настроен, недоступен или не вернул подходящего совпадения."""


@dataclass(frozen=True, slots=True)
class ImageSourceMatch:
    similarity: float
    index: str
    url: str


def image_preview(data: bytes) -> bytes:
    """Передавать только небольшое изображение без исходных метаданных."""
    if not data or len(data) > 20 * 1024 * 1024:
        raise ImageSourceError("Картинка должна быть меньше 20 МБ.")
    try:
        with Image.open(io.BytesIO(data)) as original:
            if original.width * original.height > 40_000_000:
                raise ImageSourceError("Картинка слишком большая.")
            original.thumbnail((1024, 1024))
            image = original.convert("RGB")
            output = io.BytesIO()
            image.save(output, "JPEG", quality=85)
            return output.getvalue()
    except (OSError, ValueError) as error:
        raise ImageSourceError("Не смог прочитать изображение.") from error


def parse_matches(payload: Any) -> tuple[ImageSourceMatch, ...]:
    """Отбрасывать слабые совпадения, некорректные числа и небезопасные URL."""
    if not isinstance(payload, dict) or not isinstance(payload.get("header"), dict):
        raise ImageSourceError("Сервис поиска источника вернул ошибку.")
    status = payload["header"].get("status", 0)
    results = payload.get("results", [])
    if type(status) is not int or status < 0 or not isinstance(results, list):
        raise ImageSourceError("Сервис поиска источника вернул ошибку.")
    matches: list[ImageSourceMatch] = []
    seen: set[str] = set()
    for result in results[:10]:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("header"), dict)
            or not isinstance(result.get("data"), dict)
        ):
            continue
        try:
            score = float(result["header"]["similarity"])
            index = str(result["header"].get("index_name", "Совпадение"))[:100]
            urls = result["data"].get("ext_urls", [])
            if (
                not math.isfinite(score)
                or not 80 <= score <= 100
                or not isinstance(urls, list)
            ):
                continue
            for url in urls:
                if not isinstance(url, str) or len(url) > 1000:
                    continue
                parsed = urlsplit(url)
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                ):
                    continue
                if url not in seen:
                    matches.append(ImageSourceMatch(score, index, url))
                    seen.add(url)
                    break
        except KeyError, TypeError, ValueError:
            continue
    return tuple(sorted(matches, key=lambda item: item.similarity, reverse=True)[:3])


class ImageSourceService:
    def __init__(self, api_key: str) -> None:
        self._key = api_key
        self._slots = asyncio.Semaphore(2)

    async def search(self, data: bytes) -> tuple[ImageSourceMatch, ...]:
        if not self._key:
            raise ImageSourceError(
                "Поиск источника пока не настроен: нужен SAUCENAO_API_KEY."
            )
        preview = await asyncio.to_thread(image_preview, data)
        form = FormData()
        form.add_field("api_key", self._key)
        form.add_field("output_type", "2")
        form.add_field("numres", "5")
        form.add_field("file", preview, filename="art.jpg", content_type="image/jpeg")
        try:
            async with (
                self._slots,
                ClientSession(timeout=ClientTimeout(total=20)) as session,
            ):
                async with session.post(
                    "https://saucenao.com/search.php", data=form, allow_redirects=False
                ) as response:
                    if response.status != 200:
                        raise ImageSourceError("Сервис поиска источника недоступен.")
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        body.extend(chunk)
                        if len(body) > 1_048_576:
                            raise ImageSourceError("Ответ сервиса слишком большой.")
                    return parse_matches(json.loads(body))
        except (ClientError, TimeoutError, ValueError, TypeError) as error:
            raise ImageSourceError(
                "Поиск источника не удался. Попробуй позже."
            ) from error
