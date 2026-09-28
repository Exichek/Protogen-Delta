"""Минимальный клиент публичного поиска e621/e926."""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any

from aiohttp import ClientSession, ClientTimeout
from aiohttp_socks import ProxyConnector

from protogen_delta.core.user_state import ContentMode

E621_BASE_URL = "https://e621.net"
E926_BASE_URL = "https://e926.net"
BLOCKED_AGE_TAGS = frozenset({"cub", "loli", "shota", "young", "underage"})
_QUERY_META_PREFIXES = (
    "order:",
    "rating:",
    "score:",
    "favcount:",
    "date:",
    "id:",
    "user:",
    "status:",
    "type:",
)
_MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Telegram Bot API принимает отправляемые ботом видео размером до 50 МБ.
# Оставляем небольшой запас на границе ограничения.
MAX_VIDEO_BYTES = 49 * 1024 * 1024


class E621Error(RuntimeError):
    """Базовая ошибка поиска или загрузки e621."""


class E621QueryError(E621Error):
    """Пользовательский запрос небезопасен или некорректен."""


@dataclass(frozen=True, slots=True)
class E621Post:
    """Нужные боту поля одного поста e621."""

    post_id: int
    rating: str
    file_url: str | None
    file_ext: str
    file_size: int
    sample_url: str | None
    preview_url: str | None
    fav_count: int
    score: int
    artists: tuple[str, ...]
    tags: frozenset[str]
    sources: tuple[str, ...]

    @property
    def page_url(self) -> str:
        """Вернуть постоянную ссылку на страницу поста."""
        return f"{E621_BASE_URL}/posts/{self.post_id}"

    @property
    def media_url(self) -> str | None:
        """Выбрать оригинал в пределах Telegram-лимита, затем превью."""
        limit = (
            MAX_VIDEO_BYTES
            if self.file_ext.casefold() in {"mp4", "webm"}
            else MAX_IMAGE_BYTES
        )
        if self.file_size <= limit and self.file_url:
            return self.file_url
        return self.sample_url or self.preview_url


@dataclass(frozen=True, slots=True)
class E621Query:
    """Нормализованный запрос и выбранный API-хост."""

    tags: str
    base_url: str


def looks_like_e621_query(text: str | None) -> bool:
    """Распознать явный tag-query, не перехватывая обычную переписку."""
    if not text or text.startswith("/") or len(text) > 300:
        return False
    tokens = text.casefold().split()
    return len(tokens) >= 2 and any(
        token.startswith(_QUERY_META_PREFIXES) for token in tokens
    )


def normalize_e621_query(raw: str, content_mode: ContentMode) -> E621Query:
    """Проверить теги и принудительно применить возрастные ограничения."""
    clean = " ".join(raw.split())
    if not clean:
        raise E621QueryError("После /e6 укажи хотя бы один тег.")
    if len(clean) > 300 or len(clean.split()) > 40:
        raise E621QueryError("Запрос слишком длинный: оставь до 40 тегов.")
    if any(ord(char) < 32 for char in clean) or re.search(r"https?://", clean, re.I):
        raise E621QueryError("Здесь нужны теги e621, а не ссылка.")

    tokens = clean.split()
    lowered = [token.casefold() for token in tokens]
    positive = {token for token in lowered if not token.startswith("-")}
    if positive & BLOCKED_AGE_TAGS:
        raise E621QueryError(
            "Поиск сексуализированного контента с несовершеннолетними недоступен."
        )
    if any(token.startswith("limit:") for token in lowered):
        raise E621QueryError("Лимит результатов задаёт бот; убери тег limit:.")
    if any(token in {"status:any", "status:deleted"} for token in lowered):
        raise E621QueryError("Удалённые посты через бота не ищутся.")

    rating_tokens = [
        token for token in tokens if token.casefold().startswith("rating:")
    ]
    if any(
        token.casefold() not in {"rating:s", "rating:q", "rating:e"}
        for token in rating_tokens
    ):
        raise E621QueryError("Допустимые рейтинги: rating:s, rating:q или rating:e.")
    if content_mode != "adult":
        tokens = [
            token for token in tokens if not token.casefold().startswith("rating:")
        ]
        tokens.append("rating:s")
        return E621Query(" ".join(tokens), E926_BASE_URL)
    return E621Query(" ".join(tokens), E621_BASE_URL)


class E621Client:
    """Искать посты и загружать медиа с обязательным User-Agent."""

    def __init__(
        self,
        user_agent: str,
        *,
        proxy_url: str | None = None,
        request_interval: float = 1.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("E621_USER_AGENT не может быть пустым")
        if request_interval < 0:
            raise ValueError("request_interval не может быть отрицательным")
        self._user_agent = user_agent.strip()
        self._proxy_url = proxy_url
        self._request_interval = request_interval
        self._clock = clock
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    def _session(self, timeout: float) -> ClientSession:
        connector = (
            ProxyConnector.from_url(self._proxy_url) if self._proxy_url else None
        )
        return ClientSession(connector=connector, timeout=ClientTimeout(total=timeout))

    async def _throttle(self) -> None:
        async with self._lock:
            wait = self._request_interval - (self._clock() - self._last_request)
            if self._last_request and wait > 0:
                await asyncio.sleep(wait)
            self._last_request = self._clock()

    async def search(
        self, query: E621Query, *, page: int = 1, limit: int = 20
    ) -> list[E621Post]:
        """Получить одну страницу результатов."""
        await self._throttle()
        try:
            async with self._session(15) as session:
                async with session.get(
                    f"{query.base_url}/posts.json",
                    params={"tags": query.tags, "page": str(page), "limit": str(limit)},
                    headers={"User-Agent": self._user_agent},
                ) as response:
                    if response.status == 429:
                        raise E621Error(
                            "e621 просит снизить частоту запросов. Попробуй чуть позже."
                        )
                    if response.status != 200:
                        raise E621Error(f"e621 вернул HTTP {response.status}.")
                    body = await _read_limited(response.content, _MAX_JSON_BYTES)
        except E621Error:
            raise
        except Exception as error:
            raise E621Error("Не удалось связаться с e621.") from error
        try:
            payload = json.loads(body)
            posts = payload.get("posts", [])
            parsed = [_parse_post(post) for post in posts if isinstance(post, dict)]
            return [post for post in parsed if not post.tags & BLOCKED_AGE_TAGS]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise E621Error("e621 вернул неожиданный ответ.") from error

    async def download(self, url: str, *, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
        """Загрузить медиа с тем же User-Agent и строгим лимитом размера."""
        if max_bytes <= 0:
            raise ValueError("max_bytes должен быть больше нуля")
        await self._throttle()
        try:
            async with self._session(30) as session:
                async with session.get(
                    url, headers={"User-Agent": self._user_agent}
                ) as response:
                    if response.status != 200:
                        raise E621Error(f"Файл e621 вернул HTTP {response.status}.")
                    return await _read_limited(response.content, max_bytes)
        except E621Error:
            raise
        except Exception as error:
            raise E621Error("Не удалось скачать файл e621.") from error


async def _read_limited(content: Any, limit: int) -> bytes:
    """Прочитать поток целиком, остановившись сразу после лимита."""
    data = bytearray()
    async for chunk in content.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > limit:
            raise E621Error("Ответ e621 оказался слишком большим.")
    return bytes(data)


def _parse_post(post: dict[str, Any]) -> E621Post:
    """Преобразовать JSON API в устойчивую внутреннюю модель."""
    file_data = post.get("file") or {}
    sample = post.get("sample") or {}
    preview = post.get("preview") or {}
    tags = post.get("tags") or {}
    score = post.get("score") or {}
    return E621Post(
        post_id=int(post["id"]),
        rating=str(post.get("rating", "?")),
        file_url=file_data.get("url"),
        file_ext=str(file_data.get("ext", "jpg")).lower(),
        file_size=int(file_data.get("size", 0)),
        sample_url=sample.get("url"),
        preview_url=preview.get("url"),
        fav_count=int(post.get("fav_count", 0)),
        score=int(score.get("total", 0)),
        artists=tuple(str(item) for item in tags.get("artist", [])),
        tags=frozenset(
            str(item).casefold()
            for group in tags.values()
            if isinstance(group, list)
            for item in group
        ),
        sources=tuple(str(item) for item in post.get("sources", [])),
    )
