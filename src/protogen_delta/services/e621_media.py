"""Подготовка медиа и повторное использование файлов Telegram без хранения видео на диске."""

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from time import time
from urllib.parse import urlsplit
from weakref import WeakValueDictionary

from aiogram.types import BufferedInputFile, Message

from protogen_delta.repositories.e621_history import CachedMedia, E621HistoryRepository
from protogen_delta.services.e621 import (
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
    E621Client,
    E621Error,
    E621Post,
)
from protogen_delta.services.telegram_video import TelegramVideoConverter

_RAM_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    """Готовое вложение: байты до первой отправки либо Telegram file_id."""

    post: E621Post
    key: str
    kind: str
    content: bytes | str
    notice: str = ""

    @property
    def size(self) -> int:
        return len(self.content) if isinstance(self.content, bytes) else 0

    def input_file(self) -> BufferedInputFile | str:
        return (
            BufferedInputFile(
                self.content, filename=f"e621-{self.post.post_id}.{self.kind}"
            )
            if isinstance(self.content, bytes)
            else self.content
        )


class E621MediaService:
    """Ограничить параллельную загрузку и переиспользовать результат подготовки."""

    def __init__(
        self, client: E621Client, history: E621HistoryRepository, bot_id: int = 0
    ) -> None:
        self.client, self.history, self.bot_id = client, history, bot_id
        self.converter = TelegramVideoConverter()
        self._slots = asyncio.Semaphore(2)
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._ready: OrderedDict[str, PreparedMedia] = OrderedDict()

    async def prepare(self, post: E621Post, *, album: bool) -> PreparedMedia:
        signature = repr(
            (
                post.post_id,
                post.file_url,
                post.file_size,
                post.mp4_urls,
                post.webm_urls,
                album and post.file_ext == "gif",
            )
        )
        key = hashlib.sha256(signature.encode()).hexdigest()
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = await self.history.cached_media(self.bot_id, key)
            if cached:
                return PreparedMedia(
                    post, key, cached.kind, cached.file_id, cached.notice
                )
            if key in self._ready:
                self._ready.move_to_end(key)
                return replace(self._ready[key], post=post)
            async with self._slots:
                prepared = await self._download(post, key, album)
            if prepared.notice:
                return prepared
            self._ready[key] = prepared
            while sum(item.size for item in self._ready.values()) > _RAM_BYTES:
                self._ready.popitem(last=False)
            return prepared

    async def invalidate(self, prepared: PreparedMedia) -> None:
        self._ready.pop(prepared.key, None)
        await self.history.forget_media(self.bot_id, prepared.key)

    async def remember(self, prepared: PreparedMedia, delivered: Message) -> None:
        if prepared.notice:
            return
        item = delivered.video or delivered.animation or delivered.document
        file_id = (
            item.file_id
            if item
            else delivered.photo[-1].file_id if delivered.photo else None
        )
        if isinstance(file_id, str):
            await self.history.cache_media(
                self.bot_id,
                prepared.key,
                CachedMedia(file_id, prepared.kind, prepared.notice),
                time(),
            )
            self._ready.pop(prepared.key, None)

    async def _preview(self, post: E621Post, key: str, notice: str) -> PreparedMedia:
        url = post.preview_url or post.sample_url
        if not url:
            raise E621Error("У поста нет доступного превью.")
        data = await self.client.download(url, max_bytes=MAX_IMAGE_BYTES)
        return PreparedMedia(post, key, "jpg", data, notice)

    async def _download(self, post: E621Post, key: str, album: bool) -> PreparedMedia:
        data: bytes | None = None
        ext = post.file_ext.casefold()
        maximum = MAX_VIDEO_BYTES if ext in {"mp4", "webm", "gif"} else MAX_IMAGE_BYTES
        candidates = list(dict.fromkeys(post.mp4_urls + post.webm_urls))[:3]
        if post.file_url and post.file_size <= maximum:
            candidates.append(post.file_url)
        for url in dict.fromkeys(candidates):
            kind = urlsplit(url).path.rsplit(".", 1)[-1].casefold()
            try:
                data = await self.client.download(
                    url,
                    max_bytes=(
                        MAX_VIDEO_BYTES
                        if kind in {"mp4", "webm", "gif"}
                        else MAX_IMAGE_BYTES
                    ),
                )
                ext = kind
                break
            except E621Error:
                continue
        if data is None:
            size = f"{post.file_size / 1024**2:.1f} МиБ"
            return await self._preview(
                post,
                key,
                f"⚠️ Превью: {size}; лимит {maximum // 1024**2} МиБ или ошибка загрузки. Оригинал — по ссылке.",
            )
        if ext == "webm" or (ext == "gif" and album):
            try:
                data = await self.converter.convert(data)
                ext = "mp4"
            except E621Error:
                if album:
                    return await self._preview(
                        post,
                        key,
                        "⚠️ Превью: не удалось подготовить видео для альбома. Оригинал — по ссылке.",
                    )
                return PreparedMedia(
                    post,
                    key,
                    ext,
                    data,
                    "⚠️ Не удалось подготовить MP4; отправляю исходный WebM файлом.",
                )
        return PreparedMedia(post, key, ext, data)
