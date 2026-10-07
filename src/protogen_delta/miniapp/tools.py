"""Инструменты Mini App без вызовов языковой модели."""

import asyncio
import re
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from io import BytesIO
from time import monotonic
from urllib.parse import urlsplit

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import FSInputFile
from PIL import Image, UnidentifiedImageError

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.e621 import _query
from protogen_delta.handlers.utilities import _id_keyboard
from protogen_delta.repositories.e621_history import E621Preferences
from protogen_delta.services.e621 import BLOCKED_AGE_TAGS, E621_BASE_URL, E621Client
from protogen_delta.services.media_download import MediaDownloader


class ToolsBusyError(ValueError):
    """Инструмент уже работает или запросы поступают слишком часто."""


def _gallery_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.hostname
            in {
                "static1.e621.net",
                "static2.e621.net",
                "static1.e926.net",
                "static2.e926.net",
            }
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def _gallery_image_type(data: bytes) -> str:
    try:
        with Image.open(BytesIO(data)) as image:
            formats = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
            if (
                image.format not in formats
                or image.width * image.height > 40_000_000
                or getattr(image, "is_animated", False)
            ):
                raise ValueError("Источник не отдал обычную статичную картинку.")
            content_type = formats[image.format]
            image.verify()
            return content_type
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Не удалось прочитать изображение.") from error


class MiniAppTools:
    """Использовать проверенные ID, загрузчик видео и безопасный поиск артов."""

    def __init__(
        self,
        bot: Bot,
        e621: E621Client,
        downloader: MediaDownloader,
        *,
        user_states: UserStateStore | None = None,
    ) -> None:
        self._bot = bot
        self._e621 = e621
        self._downloader = downloader
        self._pending: set[int] = set()
        self._slots = asyncio.Semaphore(2)
        self._download_limiter = UserRateLimiter(cooldown_seconds=30)
        self._gallery_limiter = UserRateLimiter(cooldown_seconds=2)
        self._user_states = user_states
        self._gallery_images: OrderedDict[tuple[int, int], tuple[float, str, str]] = (
            OrderedDict()
        )
        self._gallery_cache: OrderedDict[tuple[int, int], tuple[bytes, str]] = (
            OrderedDict()
        )
        self._gallery_slots = asyncio.Semaphore(3)

    async def lookup_id(self, user_id: int, target: str) -> dict[str, object]:
        """Не раскрывать приватные чаты по произвольному числовому ID."""
        if target == "self":
            return {"id": user_id, "title": "Твой Telegram ID", "type": "private"}
        if target == "select":
            await self._bot.send_message(
                user_id,
                "Выбери пользователя, бота, группу или канал:",
                reply_markup=_id_keyboard(),
            )
            return {"message": "Выбор ID открыт в чате с Дельтой."}
        if target == "bot":
            user = await self._bot.get_me()
            return {"id": user.id, "title": user.full_name, "type": "bot"}
        if not re.fullmatch(r"@[A-Za-z][A-Za-z0-9_]{3,31}", target):
            raise ValueError("Укажи @username публичной группы или канала.")
        chat = await self._bot.get_chat(target)
        if chat.type not in {"channel", "supergroup", "group"}:
            raise ValueError("Пользователя выбери кнопкой «Выбрать в Telegram».")
        return {"id": chat.id, "title": chat.title or target, "type": chat.type}

    async def gallery(self, user_id: int, raw: str, page: int) -> dict[str, object]:
        """Показывать обычные статичные арты в самом окне Mini App."""
        if not self._gallery_limiter.allow(user_id):
            raise ToolsBusyError("Подожди пару секунд перед новым поиском.")
        query = _query(raw, E621Preferences(media_filter="images"), "soft")
        if self._user_states and self._user_states.get(user_id).content_mode == "adult":
            query = replace(query, base_url=E621_BASE_URL)
        posts = await self._e621.search(query, page=page, limit=20)
        items = []
        for post in posts:
            if (
                post.rating != "s"
                or post.file_ext not in {"jpg", "jpeg", "png", "webp"}
                or post.tags & BLOCKED_AGE_TAGS
            ):
                continue
            url = post.sample_url or post.file_url or post.preview_url
            if not url:
                continue
            if not _gallery_url(url):
                continue
            items.append(
                {
                    "id": post.post_id,
                    "url": url,
                    "page_url": f"{query.base_url}/posts/{post.post_id}",
                    "artists": list(post.artists),
                    "favs": post.fav_count,
                }
            )
            preview = post.preview_url or url
            original = post.file_url or url
            if not _gallery_url(preview) or not _gallery_url(original):
                items.pop()
                continue
            key = (user_id, post.post_id)
            self._gallery_images[key] = (monotonic() + 600, preview, original)
            self._gallery_images.move_to_end(key)
        for key in list(self._gallery_images):
            if self._gallery_images[key][0] < monotonic():
                self._gallery_images.pop(key)
                self._gallery_cache.pop(key, None)
        while len(self._gallery_images) > 1000:
            key, _ = self._gallery_images.popitem(last=False)
            self._gallery_cache.pop(key, None)
        return {
            "items": items,
            "page": page,
            "has_more": len(posts) == 20,
            "site": "e621" if query.base_url == E621_BASE_URL else "e926",
        }

    async def gallery_image(
        self, user_id: int, post_id: int, *, full: bool
    ) -> tuple[bytes, str]:
        """Отдать изображение только из результатов собственного safe-поиска."""
        key = (user_id, post_id)
        record = self._gallery_images.get(key)
        if record is None or record[0] < monotonic():
            raise ValueError("Обнови поиск: ссылка на изображение устарела.")
        if not full and key in self._gallery_cache:
            self._gallery_cache.move_to_end(key)
            return self._gallery_cache[key]
        async with self._gallery_slots:
            data = await self._e621.download(
                record[2] if full else record[1],
                max_bytes=(20 if full else 2) * 1024 * 1024,
                allow_redirects=False,
            )
            content_type = await asyncio.to_thread(_gallery_image_type, data)
        if not full:
            self._gallery_cache[key] = (data, content_type)
            self._gallery_cache.move_to_end(key)
            while (
                sum(len(item[0]) for item in self._gallery_cache.values())
                > 16 * 1024 * 1024
            ):
                self._gallery_cache.popitem(last=False)
        return data, content_type

    @asynccontextmanager
    async def _download_slot(self, user_id: int) -> AsyncIterator[None]:
        if user_id in self._pending:
            raise ToolsBusyError("Предыдущее видео ещё загружается.")
        self._pending.add(user_id)
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._slots.acquire(), timeout=0.1)
                acquired = True
            except TimeoutError as error:
                raise ToolsBusyError("Загрузчик занят. Попробуй чуть позже.") from error
            if not self._download_limiter.allow(user_id):
                raise ToolsBusyError("Подожди 30 секунд перед следующим видео.")
            yield
        finally:
            self._pending.discard(user_id)
            if acquired:
                self._slots.release()

    async def download(self, user_id: int, url: str) -> dict[str, object]:
        """Отправлять скачанный файл только владельцу подписанного запроса."""
        async with self._download_slot(user_id):
            async with self._downloader.download(url) as media:
                file = FSInputFile(media.path)
                try:
                    if media.path.suffix.lower() == ".mp4":
                        await self._bot.send_video(
                            user_id, file, caption=media.title, supports_streaming=True
                        )
                    else:
                        await self._bot.send_document(
                            user_id, file, caption=media.title
                        )
                except TelegramAPIError as error:
                    raise ValueError(
                        "Telegram не принял файл. Открой чат с ботом и попробуй позже."
                    ) from error
        return {"message": "Видео отправлено тебе в чат с Дельтой."}
