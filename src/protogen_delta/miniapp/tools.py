"""Инструменты Mini App без вызовов языковой модели."""

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import FSInputFile

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.utilities import _id_keyboard
from protogen_delta.services.media_download import MediaDownloader


class ToolsBusyError(ValueError):
    """Инструмент уже работает или запросы поступают слишком часто."""


class MiniAppTools:
    """Использовать проверенные ID и загрузчик видео."""

    def __init__(
        self,
        bot: Bot,
        downloader: MediaDownloader,
    ) -> None:
        self._bot = bot
        self._downloader = downloader
        self._pending: set[int] = set()
        self._slots = asyncio.Semaphore(2)
        self._download_limiter = UserRateLimiter(cooldown_seconds=30)

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
