"""Инструменты Mini App и отдельный текстовый перевод."""

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import FSInputFile

from protogen_delta.core.operation_metrics import OperationMetrics, OperationSnapshot
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.utilities import _id_keyboard
from protogen_delta.services.media_download import MediaDownloader
from protogen_delta.services.translation import TranslationError, TranslationService


class ToolsBusyError(ValueError):
    """Инструмент уже работает или запросы поступают слишком часто."""


class MiniAppTools:
    """Использовать проверенные ID и загрузчик видео."""

    def __init__(
        self,
        bot: Bot,
        downloader: MediaDownloader,
        *,
        translator: TranslationService | None = None,
    ) -> None:
        self._bot = bot
        self._downloader = downloader
        self._pending: set[int] = set()
        self._slots = asyncio.Semaphore(2)
        self._download_limiter = UserRateLimiter(cooldown_seconds=30)
        self._translator = translator
        self._translation_pending: set[int] = set()
        self._translation_slots = asyncio.Semaphore(2)
        self._translation_limiter = UserRateLimiter(cooldown_seconds=2)
        self._translation_metrics = OperationMetrics(capacity=2)

    def translation_snapshot(self) -> OperationSnapshot:
        return self._translation_metrics.snapshot()

    async def translate(
        self, user_id: int, text: str, source: str, target: str
    ) -> dict[str, object]:
        """Ограничить параллельные переводы, не затрагивая чат и загрузчик."""
        async with self._translation_slot(user_id):
            assert self._translator is not None
            return await self._translator.translate(text, source, target)

    async def translate_file(
        self,
        user_id: int,
        read_file: Callable[[], Awaitable[bytes]],
        file_name: str,
        mime_type: str,
        mode: str,
        source: str,
        target: str,
    ) -> dict[str, object]:
        async with self._translation_slot(user_id):
            assert self._translator is not None
            async with asyncio.timeout(300):
                return await self._translator.translate_file(
                    await read_file(), file_name, mime_type, mode, source, target
                )

    @asynccontextmanager
    async def _translation_slot(self, user_id: int) -> AsyncIterator[None]:
        if self._translator is None:
            raise TranslationError("Переводчик сейчас недоступен.")
        if user_id in self._translation_pending:
            raise ToolsBusyError("Предыдущий перевод ещё выполняется.")
        self._translation_pending.add(user_id)
        acquired = False
        try:
            with self._translation_metrics.measure(queued=True) as measurement:
                try:
                    await asyncio.wait_for(
                        self._translation_slots.acquire(), timeout=0.1
                    )
                    acquired = True
                except TimeoutError as error:
                    raise ToolsBusyError(
                        "Переводчик занят. Попробуй чуть позже."
                    ) from error
                if not self._translation_limiter.allow(user_id):
                    raise ToolsBusyError(
                        "Подожди пару секунд перед следующим переводом."
                    )
                measurement.start()
                yield
        finally:
            self._translation_pending.discard(user_id)
            if acquired:
                self._translation_slots.release()

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
                caption = media.title + (
                    "\nСжато для отправки в Telegram." if media.compressed else ""
                )
                try:
                    if media.animation:
                        await self._bot.send_animation(user_id, file, caption=caption)
                    elif media.path.suffix.lower() == ".mp4":
                        await self._bot.send_video(
                            user_id, file, caption=caption, supports_streaming=True
                        )
                    else:
                        await self._bot.send_document(user_id, file, caption=caption)
                except TelegramAPIError as error:
                    raise ValueError(
                        "Telegram не принял файл. Открой чат с ботом и попробуй позже."
                    ) from error
        return {
            "message": (
                "GIF отправлен тебе в чат с Дельтой."
                if media.animation
                else "Видео отправлено тебе в чат с Дельтой."
            )
        }
