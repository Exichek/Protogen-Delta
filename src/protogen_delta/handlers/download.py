"""Явная загрузка публичных видео, без вызова LLM."""

from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.text import RATE_LIMIT_REPLY
from protogen_delta.services.media_download import MediaDownloader, MediaDownloadError


def create_download_router(downloader: MediaDownloader) -> Router:
    router = Router(name=__name__)
    limiter = UserRateLimiter(cooldown_seconds=30)
    active: set[int] = set()

    @router.message(Command("download"))
    async def download_command(message: Message) -> None:
        if message.from_user is None:
            return
        _, _, url = (message.text or "").partition(" ")
        url = url.strip()
        if not url:
            await message.answer(
                "📥 /download <ссылка на видео>\n\n"
                "YouTube, Instagram, TikTok, Vimeo, VK и VK Видео (включая клипы), X/Twitter. "
                "GIF из X тоже поддерживаются, включая ссылки fixupx/fxtwitter. "
                "Одно публичное видео или GIF "
                "до 10 минут и 100 МБ. Крупные файлы сжимаются для Telegram; "
                "обработка — до четырёх минут. Доступ зависит от ограничений сайта."
            )
            return
        user_id = message.from_user.id
        if user_id in active:
            await message.answer("Предыдущее видео ещё загружается.")
            return
        if not limiter.allow(user_id):
            await message.answer(RATE_LIMIT_REPLY)
            return
        active.add(user_id)
        try:
            async with downloader.download(url) as media:
                file = FSInputFile(media.path)
                caption = media.title + (
                    "\nСжато для отправки в Telegram." if media.compressed else ""
                )
                if media.animation:
                    await message.answer_animation(file, caption=caption)
                elif media.path.suffix.lower() == ".mp4":
                    await message.answer_video(
                        file, caption=caption, supports_streaming=True
                    )
                else:
                    await message.answer_document(file, caption=caption)
        except MediaDownloadError as error:
            await message.answer(str(error))
        except TelegramAPIError:
            await message.answer(
                "Видео скачано, но Telegram не принял файл. Попробуй позже."
            )
        finally:
            active.discard(user_id)

    return router
