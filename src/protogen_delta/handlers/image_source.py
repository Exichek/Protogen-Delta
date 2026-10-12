"""Поиск источника только по явной reply-команде /source."""

import io

from aiogram import Bot, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.text import RATE_LIMIT_REPLY
from protogen_delta.services.image_source import ImageSourceError, ImageSourceService


def create_image_source_router(bot: Bot, service: ImageSourceService) -> Router:
    router = Router(name=__name__)
    limiter = UserRateLimiter(cooldown_seconds=20)

    @router.message(Command("source"))
    async def source_command(message: Message) -> None:
        replied = message.reply_to_message
        if replied is None or message.from_user is None or not replied.photo:
            await message.answer(
                "Ответь /source на фото арта — найду похожую публикацию и ссылку "
                "через SauceNAO. Команда отправляет сервису уменьшенную копию "
                "изображения. Это поиск источника, совпадение не гарантирует точность."
            )
            return
        if not limiter.allow(message.from_user.id):
            await message.answer(RATE_LIMIT_REPLY)
            return
        photo = replied.photo[-1]
        if photo.file_size is not None and photo.file_size > 20 * 1024 * 1024:
            await message.answer("Картинка должна быть меньше 20 МБ.")
            return
        try:
            buffer = io.BytesIO()
            await bot.download(photo.file_id, destination=buffer)
            matches = await service.search(buffer.getvalue())
        except ImageSourceError as error:
            await message.answer(str(error))
            return
        except TelegramAPIError, OSError:
            await message.answer("Не смог скачать изображение из Telegram.")
            return
        if not matches:
            await message.answer(
                "Уверенных совпадений не нашёл. По картинке источник угадывать не буду."
            )
            return
        await message.answer(
            "🔎 Похожие публикации (это ещё не подтверждение источника):\n\n"
            + "\n\n".join(
                f"{item.index}: сходство {item.similarity:.1f}%\n{item.url}"
                for item in matches
            )
        )

    return router
