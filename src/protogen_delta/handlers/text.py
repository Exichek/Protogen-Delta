"""Обработчик обычных текстовых сообщений."""

from aiogram import F, Router
from aiogram.types import Message

from protogen_delta.core.message_utils import split_message
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.services.response_engine import ResponseEngine

RATE_LIMIT_REPLY = "Слишком быстро :D Подожди пару секунд."


def create_text_router(
    response_engine: ResponseEngine,
    rate_limiter: UserRateLimiter | None = None,
) -> Router:
    """Создать роутер обычных текстовых сообщений."""
    router = Router(name=__name__)
    limiter = rate_limiter or UserRateLimiter()

    @router.message(F.text)
    async def handle_text(message: Message) -> None:
        """Передать сообщение движку и отправить сформированный ответ."""
        if message.text is None:
            return

        if message.text.startswith("/"):
            return

        if message.from_user is not None:
            user_id = message.from_user.id

            if not limiter.allow(user_id):
                await message.answer(RATE_LIMIT_REPLY)
                return

        reply = await response_engine.respond(message.text)

        for chunk in split_message(reply):
            await message.answer(chunk)

    return router
