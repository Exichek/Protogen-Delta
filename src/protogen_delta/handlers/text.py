"""Обработчик обычных текстовых сообщений."""

import asyncio
import logging
from time import perf_counter

from aiogram import F, Router
from aiogram.types import Message

from protogen_delta.core.message_utils import split_message
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.rp import (
    RP_ALREADY_DISABLED_REPLY,
    RP_DISABLED_REPLY,
    is_roleplay_stop_message,
)
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine

RATE_LIMIT_REPLY = "Слишком быстро :D Подожди пару секунд."
BUSY_REPLY = "Я ещё отвечаю на предыдущее сообщение. Подожди немного."
logger = logging.getLogger(__name__)


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

        if message.from_user is None:
            return

        user_id = message.from_user.id

        if is_roleplay_stop_message(message.text):
            was_active = await response_engine.disable_roleplay(
                user_id,
            )

            if was_active:
                await message.answer(RP_DISABLED_REPLY)
            else:
                await message.answer(RP_ALREADY_DISABLED_REPLY)

            return

        if not limiter.allow(user_id):
            await message.answer(RATE_LIMIT_REPLY)
            return

        async def deliver(reply: str) -> None:
            chunks = split_message(reply)
            started = perf_counter()
            sent = 0
            outcome = "failed"
            try:
                for chunk in chunks:
                    await message.answer(chunk)
                    sent += 1
                outcome = "sent"
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                logger.info(
                    "Telegram reply outcome=%s chunks=%d/%d duration_ms=%.1f",
                    outcome,
                    sent,
                    len(chunks),
                    (perf_counter() - started) * 1000,
                )

        try:
            await response_engine.respond_and_deliver(user_id, message.text, deliver)
        except ResponseBusyError:
            await message.answer(BUSY_REPLY)

    return router
