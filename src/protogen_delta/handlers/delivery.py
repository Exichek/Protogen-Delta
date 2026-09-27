"""Доставка сформированных ответов в Telegram."""

import asyncio
import logging
from time import perf_counter

from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from protogen_delta.core.message_utils import reply_delay_seconds, split_reply
from protogen_delta.services.response_engine import ReplyDelivery

logger = logging.getLogger(__name__)


def create_reply_delivery(message: Message, bot: Bot | None) -> ReplyDelivery:
    """Создать отправку ответа частями со статусом набора и паузами."""

    async def deliver(reply: str) -> None:
        chunks = split_reply(reply)
        started = perf_counter()
        sent = 0
        outcome = "failed"
        try:
            for index, chunk in enumerate(chunks):
                if index and bot is not None:
                    try:
                        await bot.send_chat_action(
                            chat_id=message.chat.id,
                            action=ChatAction.TYPING,
                        )
                    except TelegramAPIError:
                        logger.debug(
                            "Не удалось показать статус набора сообщения",
                            exc_info=True,
                        )
                    await asyncio.sleep(reply_delay_seconds(chunk))
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

    return deliver
