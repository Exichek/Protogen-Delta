"""Доставка сформированных ответов в Telegram."""

import logging
from asyncio import CancelledError, Event, create_task, sleep, wait_for
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from time import perf_counter

from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from protogen_delta.core.message_utils import reply_delay_seconds, split_reply
from protogen_delta.services.response_engine import ReplyDelivery
from protogen_delta.services.stickers import ContextualStickerService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def show_typing(
    message: Message,
    bot: Bot | None,
    *,
    initial_delay_seconds: float = 0.7,
) -> AsyncIterator[None]:
    """Показывать статус набора, пока бот формирует содержательный ответ."""
    if initial_delay_seconds < 0:
        raise ValueError("Задержка статуса набора не может быть отрицательной")
    if bot is None:
        yield
        return

    stopped = Event()

    async def worker() -> None:
        """Обновлять Telegram chat action до завершения основного ответа."""
        delay = initial_delay_seconds
        while not stopped.is_set():
            try:
                await wait_for(stopped.wait(), timeout=delay)
                return
            except TimeoutError:
                pass
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
                return
            delay = 4.5

    task = create_task(worker())
    try:
        yield
    finally:
        stopped.set()
        await task


def create_reply_delivery(
    message: Message,
    bot: Bot | None,
    sticker_service: ContextualStickerService | None = None,
    *,
    user_id: int | None = None,
    context_tags: Collection[str] = (),
) -> ReplyDelivery:
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
                    await sleep(reply_delay_seconds(chunk))
                await message.answer(chunk)
                sent += 1
            if sticker_service is not None and user_id is not None:
                try:
                    await sticker_service.maybe_send(
                        chat_id=message.chat.id,
                        user_id=user_id,
                        context_tags=context_tags,
                        context_text=message.text or message.caption or "",
                    )
                except Exception:
                    logger.exception("Не удалось обработать контекстный стикер")
            outcome = "sent"
        except CancelledError:
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
