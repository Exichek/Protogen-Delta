"""Не выполнять один Telegram запрос повторно, включая повторную доставку update."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.types import Message, TelegramObject

from protogen_delta.handlers.dialogue_archive import telegram_source_key
from protogen_delta.repositories.delivery_ledger import DeliveryLedgerRepository
from protogen_delta.services.delivery_ledger import DeliveryOperation, current_delivery

logger = logging.getLogger(__name__)


class DeliveryLedgerMiddleware(BaseMiddleware):
    def __init__(self, repository: DeliveryLedgerRepository) -> None:
        self.repository = repository

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        bot = data.get("bot")
        if (
            not isinstance(event, Message)
            or not event.from_user
            or event.from_user.is_bot
            or not isinstance(bot, Bot)
        ):
            return await handler(event, data)
        key = telegram_source_key(event, bot.id)
        try:
            claimed = await self.repository.claim(event.from_user.id, key)
        except Exception:
            # При недоступной дедупликации не запускать действия/модель вслепую.
            logger.error(
                "Не удалось зарегистрировать входящий запрос; обработка остановлена"
            )
            return None
        if not claimed:
            return None
        operation = DeliveryOperation(event.from_user.id, key)
        token = current_delivery.set(operation)
        try:
            await self.repository.mark(key, "processing")
            return await handler(event, data)
        except BaseException:
            operation.failed = True
            raise
        finally:
            current_delivery.reset(token)
            state = operation.result_state(
                command=isinstance(event.text, str) and event.text.startswith("/")
            )
            try:
                await self.repository.mark(key, state, operation.confirmed)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("Не удалось сохранить итог обработки запроса")
