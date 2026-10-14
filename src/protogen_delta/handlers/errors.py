"""Глобальная обработка ошибок Telegram-бота."""

import logging

from aiogram import Dispatcher
from aiogram.exceptions import (
    TelegramConflictError,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramUnauthorizedError,
)
from aiogram.types import ErrorEvent, Update

from protogen_delta.handlers.failure_reply import send_failure_reply
from protogen_delta.services.stickers import ContextualStickerService

logger = logging.getLogger(__name__)


async def handle_error(
    event: ErrorEvent, sticker_service: ContextualStickerService | None = None
) -> None:
    """Обработать ошибку, возникшую при обработке Telegram update."""
    exception = event.exception

    if isinstance(exception, TelegramForbiddenError):
        logger.warning(
            "Telegram запретил отправку сообщения: "
            "пользователь мог заблокировать бота или удалить чат."
        )
        return

    if isinstance(exception, TelegramRetryAfter):
        logger.warning(
            "Telegram ограничил частоту запросов. " "Повтор возможен через %s сек.",
            exception.retry_after,
        )
        return

    if isinstance(exception, TelegramNetworkError):
        logger.warning(
            "Ошибка соединения с Telegram API.",
            exc_info=(
                type(exception),
                exception,
                exception.__traceback__,
            ),
        )
        return

    if isinstance(exception, TelegramUnauthorizedError):
        logger.critical(
            "Telegram отклонил токен бота.",
            exc_info=(
                type(exception),
                exception,
                exception.__traceback__,
            ),
        )
        return

    if isinstance(exception, TelegramConflictError):
        logger.critical(
            "Обнаружен конфликт Telegram polling: "
            "возможно, запущен второй экземпляр бота.",
            exc_info=(
                type(exception),
                exception,
                exception.__traceback__,
            ),
        )
        return

    logger.error(
        "Необработанная ошибка при обработке Telegram update.",
        exc_info=(
            type(exception),
            exception,
            exception.__traceback__,
        ),
    )
    if sticker_service is None:
        return
    message = event.update.message if isinstance(event.update, Update) else None
    if (
        message is not None
        and message.from_user is not None
        and not message.from_user.is_bot
    ):
        try:
            await send_failure_reply(
                message,
                "Не получилось обработать сообщение. Попробуй повторить вопрос.",
                sticker_service,
            )
        except Exception:
            logger.warning("Не удалось доставить уведомление о сбое")


def register_error_handler(dispatcher: Dispatcher) -> None:
    """Зарегистрировать глобальный обработчик ошибок."""
    dispatcher.errors.register(handle_error)
