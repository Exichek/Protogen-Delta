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
from aiogram.types import ErrorEvent

logger = logging.getLogger(__name__)


async def handle_error(event: ErrorEvent) -> None:
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


def register_error_handler(dispatcher: Dispatcher) -> None:
    """Зарегистрировать глобальный обработчик ошибок."""
    dispatcher.errors.register(handle_error)
