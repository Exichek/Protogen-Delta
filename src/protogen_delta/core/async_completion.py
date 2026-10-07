"""Дождаться освобождения внешнего ресурса при отмене ожидающей операции."""

import asyncio
from collections.abc import Awaitable


async def finish_operation[T](operation: Awaitable[T]) -> T:
    """Не отменять запись/cleanup вместе с запросом; затем восстановить отмену."""
    task = asyncio.ensure_future(operation)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result
