"""Ограниченная фоновая синхронизация личных меню после запуска polling."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)


async def synchronize_menus(
    users: Iterable[int], update: Callable[[int], Awaitable[None]], *, limit: int = 3
) -> None:
    if limit < 1:
        raise ValueError("limit должен быть положительным")
    remaining = iter(users)

    async def worker() -> None:
        for user_id in remaining:
            try:
                async with asyncio.timeout(15):
                    await update(user_id)
            except Exception:
                logger.warning("Menu synchronization failed for one user")

    # Создаются только limit задач, независимо от размера базы пользователей.
    async with asyncio.TaskGroup() as group:
        for _ in range(limit):
            group.create_task(worker())
