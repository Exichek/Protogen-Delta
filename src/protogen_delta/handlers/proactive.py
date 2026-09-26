"""Пользовательское управление проактивными сообщениями."""

from time import time

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.repositories.memories import MemoriesRepository


def create_proactive_router(repository: MemoriesRepository) -> Router:
    """Создать роутер команды /proactive."""
    router = Router(name=__name__)

    @router.message(Command("proactive"))
    async def proactive(message: Message) -> None:
        if message.from_user is None:
            return
        parts = (message.text or "").split(maxsplit=1)
        action = parts[1].strip().lower() if len(parts) > 1 else "status"
        if action not in {"on", "off", "status"}:
            await message.answer("Использование: /proactive on, off или status")
            return
        if action == "status":
            enabled = await repository.proactive_enabled(message.from_user.id)
        else:
            enabled = action == "on"
            await repository.set_proactive(message.from_user.id, enabled, time())
        state = "включены" if enabled else "выключены"
        await message.answer(f"Проактивные сообщения {state}.")

    return router
