"""Обработчик команды сброса контекста диалога."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.services.response_engine import ResponseEngine

RESET_REPLY = "Контекст диалога очищен."


def create_reset_router(
    response_engine: ResponseEngine,
) -> Router:
    """Создать роутер команды /reset."""
    router = Router(name=__name__)

    @router.message(Command("reset"))
    async def reset_command(message: Message) -> None:
        """Очистить пользовательский контекст диалога."""
        if message.from_user is None:
            return

        await response_engine.reset_user_context(
            message.from_user.id,
        )

        await message.answer(RESET_REPLY)

    return router
