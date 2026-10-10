"""Единая локальная справка /funcs и /help."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.config.settings import Settings
from protogen_delta.core.message_utils import split_message
from protogen_delta.core.user_state import ContentMode, UserStateStore
from protogen_delta.services.functions_catalog import functions_text


def create_help_router(
    user_states: UserStateStore | None = None, *, settings: Settings | None = None
) -> Router:
    """Создать справку для пользовательских команд без LLM."""
    router = Router(name=__name__)

    @router.message(Command("help", "funcs"))
    async def help_command(message: Message) -> None:
        mode: ContentMode = "unselected"
        if user_states is not None and message.from_user is not None:
            async with user_states.use(message.from_user.id) as state:
                mode = "soft" if state.age_restricted else state.content_mode
        for chunk in split_message(functions_text(mode, settings=settings)):
            await message.answer(chunk, parse_mode=None)

    return router
