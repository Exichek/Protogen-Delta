"""Обработчик команды /start."""

import logging
import random

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.message_utils import split_message
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.deepseek import (
    DeepSeekError,
    DeepSeekService,
)

logger = logging.getLogger(__name__)

FIRST_START_PREFIXES = (
    "Привет! Я Протоген Дельта.",
    "Йоу! Я Протоген Дельта.",
    "О, привет! Я Протоген Дельта.",
    "О, приветик! Я Протоген Дельта.",
    "Хей! Я Протоген Дельта.",
    "Здорова! Я Протоген Дельта.",
    "Ну привет! Я Протоген Дельта.",
    "Рад встрече! Я Протоген Дельта.",
    "Опа, новенький, привет! Я Протоген Дельта.",
    "Хай! Я Протоген Дельта.",
)

FIRST_START_FALLBACK_BODY = (
    "Со мной можно просто поболтать, спросить что-нибудь — "
    "в том числе техническое — или устроить RP, начав действие "
    "в *звёздочках*.\n\n"
    "За случайным артом есть /randomart, а если захочешь посмотреть "
    "доступные команды — /help.\n\n"
    "Ну а дальше разберёмся по ходу дела. С чего начнём?"
)

_FIRST_START_REQUEST = (
    "Продолжи уже начатое первое знакомство с новым пользователем. "
    "Не здоровайся и не представляйся заново."
)


async def _generate_first_start_message(
    deepseek: DeepSeekService,
    prompt: str,
) -> str:
    """Сгенерировать живое первое приветствие с безопасным fallback."""
    prefix = random.choice(FIRST_START_PREFIXES)

    try:
        generated = await deepseek.chat(
            system_prompt=prompt,
            user_message=_FIRST_START_REQUEST,
        )
    except DeepSeekError:
        logger.warning(
            "Не удалось сгенерировать первое приветствие через DeepSeek",
            exc_info=True,
        )
        return f"{prefix}\n\n{FIRST_START_FALLBACK_BODY}"

    generated = generated.strip()

    if not generated:
        logger.warning("DeepSeek вернул пустое первое приветствие")
        return f"{prefix}\n\n{FIRST_START_FALLBACK_BODY}"

    return f"{prefix}\n\n{generated}"


def create_start_router(
    users_repository: UsersRepository,
    start_messages: list[str],
    deepseek: DeepSeekService,
    first_start_prompt: str,
) -> Router:
    """Создать роутер команды /start."""
    router = Router(name=__name__)

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        """Зарегистрировать нового пользователя или ответить повторно."""
        user = message.from_user

        if user is None:
            logger.warning("Команда /start получена без данных пользователя")
            return

        if users_repository.add(user.id):
            logger.info("Зарегистрирован новый пользователь: %s", user.id)

            reply = await _generate_first_start_message(
                deepseek,
                first_start_prompt,
            )

            for chunk in split_message(reply):
                await message.answer(chunk)

            return

        if start_messages:
            reply = random.choice(start_messages)
        else:
            reply = "Я уже запущен :D"

        for chunk in split_message(reply):
            await message.answer(chunk)

    return router
