"""Обработчик команды /start."""

import logging
import random

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.message_utils import split_message
from protogen_delta.core.user_state import UserStateStore
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

REPEAT_START_FALLBACK = (
    "Снова привет. Я на месте: можем просто поболтать, продолжить RP, "
    "разобрать вопрос или проверить что-нибудь актуальное вроде погоды и курса. "
    "Что сегодня делаем?"
)

_FIRST_START_REQUEST = (
    "Продолжи уже начатое первое знакомство с новым пользователем. "
    "Не здоровайся и не представляйся заново."
)

_REPEAT_START_REQUEST = (
    "Пользователь снова вызвал /start. Поприветствуй его заново одним живым "
    "сообщением и кратко напомни о нескольких своих возможностях."
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


async def _generate_repeat_start_message(
    deepseek: DeepSeekService,
    prompt: str,
) -> str:
    """Сгенерировать новое приветствие для уже зарегистрированного пользователя."""
    try:
        generated = await deepseek.chat(
            system_prompt=prompt,
            user_message=_REPEAT_START_REQUEST,
        )
    except DeepSeekError:
        logger.warning(
            "Не удалось сгенерировать повторное приветствие через DeepSeek",
            exc_info=True,
        )
        return REPEAT_START_FALLBACK

    generated = generated.strip()
    if not generated:
        logger.warning("DeepSeek вернул пустое повторное приветствие")
        return REPEAT_START_FALLBACK
    return generated


def create_start_router(
    users_repository: UsersRepository,
    deepseek: DeepSeekService,
    first_start_prompt: str,
    repeat_start_prompt: str | None = None,
    user_states: UserStateStore | None = None,
) -> Router:
    """Создать роутер команды /start."""
    router = Router(name=__name__)
    states = user_states or UserStateStore()
    pending: set[int] = set()

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        """Зарегистрировать нового пользователя или ответить повторно."""
        user = message.from_user

        if user is None:
            logger.warning("Команда /start получена без данных пользователя")
            return

        if user.id in pending:
            await message.answer("Я уже готовлю приветствие. Подожди немного.")
            return

        pending.add(user.id)
        try:
            async with states.use(user.id):
                await greet(message, user.id)
        finally:
            pending.remove(user.id)

    async def greet(message: Message, user_id: int) -> None:
        """Отправить приветствие под общей блокировкой состояния."""
        if user_id not in users_repository.get_all():

            reply = await _generate_first_start_message(
                deepseek,
                first_start_prompt,
            )

            for chunk in split_message(reply):
                await message.answer(chunk)

            users_repository.add(user_id)
            logger.info("Зарегистрирован новый пользователь: %s", user_id)
            return

        reply = await _generate_repeat_start_message(
            deepseek,
            repeat_start_prompt or first_start_prompt,
        )

        for chunk in split_message(reply):
            await message.answer(chunk)

    return router
