"""Обработчик управления RP-режимом."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.services.response_engine import ResponseEngine

RP_DISABLED_REPLY = "RP-режим завершён."
RP_ALREADY_DISABLED_REPLY = "RP-режим уже выключен."
RP_USAGE_REPLY = "Использование: /rp off"

ROLEPLAY_STOP_MESSAGES = frozenset(
    {
        "стоп rp",
        "стоп рп",
        "хватит rp",
        "хватит рп",
        "выйди из rp",
        "выйди из рп",
        "закончим rp",
        "закончим рп",
        "закончи rp",
        "закончи рп",
    }
)


def is_roleplay_stop_message(text: str) -> bool:
    """Проверить, просит ли пользователь явно завершить RP."""
    normalized = " ".join(text.lower().strip().split())

    return normalized in ROLEPLAY_STOP_MESSAGES


def create_rp_router(
    response_engine: ResponseEngine,
) -> Router:
    """Создать роутер управления RP-режимом."""
    router = Router(name=__name__)

    @router.message(Command("rp"))
    async def rp_command(message: Message) -> None:
        """Обработать команду управления RP-режимом."""
        if message.from_user is None:
            return

        text = (message.text or "").strip()
        parts = text.split(maxsplit=1)

        if len(parts) < 2 or parts[1].strip().lower() != "off":
            await message.answer(RP_USAGE_REPLY)
            return

        was_active = await response_engine.disable_roleplay(
            message.from_user.id,
        )

        if was_active:
            await message.answer(RP_DISABLED_REPLY)
        else:
            await message.answer(RP_ALREADY_DISABLED_REPLY)

    return router
