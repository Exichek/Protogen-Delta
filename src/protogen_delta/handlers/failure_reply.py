"""Сначала доставить сообщение о сбое, затем безопасную реакцию 404."""

import logging

from aiogram.types import Message

from protogen_delta.services.stickers import ContextualStickerService, declines_stickers

logger = logging.getLogger(__name__)


async def send_failure_reply(
    message: Message, text: str, stickers: ContextualStickerService | None = None
) -> None:
    await message.answer(text)
    original = getattr(message, "text", None) or getattr(message, "caption", None)
    visible = original if isinstance(original, str) else ""
    if (
        stickers is not None
        and message.from_user is not None
        and not declines_stickers(visible)
    ):
        try:
            await stickers.send_error(
                chat_id=message.chat.id, user_id=message.from_user.id
            )
        except Exception:
            logger.warning("Не удалось дополнить сообщение о сбое стикером")
