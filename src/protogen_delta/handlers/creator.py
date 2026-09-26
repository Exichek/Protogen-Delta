"""Команды создателя для адресных сообщений и рассылки."""

import asyncio
import logging
import secrets

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from protogen_delta.repositories.users import UsersRepository

logger = logging.getLogger(__name__)
_BROADCAST_PREFIX = "broadcast"


def create_creator_router(
    *, bot: Bot, users_repository: UsersRepository, creator_id: int | None
) -> Router:
    """Создать доступные только создателю команды отправки."""
    router = Router(name=__name__)
    pending: dict[str, str] = {}

    def allowed(message: Message) -> bool:
        return message.from_user is not None and message.from_user.id == creator_id

    async def deny(message: Message) -> None:
        await message.answer("⛔ Эта команда доступна только создателю.")

    @router.message(Command("message"))
    async def message_user(message: Message) -> None:
        if not allowed(message):
            await deny(message)
            return
        parts = (message.text or "").split(maxsplit=2)
        if len(parts) < 3 or not parts[1].lstrip("-").isdigit():
            await message.answer("Использование: /message <user_id> <текст>")
            return
        target = int(parts[1])
        try:
            await bot.send_message(target, parts[2])
        except TelegramAPIError:
            logger.warning(
                "Не удалось отправить сообщение пользователю %s", target, exc_info=True
            )
            await message.answer("⚠️ Telegram не принял сообщение.")
            return
        await message.answer(f"✅ Сообщение отправлено пользователю {target}.")

    @router.message(Command("broadcast"))
    async def prepare_broadcast(message: Message) -> None:
        if not allowed(message):
            await deny(message)
            return
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            await message.answer("Использование: /broadcast <текст>")
            return
        token = secrets.token_urlsafe(8)
        pending.clear()
        pending[token] = parts[1].strip()
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Отправить всем",
                        callback_data=f"{_BROADCAST_PREFIX}:confirm:{token}",
                    ),
                    InlineKeyboardButton(
                        text="❌ Отмена",
                        callback_data=f"{_BROADCAST_PREFIX}:cancel:{token}",
                    ),
                ]
            ]
        )
        await message.answer(
            f"Рассылка для {users_repository.count()} пользователей:\n\n{pending[token]}",
            reply_markup=keyboard,
        )

    @router.callback_query(F.data.startswith(f"{_BROADCAST_PREFIX}:"))
    async def confirm_broadcast(callback: CallbackQuery) -> None:
        if callback.from_user.id != creator_id or callback.data is None:
            await callback.answer("Нет доступа.", show_alert=True)
            return
        _, action, token = callback.data.split(":", maxsplit=2)
        text = pending.pop(token, None)
        if text is None:
            await callback.answer("Подтверждение устарело.", show_alert=True)
            return
        if action == "cancel":
            await callback.answer("Рассылка отменена.")
            if isinstance(callback.message, Message):
                await callback.message.edit_text("Рассылка отменена.")
            return
        sent = failed = 0
        for user_id in users_repository.get_all():
            try:
                await bot.send_message(user_id, text)
                sent += 1
            except TelegramAPIError:
                failed += 1
                logger.warning(
                    "Ошибка рассылки пользователю %s", user_id, exc_info=True
                )
            await asyncio.sleep(0.05)
        result = f"✅ Рассылка завершена. Отправлено: {sent}, ошибок: {failed}."
        await callback.answer()
        if isinstance(callback.message, Message):
            await callback.message.edit_text(result)

    return router
