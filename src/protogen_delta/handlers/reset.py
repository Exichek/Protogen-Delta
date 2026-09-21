"""Обработчик безопасного полного сброса памяти пользователя."""

from time import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.response_engine import ResponseEngine

_RESET_CONFIRMATION_TTL_SECONDS = 300.0

_RESET_CALLBACK_PREFIX = "reset"
_RESET_CONFIRM_ACTION = "confirm"
_RESET_CANCEL_ACTION = "cancel"

RESET_CONFIRMATION_TEXT = (
    "⚠️ Ты уверен, что хочешь полностью сбросить мою память о тебе?\n\n"
    "Будут удалены:\n"
    "• история диалога;\n"
    "• RP-состояние;\n"
    "• накопленные эмоции;\n"
    "• отношения и степень знакомства;\n"
    "• всё сохранённое пользовательское состояние.\n\n"
    "Это действие нельзя будет отменить."
)

RESET_SUCCESS_TEXT = "Память о тебе полностью очищена.\n" "Начинаем с нуля."

RESET_CANCELLED_TEXT = "Сброс отменён. Ничего не изменилось."

RESET_EXPIRED_TEXT = (
    "Подтверждение сброса устарело.\n"
    "Если всё ещё хочешь очистить память — введи /reset ещё раз."
)

RESET_FOREIGN_CALLBACK_TEXT = "Эта кнопка предназначена не тебе."


def _build_callback_data(
    action: str,
    user_id: int,
    created_at: int,
) -> str:
    """Собрать callback-data для подтверждения сброса."""
    return f"{_RESET_CALLBACK_PREFIX}:" f"{action}:" f"{user_id}:" f"{created_at}"


def _build_reset_keyboard(
    user_id: int,
    created_at: int,
) -> InlineKeyboardMarkup:
    """Создать клавиатуру подтверждения полного сброса."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Да, забыть всё",
                    callback_data=_build_callback_data(
                        _RESET_CONFIRM_ACTION,
                        user_id,
                        created_at,
                    ),
                    style="success",
                ),
                InlineKeyboardButton(
                    text="❌ Нет",
                    callback_data=_build_callback_data(
                        _RESET_CANCEL_ACTION,
                        user_id,
                        created_at,
                    ),
                    style="danger",
                ),
            ]
        ]
    )


def _parse_callback_data(
    data: str,
) -> tuple[str, int, int] | None:
    """Разобрать callback-data подтверждения сброса."""
    parts = data.split(":")

    if len(parts) != 4:
        return None

    prefix, action, raw_user_id, raw_created_at = parts

    if prefix != _RESET_CALLBACK_PREFIX:
        return None

    if action not in {
        _RESET_CONFIRM_ACTION,
        _RESET_CANCEL_ACTION,
    }:
        return None

    try:
        user_id = int(raw_user_id)
        created_at = int(raw_created_at)
    except ValueError:
        return None

    return (
        action,
        user_id,
        created_at,
    )


async def _edit_callback_message(
    callback: CallbackQuery,
    text: str,
) -> None:
    """Заменить сообщение подтверждения и убрать кнопки."""
    if not isinstance(callback.message, Message):
        return

    await callback.message.edit_text(
        text,
        reply_markup=None,
    )


def create_reset_router(
    response_engine: ResponseEngine,
    users_repository: UsersRepository,
) -> Router:
    """Создать роутер безопасного полного сброса."""
    router = Router(name=__name__)

    @router.message(Command("reset"))
    async def reset_command(message: Message) -> None:
        """Запросить подтверждение полного сброса памяти."""
        if message.from_user is None:
            return

        created_at = int(time())

        await message.answer(
            RESET_CONFIRMATION_TEXT,
            reply_markup=_build_reset_keyboard(
                message.from_user.id,
                created_at,
            ),
        )

    @router.callback_query(
        F.data.startswith(f"{_RESET_CALLBACK_PREFIX}:"),
    )
    async def reset_callback(callback: CallbackQuery) -> None:
        """Обработать подтверждение или отмену полного сброса."""
        if callback.data is None:
            await callback.answer()
            return

        parsed = _parse_callback_data(callback.data)

        if parsed is None:
            await callback.answer()
            return

        action, expected_user_id, created_at = parsed

        if callback.from_user.id != expected_user_id:
            await callback.answer(
                RESET_FOREIGN_CALLBACK_TEXT,
                show_alert=True,
            )
            return

        if time() - created_at > _RESET_CONFIRMATION_TTL_SECONDS:
            await _edit_callback_message(
                callback,
                RESET_EXPIRED_TEXT,
            )
            await callback.answer()
            return

        if action == _RESET_CANCEL_ACTION:
            await _edit_callback_message(
                callback,
                RESET_CANCELLED_TEXT,
            )
            await callback.answer()
            return

        await response_engine.reset_user(
            expected_user_id,
        )

        users_repository.remove(
            expected_user_id,
        )

        await _edit_callback_message(
            callback,
            RESET_SUCCESS_TEXT,
        )

        await callback.answer()

    return router
