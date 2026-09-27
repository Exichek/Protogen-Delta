"""Выбор возрастного режима содержимого пользователем."""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from protogen_delta.core.user_state import ContentMode, UserStateStore

_CALLBACK_PREFIX = "adult"

AGE_PROMPT_TEXT = (
    "🍓 Выбери возрастной режим общения. Он определяет, насколько откровенно "
    "Дельта может обсуждать взрослые темы, реагировать на эротические изображения "
    "и участвовать в RP. На обычные вопросы, память и характер выбор не влияет.\n\n"
    "🔞 Мне есть 18 — доступны прямые реакции на 18+ контент, взрослая лексика "
    "и откровенный RP между совершеннолетними персонажами. Это не делает каждый "
    "разговор пошлым: взрослый тон включается только по контексту.\n\n"
    "🍓 Мне нет 18 — остаются дружеское общение, романтика, флирт и лёгкие "
    "намёки, но без подробных сексуальных сцен и описаний гениталий.\n\n"
    "Документы подтверждать не нужно. Выбор сохраняется только как настройка бота "
    "и в любой момент меняется командой /adult."
)

ADULT_ENABLED_TEXT = (
    "🔞 Взрослый режим включён. Дельта может поддерживать откровенные "
    "взрослые темы, реагировать на 18+ арты и участвовать в 18+ RP.\n\n"
    "Изменить выбор можно командой /adult."
)

SOFT_ENABLED_TEXT = (
    "🍓 Включён мягкий режим: флирт, романтика и лёгкие намёки остаются, "
    "но без откровенных сексуальных описаний и 18+ RP.\n\n"
    "Изменить выбор можно командой /adult."
)

FOREIGN_CALLBACK_TEXT = "Эта настройка предназначена не тебе."


def _callback_data(mode: ContentMode, user_id: int) -> str:
    """Собрать callback-data выбора режима."""
    return f"{_CALLBACK_PREFIX}:{mode}:{user_id}"


def _build_age_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Создать клавиатуру выбора возрастного режима."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔞 Мне есть 18",
                    callback_data=_callback_data("adult", user_id),
                    style="success",
                ),
                InlineKeyboardButton(
                    text="🍓 Мне нет 18",
                    callback_data=_callback_data("soft", user_id),
                    style="danger",
                ),
            ]
        ]
    )


async def send_age_prompt(message: Message, user_id: int) -> None:
    """Отправить пользователю описание режимов и кнопки выбора."""
    await message.answer(
        AGE_PROMPT_TEXT,
        reply_markup=_build_age_keyboard(user_id),
    )


def _parse_callback(data: str) -> tuple[ContentMode, int] | None:
    """Разобрать callback выбора режима."""
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != _CALLBACK_PREFIX:
        return None
    raw_mode, raw_user_id = parts[1:]
    if raw_mode not in {"adult", "soft"}:
        return None
    try:
        user_id = int(raw_user_id)
    except ValueError:
        return None
    mode: ContentMode = "adult" if raw_mode == "adult" else "soft"
    return mode, user_id


def create_adult_router(user_states: UserStateStore) -> Router:
    """Создать роутер команды и кнопок возрастного режима."""
    router = Router(name=__name__)

    @router.message(Command("adult"))
    async def adult_command(message: Message) -> None:
        """Показать текущий режим и дать изменить его кнопками."""
        if message.from_user is None:
            return
        async with user_states.use(message.from_user.id) as state:
            current = {
                "unselected": "ещё не выбран",
                "soft": "мягкий",
                "adult": "18+",
            }[state.content_mode]
        await message.answer(
            f"Текущий режим: {current}.\n\n{AGE_PROMPT_TEXT}",
            reply_markup=_build_age_keyboard(message.from_user.id),
        )

    @router.callback_query(F.data.startswith(f"{_CALLBACK_PREFIX}:"))
    async def adult_callback(callback: CallbackQuery) -> None:
        """Сохранить подтверждённый пользователем режим."""
        parsed = _parse_callback(callback.data or "")
        if parsed is None:
            await callback.answer()
            return
        mode, expected_user_id = parsed
        if callback.from_user.id != expected_user_id:
            await callback.answer(FOREIGN_CALLBACK_TEXT, show_alert=True)
            return

        async with user_states.use(expected_user_id) as state:
            state.content_mode = mode

        if isinstance(callback.message, Message):
            await callback.message.edit_text(
                ADULT_ENABLED_TEXT if mode == "adult" else SOFT_ENABLED_TEXT,
                reply_markup=None,
            )
        await callback.answer()

    return router
