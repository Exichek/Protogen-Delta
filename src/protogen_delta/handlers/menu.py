"""Компактная панель управления ботом в одном Telegram-сообщении."""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)

from protogen_delta.core.user_state import UserStateStore

_PREFIX = "delta-menu"

_PAGES = {
    "main": (
        "⚙️ ИИ-ассистент Протоген Дельта\n\n"
        "Здесь собраны основные возможности. Выбери раздел — я обновлю это же "
        "сообщение, не засоряя чат."
    ),
    "chat": (
        "💬 Общение и RP\n\n"
        "Пиши обычным сообщением, присылай изображения, стикеры, документы, "
        "голосовые и видео. Действие в *звёздочках* включает RP по контексту.\n\n"
        "/rp off — закончить текущую сцену\n"
        "/adult — выбрать возрастной режим\n"
        "/reset — стереть диалог, память и профиль"
    ),
    "art": (
        "🎨 Арты e621/e926\n\n"
        "Используй /e6 и теги через пробел:\n"
        "/e6 dragon order:favcount\n\n"
        "Для серии до 10 постов добавь count:10.\n\n"
        "Можно написать тот же tag-query обычным сообщением. Кнопка «Ещё» "
        "продолжит поиск без повторов. Доступный рейтинг зависит от /adult."
    ),
    "tools": (
        "🧰 Инструменты\n\n"
        "/id — твой ID и кнопки выбора пользователя, бота, группы или канала\n"
        "/id @username — ID доступной публичной группы или канала\n"
        "Ответь командой /id на сообщение, чтобы узнать ID отправителя.\n\n"
        "Также можно попросить проверить сайт, найти свежую информацию, курс, "
        "погоду или текущее время."
    ),
    "media": (
        "🖼 Медиа и файлы\n\n"
        "/download <ссылка> — видео из YouTube, Instagram, TikTok, Vimeo, X/Twitter или VK "
        "до 10 минут и 100 МБ.\n\n"
        "/source — ответом на фото арта; отправляет уменьшенную копию "
        "в SauceNAO для поиска похожих публикаций.\n\n"
        "Дельта рассматривает изображения и стикеры, извлекает кадры из GIF и "
        "видео, читает PDF, DOCX, XLSX и текстовые файлы, распознаёт речь и "
        "анализирует звук. Добавь подпись или вопрос, если нужен конкретный разбор."
    ),
}


def _keyboard(page: str, mini_app_url: str | None) -> InlineKeyboardMarkup:
    """Собрать клавиатуру главной страницы или кнопку возврата."""
    if page != "main":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="⬅️ Назад",
                        callback_data=f"{_PREFIX}:main",
                    )
                ]
            ]
        )

    rows = [
        [
            InlineKeyboardButton(
                text="💬 Общение и RP", callback_data=f"{_PREFIX}:chat"
            ),
            InlineKeyboardButton(text="🎨 Арты", callback_data=f"{_PREFIX}:art"),
        ],
        [
            InlineKeyboardButton(
                text="🧰 Инструменты", callback_data=f"{_PREFIX}:tools"
            ),
            InlineKeyboardButton(text="🖼 Медиа", callback_data=f"{_PREFIX}:media"),
        ],
    ]
    if mini_app_url is not None:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🚀 Открыть Mini App",
                    web_app=WebAppInfo(url=mini_app_url),
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def create_menu_router(
    mini_app_url: str | None = None, user_states: UserStateStore | None = None
) -> Router:
    """Создать роутер панели управления и её навигации."""
    router = Router(name=__name__)

    @router.message(Command("menu"))
    async def menu_command(message: Message) -> None:
        """Показать главную страницу панели."""
        await message.answer(
            _PAGES["main"],
            reply_markup=_keyboard("main", mini_app_url),
        )

    @router.callback_query(F.data.startswith(f"{_PREFIX}:"))
    async def menu_callback(callback: CallbackQuery) -> None:
        """Переключить раздел, редактируя существующее сообщение."""
        page = (callback.data or "").partition(":")[2]
        text = _PAGES.get(page)
        if text is None:
            await callback.answer("Этот раздел больше недоступен.", show_alert=True)
            return

        if page == "art" and user_states is not None:
            async with user_states.use(callback.from_user.id) as state:
                if state.content_mode == "adult":
                    text += (
                        "\n\n/randomart — случайный арт из общей коллекции. "
                        "Выданные посты /e6 сохраняются в неё автоматически."
                    )

        if isinstance(callback.message, Message):
            await callback.message.edit_text(
                text,
                reply_markup=_keyboard(page, mini_app_url),
            )
        await callback.answer()

    return router
