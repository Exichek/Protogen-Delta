"""Поиск артов e621/e926 по тегам без участия языковой модели."""

import logging
from dataclasses import dataclass, field
from time import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.e621_history import E621HistoryRepository
from protogen_delta.services.e621 import (
    E621Client,
    E621Error,
    E621Post,
    E621Query,
    E621QueryError,
    looks_like_e621_query,
    normalize_e621_query,
)

logger = logging.getLogger(__name__)

E621_USAGE = (
    "🎨 Поиск по тегам e621: /e6 dragon order:favcount\n\n"
    "Можно отправить явный tag-query без команды, например "
    "dragon order:favcount. Теги разделяются пробелами, исключение — через -tag. "
    "Кнопка «Ещё» продолжает тот же поиск без повторов. Возрастной режим /adult "
    "определяет доступный рейтинг."
)
_NO_RESULTS = "По этим тегам свежих результатов не нашлось. Попробуй изменить запрос."
_CALLBACK_PREFIX = "e6:next"


@dataclass(slots=True)
class _SearchSession:
    query: E621Query
    page: int = 1
    pending: list[E621Post] = field(default_factory=list)


def _keyboard(user_id: int, post: E621Post) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Ещё",
                    callback_data=f"{_CALLBACK_PREFIX}:{user_id}",
                ),
                InlineKeyboardButton(text="🔗 Открыть e621", url=post.page_url),
            ]
        ]
    )


def _caption(post: E621Post) -> str:
    rating = {"s": "safe", "q": "questionable", "e": "explicit"}.get(
        post.rating, post.rating
    )
    artists = ", ".join(post.artists[:4]) if post.artists else "не указан"
    source = post.sources[0] if post.sources else post.page_url
    return (
        f"e621 #{post.post_id} · рейтинг: {rating}\n"
        f"❤️ {post.fav_count} · рейтинг поста: {post.score}\n"
        f"Автор: {artists}\nИсточник: {source}"
    )[:1024]


def _filename(post: E621Post, ext: str | None = None) -> str:
    return f"e621-{post.post_id}.{ext or post.file_ext or 'jpg'}"


def create_e621_router(
    client: E621Client,
    history: E621HistoryRepository,
    user_states: UserStateStore,
) -> Router:
    """Создать команду, прямой tag-query и кнопку следующего результата."""
    router = Router(name=__name__)
    sessions: dict[int, _SearchSession] = {}

    async def send_next(
        message: Message, user_id: int, session: _SearchSession
    ) -> None:
        seen = await history.seen_ids(user_id)
        post: E621Post | None = None
        for _ in range(5):
            while session.pending:
                candidate = session.pending.pop(0)
                if candidate.post_id not in seen:
                    post = candidate
                    break
            if post is not None:
                break
            posts = await client.search(session.query, page=session.page)
            session.page += 1
            session.pending.extend(posts)
            if not posts:
                break
        if post is None:
            await message.answer(_NO_RESULTS)
            return
        url = post.media_url
        if url is None:
            await message.answer(
                "У поста нет доступного файла.", reply_markup=_keyboard(user_id, post)
            )
            return
        data = await client.download(url)
        original = url == post.file_url
        ext = post.file_ext if original else "jpg"
        media = BufferedInputFile(data, filename=_filename(post, ext))
        caption = _caption(post)
        markup = _keyboard(user_id, post)
        if ext == "gif":
            await message.answer_animation(media, caption=caption, reply_markup=markup)
        elif ext in {"mp4", "webm"}:
            await message.answer_video(media, caption=caption, reply_markup=markup)
        else:
            await message.answer_photo(media, caption=caption, reply_markup=markup)
        await history.mark_seen(user_id, post.post_id, time())

    async def run_query(message: Message, raw: str) -> None:
        if message.from_user is None:
            return
        user_id = message.from_user.id
        try:
            async with user_states.use(user_id) as state:
                query = normalize_e621_query(raw, state.content_mode)
            session = _SearchSession(query=query)
            sessions[user_id] = session
            await send_next(message, user_id, session)
        except E621QueryError as error:
            await message.answer(str(error))
        except E621Error as error:
            logger.info("Ошибка e621 для пользователя %s: %s", user_id, error)
            await message.answer(str(error))

    @router.message(Command("e6"))
    async def e621_command(message: Message) -> None:
        """Запустить поиск из аргументов команды."""
        text = message.text or ""
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            await message.answer(E621_USAGE)
            return
        await run_query(message, parts[1])

    @router.message(F.text.func(looks_like_e621_query))
    async def e621_direct_query(message: Message) -> None:
        """Обработать очевидный e621 tag-query без команды."""
        await run_query(message, message.text or "")

    @router.callback_query(F.data.startswith(f"{_CALLBACK_PREFIX}:"))
    async def e621_next(callback: CallbackQuery) -> None:
        """Продолжить сохранённый поиск только для владельца кнопки."""
        parts = (callback.data or "").split(":")
        try:
            expected_user_id = int(parts[2]) if len(parts) == 3 else 0
        except ValueError:
            expected_user_id = 0
        if callback.from_user.id != expected_user_id:
            await callback.answer("Эта кнопка предназначена не тебе.", show_alert=True)
            return
        session = sessions.get(expected_user_id)
        if session is None or not isinstance(callback.message, Message):
            await callback.answer(
                "Поиск устарел. Запусти /e6 ещё раз.", show_alert=True
            )
            return
        await callback.answer()
        try:
            await send_next(callback.message, expected_user_id, session)
        except E621Error as error:
            await callback.message.answer(str(error))

    return router
