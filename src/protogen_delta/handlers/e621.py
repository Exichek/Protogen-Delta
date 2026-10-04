"""Поиск артов e621/e926 по тегам без участия языковой модели."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from time import time
from urllib.parse import urlsplit

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
    MAX_IMAGE_BYTES,
    MAX_VIDEO_BYTES,
    E621Client,
    E621Error,
    E621Post,
    E621Query,
    E621QueryError,
    looks_like_e621_query,
    normalize_e621_query,
)
from protogen_delta.services.telegram_video import TelegramVideoConverter

logger = logging.getLogger(__name__)

E621_USAGE = (
    "🎨 Поиск артов e621 по тегам\n\n"
    "Примеры:\n"
    "• /e6 dragon — арты с драконом\n"
    "• /e6 dragon order:favcount — сначала самые популярные\n"
    "• /e6 dragon order:favcount count:10 — прислать до 10 постов\n"
    "• /e6 dragon -male — исключить тег male\n"
    "• /e6 dragon rating:e — только explicit в режиме 18+\n\n"
    "Теги пишутся по-английски через пробел. Явный запрос вроде "
    "dragon order:favcount можно отправить и без /e6. Кнопка «Ещё» продолжит "
    "тот же поиск без уже показанных постов. /adult определяет доступные рейтинги: "
    "в мягком режиме выдаётся только safe, во взрослом доступны safe, "
    "questionable и explicit."
)
_NO_RESULTS = "По этим тегам свежих результатов не нашлось. Попробуй изменить запрос."
_CALLBACK_PREFIX = "e6:next"


@dataclass(slots=True)
class _SearchSession:
    query: E621Query
    page: int = 1
    pending: list[E621Post] = field(default_factory=list)
    count: int = 1


def _split_count(raw: str) -> tuple[str, int]:
    """Количество выдачи — параметр бота, не тег e621."""
    tokens = raw.split()
    counts = [token for token in tokens if token.casefold().startswith("count:")]
    if (
        not counts
        and tokens
        and tokens[-1].isdigit()
        and any(token.casefold().startswith("order:") for token in tokens)
    ):
        counts = ["count:" + tokens.pop()]
    if len(counts) > 1 or (
        counts and not re.fullmatch(r"count:(?:[1-9]|10)", counts[0], re.I)
    ):
        raise E621QueryError("Количество задаётся как count:1 … count:10.")
    count = int(counts[0].split(":")[1]) if counts else 1
    return (
        " ".join(
            token for token in tokens if not token.casefold().startswith("count:")
        ),
        count,
    )


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
    locks: dict[int, asyncio.Lock] = {}
    converter = TelegramVideoConverter()

    async def send_next(
        message: Message, user_id: int, session: _SearchSession
    ) -> bool:
        # Старые кнопки не должны обходить новый возрастной режим.
        async with user_states.use(user_id) as state:
            current = normalize_e621_query(session.query.tags, state.content_mode)
        if current != session.query:
            session.query = current
            session.page = 1
            session.pending.clear()
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
            return False
        url = post.media_url
        if url is None:
            await message.answer(
                "У поста нет доступного файла.", reply_markup=_keyboard(user_id, post)
            )
            return False
        original = url == post.file_url
        ext = post.file_ext if original else "jpg"
        caption = _caption(post)[:800]
        data: bytes | None = None
        if post.file_ext in {"mp4", "webm"}:
            for alternate in post.mp4_urls + post.webm_urls:
                try:
                    data = await client.download(alternate, max_bytes=MAX_VIDEO_BYTES)
                    ext = urlsplit(alternate).path.rsplit(".", 1)[-1].lower()
                    break
                except E621Error:
                    continue
        if data is None:
            ext = urlsplit(url).path.rsplit(".", 1)[-1].lower() if not original else ext
            max_bytes = MAX_VIDEO_BYTES if ext in {"mp4", "webm"} else MAX_IMAGE_BYTES
            data = await client.download(url, max_bytes=max_bytes)
            if post.file_ext in {"mp4", "webm"} and ext not in {"mp4", "webm"}:
                caption += "\n⚠️ Видео превышает лимит; отправлено превью. Оригинал — по кнопке."
        if ext == "webm":
            try:
                data = await converter.convert(data)
                ext = "mp4"
            except E621Error:
                caption += (
                    "\n⚠️ Не удалось подготовить MP4; отправляю исходный WebM файлом."
                )
        media = BufferedInputFile(data, filename=_filename(post, ext))
        markup = _keyboard(user_id, post)
        if ext == "gif":
            await message.answer_animation(media, caption=caption, reply_markup=markup)
        elif ext == "mp4":
            await message.answer_video(
                media,
                caption=caption[:1024],
                reply_markup=markup,
                supports_streaming=True,
            )
        elif ext == "webm":
            # Bot API гарантирует sendVideo только для MPEG-4. WebM отправляем
            # исходным файлом, чтобы не заменять ролик статичным preview.
            await message.answer_document(media, caption=caption, reply_markup=markup)
        else:
            await message.answer_photo(media, caption=caption, reply_markup=markup)
        await history.mark_seen(user_id, post.post_id, time())
        return True

    async def send_batch(
        message: Message, user_id: int, session: _SearchSession
    ) -> None:
        """Сохранить порядок API и отмечать только успешно доставленные посты."""
        for _ in range(session.count):
            if not await send_next(message, user_id, session):
                break

    async def run_query(message: Message, raw: str) -> None:
        if message.from_user is None:
            return
        user_id = message.from_user.id
        try:
            raw, count = _split_count(raw)
            async with user_states.use(user_id) as state:
                query = normalize_e621_query(raw, state.content_mode)
            async with locks.setdefault(user_id, asyncio.Lock()):
                session = _SearchSession(query=query, count=count)
                sessions[user_id] = session
                await send_batch(message, user_id, session)
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
            async with locks.setdefault(expected_user_id, asyncio.Lock()):
                await send_batch(callback.message, expected_user_id, session)
        except E621Error as error:
            await callback.message.answer(str(error))

    return router
