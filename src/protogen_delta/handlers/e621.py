"""Альбомы и индивидуальная панель поиска e621/e926."""

import asyncio
import calendar
import logging
import re
import secrets
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from html import escape
from time import monotonic, time
from typing import cast

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaAudio,
    InputMediaDocument,
    InputMediaLivePhoto,
    InputMediaPhoto,
    InputMediaVideo,
    Message,
)

from protogen_delta.core.user_state import ContentMode, UserStateStore
from protogen_delta.repositories.e621_history import (
    E621HistoryRepository,
    E621Preferences,
    MediaFilter,
    PopularPeriod,
    SearchOrder,
)
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.services.art_gallery import remember_art
from protogen_delta.services.e621 import (
    E621Client,
    E621Error,
    E621Post,
    E621Query,
    E621QueryError,
    looks_like_e621_query,
    normalize_e621_query,
)
from protogen_delta.services.e621_media import E621MediaService, PreparedMedia

logger = logging.getLogger(__name__)
E621_USAGE = (
    "🎨 Поиск e621 по тегам\n\n"
    "/e6 dragon — найти арты\n"
    "/e6 dragon order:favcount count:10 — до 10 постов альбомом\n"
    "/e6 dragon -male — исключить тег\n"
    "В панели выбери арты, видео/GIF или всё; настройки сохраняются. "
    "🔥 Popular — подборки за день, неделю или месяц по оценке поста. "
    "Топ одинаков для всех, история просмотренного индивидуальна. "
    "Для разнообразия выбери «Случайно». /adult задаёт доступный рейтинг."
)
_NO_RESULTS = "По этим тегам свежих результатов не нашлось. Попробуй изменить запрос."
_CALLBACK_PREFIX = "e6:next"
_CONTROL_PREFIX = "e6:ctl"
_IMAGE_TYPES = {"jpg", "jpeg", "png", "webp"}
_VIDEO_TYPES = {"gif", "webm", "mp4"}
_BATCH_SECONDS = 150.0
_BUFFER_BYTES = 96 * 1024 * 1024


@dataclass(slots=True)
class _SearchSession:
    raw: str
    preferences: E621Preferences
    count: int = 1
    query: E621Query | None = None
    page: int = 1
    pending: list[E621Post] = field(default_factory=list)
    token: str = field(default_factory=lambda: secrets.token_hex(4))
    panel: Message | None = None
    task: asyncio.Task[object] | None = None
    expanded: bool = False
    closed: bool = False
    results: list[E621Post] = field(default_factory=list)
    progress: str = ""
    progress_at: float = 0
    popular_menu: bool = False
    anchor: date = field(default_factory=lambda: datetime.now(timezone.utc).date())


def _popular_dates(period: PopularPeriod, anchor: date) -> tuple[date, date]:
    if period == "week":
        start = anchor - timedelta(days=anchor.weekday())
        return start, start + timedelta(days=6)
    if period == "month":
        return anchor.replace(day=1), anchor.replace(
            day=calendar.monthrange(anchor.year, anchor.month)[1]
        )
    return anchor, anchor


def _shift_period(period: PopularPeriod, anchor: date, delta: int) -> date:
    if period == "month":
        year, month = divmod(anchor.year * 12 + anchor.month - 1 + delta, 12)
        return date(year, month + 1, 1)
    return anchor + timedelta(days=delta * (7 if period == "week" else 1))


def _split_count(raw: str) -> tuple[str, int]:
    tokens = raw.split()
    counts = [token for token in tokens if token.casefold().startswith("count:")]
    if (
        not counts
        and tokens
        and tokens[-1].isdigit()
        and any(t.casefold().startswith("order:") for t in tokens)
    ):
        counts = ["count:" + tokens.pop()]
    if len(counts) > 1 or (
        counts and not re.fullmatch(r"count:(?:[1-9]|10)", counts[0], re.I)
    ):
        raise E621QueryError("Количество задаётся как count:1 … count:10.")
    return " ".join(t for t in tokens if not t.casefold().startswith("count:")), (
        int(counts[0].split(":")[1]) if counts else 1
    )


def _query(
    raw: str,
    preferences: E621Preferences,
    content_mode: ContentMode,
    anchor: date | None = None,
) -> E621Query:
    tokens = raw.split()
    if preferences.period != "all":
        anchor = anchor or datetime.now(timezone.utc).date()
        start, end = _popular_dates(preferences.period, anchor)
        tokens = [t for t in tokens if not t.casefold().startswith(("order:", "date:"))]
        # API сравнивает время с полуночью: включаем начало и исключаем следующий день.
        stop = end + timedelta(days=1)
        tokens.extend(
            [f"date:>={start.isoformat()}", f"date:<{stop.isoformat()}", "order:score"]
        )
    positive_types = {
        t.casefold().split(":", 1)[1]
        for t in tokens
        if t.casefold().startswith("type:")
    }
    if preferences.media_filter == "images":
        if positive_types & _VIDEO_TYPES:
            raise E621QueryError(
                "В запросе указан видеоформат, но выбраны только арты. Выбери «Всё» или убери type:."
            )
        tokens.extend("-type:" + kind for kind in sorted(_VIDEO_TYPES | {"swf"}))
    elif preferences.media_filter == "videos":
        if positive_types & _IMAGE_TYPES:
            raise E621QueryError(
                "В запросе указан формат картинки, но выбраны видео/GIF. Выбери «Всё» или убери type:."
            )
        tokens.extend("-type:" + kind for kind in sorted(_IMAGE_TYPES | {"swf"}))
    if preferences.order != "site" and not any(
        t.casefold().startswith("order:") for t in tokens
    ):
        tokens.append("order:" + preferences.order)
    return normalize_e621_query(" ".join(dict.fromkeys(tokens)), content_mode)


def _allowed(post: E621Post, preferences: E621Preferences, adult: bool) -> bool:
    return (adult or post.rating == "s") and (
        preferences.media_filter == "all"
        or (preferences.media_filter == "images" and post.file_ext in _IMAGE_TYPES)
        or (preferences.media_filter == "videos" and post.file_ext in _VIDEO_TYPES)
    )


def _caption(post: E621Post) -> str:
    rating = {"s": "safe", "q": "questionable", "e": "explicit"}.get(
        post.rating, post.rating
    )
    artists = escape(", ".join(post.artists[:3])[:180] or "не указан")
    links = f'<a href="{post.page_url}">Открыть пост</a>'
    if (
        post.sources
        and len(post.sources[0]) <= 600
        and post.sources[0].startswith(("https://", "http://"))
    ):
        links += f' · <a href="{escape(post.sources[0], quote=True)}">Источник</a>'
    return f"e621 #{post.post_id} · {rating}\n❤️ {post.fav_count} · Автор: {artists}\n{links}"


def _keyboard(user_id: int, session: _SearchSession) -> InlineKeyboardMarkup:
    def button(label: str, action: str, selected: bool = False) -> InlineKeyboardButton:
        return InlineKeyboardButton(
            text=("✅ " if selected else "") + label,
            callback_data=f"{_CONTROL_PREFIX}:{user_id}:{session.token}:{action}",
        )

    pref = session.preferences
    effective_order = next(
        (
            t.split(":", 1)[1]
            for t in session.raw.split()
            if t.casefold().startswith("order:")
        ),
        pref.order,
    )
    rows = [
        [
            button("Арты", "images", pref.media_filter == "images"),
            button("Видео/GIF", "videos", pref.media_filter == "videos"),
            button("Всё", "all", pref.media_filter == "all"),
        ]
    ]
    if session.expanded:
        rows.extend(
            [
                [
                    button(
                        "Новые",
                        "site",
                        effective_order == "site" and pref.period == "all",
                    ),
                    button(
                        "Топ избранного",
                        "favcount",
                        effective_order == "favcount" and pref.period == "all",
                    ),
                    button(
                        "Случайно",
                        "random",
                        effective_order == "random" and pref.period == "all",
                    ),
                ],
                [
                    button(str(count), f"count{count}", session.count == count)
                    for count in (1, 5, 10)
                ],
            ]
        )
    rows.append([button("🔥 Popular", "popular", pref.period != "all")])
    if session.popular_menu or pref.period != "all":
        rows.append(
            [
                button(label, period, pref.period == period)
                for label, period in (
                    ("День", "day"),
                    ("Неделя", "week"),
                    ("Месяц", "month"),
                )
            ]
        )
        if pref.period != "all":
            rows.append(
                [
                    button("← Период", "prevperiod"),
                    button("Текущий", "nowperiod"),
                    button("Период →", "nextperiod"),
                ]
            )
    rows.append([button("🔄 Ещё", "next"), button("⚙️ Настройки", "settings")])
    rows.append(
        [
            button(
                "⏹ Стоп" if session.task else "✖ Скрыть панель",
                "stop" if session.task else "close",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _panel_text(session: _SearchSession) -> str:
    modes = {
        "all": "арты + видео/GIF",
        "images": "только арты",
        "videos": "только видео/GIF",
    }
    text = "🎨 e621 · " + modes[session.preferences.media_filter]
    if session.raw or session.preferences.period != "all":
        text += (
            "\nПоиск: "
            + escape(session.raw or "все теги")
            + f"\nДо {session.count} результатов за раз."
        )
    else:
        text += "\n" + escape(E621_USAGE)
    if session.preferences.period != "all":
        start, end = _popular_dates(session.preferences.period, session.anchor)
        text += (
            f"\n🔥 Popular: {start:%d.%m.%Y} — {end:%d.%m.%Y} (UTC). По оценке поста."
        )
    if session.progress:
        text += "\n\n" + escape(session.progress)
    if session.results:
        text += "\nПосты: " + " · ".join(
            f'<a href="{p.page_url}">{i}</a>' for i, p in enumerate(session.results, 1)
        )
    return text


def create_e621_router(
    client: E621Client,
    history: E621HistoryRepository,
    user_states: UserStateStore,
    *,
    bot_id: int = 0,
    images_repository: ImagesRepository | None = None,
) -> Router:
    router = Router(name=__name__)
    sessions: dict[int, _SearchSession] = {}
    locks: dict[int, asyncio.Lock] = {}
    media = E621MediaService(client, history, bot_id)

    async def discard_panel(session: _SearchSession) -> None:
        if session.panel:
            try:
                await session.panel.delete()
            except TelegramBadRequest:
                logger.info("E621 panel cleanup unavailable")
            session.panel = None

    async def update_panel(
        message: Message, user_id: int, session: _SearchSession
    ) -> None:
        if session.closed:
            return
        text, markup = _panel_text(session), _keyboard(user_id, session)
        if session.panel is None:
            response = await message.answer(
                text,
                parse_mode="HTML",
                reply_markup=markup,
                disable_web_page_preview=True,
            )
            if isinstance(response, Message):
                session.panel = response
        else:
            try:
                await session.panel.edit_text(
                    text,
                    parse_mode="HTML",
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )
            except TelegramBadRequest as error:
                reason = str(error).lower()
                if "message to edit not found" in reason:
                    session.panel = None
                    await update_panel(message, user_id, session)
                elif "message is not modified" not in reason:
                    raise

    async def current_query(
        user_id: int, session: _SearchSession
    ) -> tuple[E621Query, bool]:
        async with user_states.use(user_id) as state:
            return (
                _query(
                    session.raw, session.preferences, state.content_mode, session.anchor
                ),
                state.content_mode == "adult",
            )

    async def pick(
        session: _SearchSession, reserved: set[int], adult: bool
    ) -> E621Post | None:
        assert session.query is not None
        for page_attempt in range(6):
            while session.pending:
                post = session.pending.pop(0)
                if post.post_id not in reserved and _allowed(
                    post, session.preferences, adult
                ):
                    reserved.add(post.post_id)
                    return post
            if page_attempt == 5:
                return None
            session.pending.extend(
                await client.search(session.query, page=session.page)
            )
            session.page += 1
            if not session.pending:
                return None
        return None

    async def send_group(
        message: Message,
        user_id: int,
        session: _SearchSession,
        prepared: list[PreparedMedia],
        *,
        retry: bool = True,
    ) -> None:
        query, adult = await current_query(user_id, session)
        if query != session.query or any(
            not _allowed(p.post, session.preferences, adult) for p in prepared
        ):
            raise E621QueryError(
                "Возрастной режим или настройки изменились. Запусти поиск заново."
            )
        try:
            if len(prepared) > 1:
                items: list[
                    InputMediaAudio
                    | InputMediaDocument
                    | InputMediaLivePhoto
                    | InputMediaPhoto
                    | InputMediaVideo
                ] = []
                for index, p in enumerate(prepared, len(session.results) + 1):
                    caption = (
                        f"{index}. "
                        + _caption(p.post)
                        + ("\n" + escape(p.notice) if p.notice else "")
                    )
                    if p.kind == "mp4":
                        items.append(
                            InputMediaVideo(
                                media=p.input_file(),
                                caption=caption,
                                parse_mode="HTML",
                                supports_streaming=True,
                            )
                        )
                    else:
                        items.append(
                            InputMediaPhoto(
                                media=p.input_file(), caption=caption, parse_mode="HTML"
                            )
                        )
                delivered = await message.answer_media_group(items)
            else:
                p = prepared[0]
                caption = _caption(p.post) + (
                    "\n" + escape(p.notice) if p.notice else ""
                )
                if p.kind == "mp4":
                    result = await message.answer_video(
                        p.input_file(),
                        caption=caption,
                        parse_mode="HTML",
                        supports_streaming=True,
                    )
                elif p.kind == "gif":
                    result = await message.answer_animation(
                        p.input_file(), caption=caption, parse_mode="HTML"
                    )
                elif p.kind == "webm":
                    result = await message.answer_document(
                        p.input_file(), caption=caption, parse_mode="HTML"
                    )
                else:
                    result = await message.answer_photo(
                        p.input_file(), caption=caption, parse_mode="HTML"
                    )
                delivered = [result]
        except TelegramBadRequest:
            if not retry or not any(isinstance(p.content, str) for p in prepared):
                raise
            for p in prepared:
                await media.invalidate(p)
            refreshed = [
                await media.prepare(p.post, album=session.count > 1) for p in prepared
            ]
            await send_group(message, user_id, session, refreshed, retry=False)
            return
        for p, result in zip(prepared, delivered, strict=True):
            await history.mark_seen(user_id, p.post.post_id, time())
            if isinstance(result, Message):
                await media.remember(p, result)
                if images_repository is not None:
                    try:
                        remember_art(images_repository, result)
                    except OSError, ValueError, TypeError:
                        logger.warning(
                            "Пост %s доставлен, но не сохранён в общую коллекцию",
                            p.post.post_id,
                        )
            session.results.append(p.post)

    async def send_batch(
        message: Message, user_id: int, session: _SearchSession
    ) -> None:
        session.task = asyncio.current_task()
        session.results.clear()
        prepared: list[PreparedMedia] = []
        pending: list[tuple[E621Post, asyncio.Task[PreparedMedia]]] = []
        scheduled = 0
        completed = 0
        exhausted = False
        skipped = 0
        started = monotonic()

        async def stop_preparation(*, keep_ready: bool = False) -> None:
            nonlocal skipped
            unfinished: list[E621Post] = []
            tasks = [task for _, task in pending]
            for post, task in pending:
                if keep_ready and task.done() and not task.cancelled():
                    try:
                        prepared.append(task.result())
                    except E621Error:
                        skipped += 1
                    except TimeoutError:
                        unfinished.append(post)
                else:
                    unfinished.append(post)
                    task.cancel()
            pending.clear()
            # Неготовые посты доступны по «Ещё» после таймаута или отмены.
            session.pending[:0] = unfinished
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        session.progress = (
            "Подбираю и готовлю медиа… Видео может потребовать конвертацию."
        )
        try:
            query, adult = await current_query(user_id, session)
            if query != session.query:
                session.query, session.page, session.pending = query, 1, []
            await update_panel(message, user_id, session)
            reserved = await history.seen_ids(user_id)
            while completed < session.count:
                async with asyncio.timeout(
                    max(0, _BATCH_SECONDS - (monotonic() - started))
                ):
                    while (
                        len(pending) < 2 and scheduled < session.count and not exhausted
                    ):
                        post = await pick(session, reserved, adult)
                        if post is None:
                            exhausted = True
                            break
                        pending.append(
                            (
                                post,
                                asyncio.create_task(
                                    media.prepare(post, album=session.count > 1)
                                ),
                            )
                        )
                        scheduled += 1
                    if not pending:
                        break
                    session.progress = f"Готовлю {completed + 1}/{session.count}…"
                    if monotonic() - session.progress_at >= 1:
                        await update_panel(message, user_id, session)
                        session.progress_at = monotonic()
                    try:
                        p = await pending[0][1]
                    except E621Error:
                        skipped += 1
                        pending.pop(0)
                        completed += 1
                        continue
                    pending.pop(0)
                    completed += 1
                prepared.append(p)
                if (
                    len(prepared) >= 2
                    and sum(item.size for item in prepared) >= _BUFFER_BYTES
                ):
                    await send_group(message, user_id, session, prepared)
                    prepared.clear()
            if prepared:
                session.progress = "Отправляю подборку…"
                await update_panel(message, user_id, session)
                await send_group(message, user_id, session, prepared)
            session.progress = (
                f"Готово: {len(session.results)}. Открой элемент альбома для подписи и источника."
                if session.results
                else _NO_RESULTS
            )
            if skipped:
                session.progress += f" Недоступных файлов: {skipped}."
        except TimeoutError:
            await stop_preparation(keep_ready=True)
            if prepared:
                try:
                    await send_group(message, user_id, session, prepared)
                except E621Error, TelegramBadRequest:
                    session.progress = (
                        "Не удалось отправить подготовленную часть. "
                        "Запусти поиск заново."
                    )
                    return
            session.progress = (
                "Подготовка достигла лимита времени. "
                f"Отправлено: {len(session.results)}. "
                "Можно продолжить или выбрать только арты."
            )
        except asyncio.CancelledError:
            session.progress = f"Поиск остановлен. Отправлено: {len(session.results)}."
        except (E621Error, TelegramBadRequest) as error:
            reason = (
                "photo_dimensions"
                if isinstance(error, TelegramBadRequest)
                and any(
                    t in error.message.casefold()
                    for t in (
                        "photo_invalid_dimensions",
                        "image_process_failed",
                        "photo_invalid",
                        "file is too big",
                    )
                )
                else "other"
            )
            logger.info(
                "E621 delivery outcome=failed type=%s reason=%s",
                type(error).__name__,
                reason,
            )
            session.progress = (
                str(error)
                if isinstance(error, E621Error)
                else "Telegram не принял вложение. Попробуй другую подборку."
            )
        finally:
            await stop_preparation()
            session.task = None
            prepared.clear()
            if session.results and session.panel and not session.closed:
                await discard_panel(session)
            await update_panel(message, user_id, session)

    async def run_query(message: Message, raw: str) -> None:
        if message.from_user is None:
            return
        user_id = message.from_user.id
        lock = locks.setdefault(user_id, asyncio.Lock())
        if lock.locked():
            await message.answer(
                "Подборка ещё готовится. Дождись её или нажми «Стоп» в панели e621."
            )
            return
        async with lock:
            await run_locked_query(message, raw, user_id)

    async def run_locked_query(message: Message, raw: str, user_id: int) -> None:
        try:
            preferences = await history.preferences(user_id)
            tags, count = _split_count(raw)
            if any(t.casefold().startswith(("order:", "date:")) for t in tags.split()):
                preferences = replace(preferences, period="all")
            explicit_count = any(
                t.casefold().startswith("count:") for t in raw.split()
            ) or tags != " ".join(raw.split())
            session = _SearchSession(
                tags, preferences, count if explicit_count else preferences.count
            )
            await current_query(user_id, session)
            previous = sessions.get(user_id)
            if previous and previous.panel:
                await discard_panel(previous)
            sessions[user_id] = session
            await send_batch(message, user_id, session)
        except E621QueryError as error:
            await message.answer(str(error))

    @router.message(Command("e6"))
    async def e621_command(message: Message) -> None:
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) > 1:
            await run_query(message, parts[1])
        elif message.from_user:
            user_id = message.from_user.id
            lock = locks.setdefault(user_id, asyncio.Lock())
            if lock.locked():
                await message.answer(
                    "Подборка ещё готовится. Дождись её или нажми «Стоп»."
                )
                return
            async with lock:
                preferences = await history.preferences(user_id)
                session = sessions.setdefault(
                    user_id, _SearchSession("", preferences, preferences.count)
                )
                await discard_panel(session)
                session.token = secrets.token_hex(4)
                session.closed, session.expanded = False, True
                session.progress = ""
                session.results.clear()
                await update_panel(message, user_id, session)

    @router.message(F.text.func(looks_like_e621_query))
    async def e621_direct_query(message: Message) -> None:
        await run_query(message, message.text or "")

    @router.callback_query(F.data.startswith(f"{_CALLBACK_PREFIX}:"))
    async def e621_next(callback: CallbackQuery) -> None:
        parts = (callback.data or "").split(":")
        expected = parts[2] if len(parts) == 3 else ""
        if expected != str(callback.from_user.id):
            await callback.answer("Эта кнопка предназначена не тебе.", show_alert=True)
            return
        await callback.answer("Поиск устарел. Запусти /e6 ещё раз.", show_alert=True)

    @router.callback_query(F.data.startswith(f"{_CONTROL_PREFIX}:"))
    async def controls(callback: CallbackQuery) -> None:
        parts = (callback.data or "").split(":")
        user_id = callback.from_user.id
        if len(parts) != 5 or parts[2] != str(user_id):
            await callback.answer("Эта кнопка предназначена не тебе.", show_alert=True)
            return
        session = sessions.get(user_id)
        if (
            session is None
            or session.token != parts[3]
            or not isinstance(callback.message, Message)
        ):
            await callback.answer(
                "Панель устарела. Открой /e6 заново.", show_alert=True
            )
            return
        action = parts[4]
        if action in {"stop", "close"}:
            await callback.answer()
            if action == "close":
                session.closed = True
                await callback.message.edit_reply_markup(reply_markup=None)
            if session.task:
                session.task.cancel()
            return
        if session.task:
            await callback.answer(
                "Подборка готовится. Можно остановить её кнопкой «Стоп»."
            )
            return
        await callback.answer()
        if action == "next":
            if not session.raw and session.preferences.period == "all":
                session.progress = "Укажи запрос: /e6 dragon"
                await update_panel(callback.message, user_id, session)
                return
            lock = locks.setdefault(user_id, asyncio.Lock())
            if lock.locked():
                return
            async with lock:
                await send_batch(callback.message, user_id, session)
            return
        if action == "settings":
            session.expanded = not session.expanded
        elif action == "popular":
            session.popular_menu = not session.popular_menu
        elif action in {
            "day",
            "week",
            "month",
            "prevperiod",
            "nextperiod",
            "nowperiod",
        }:
            lock = locks.setdefault(user_id, asyncio.Lock())
            if lock.locked():
                return
            async with lock:
                if action in {"day", "week", "month"}:
                    session.preferences = replace(
                        session.preferences, period=cast(PopularPeriod, action)
                    )
                    session.anchor = datetime.now(timezone.utc).date()
                elif session.preferences.period == "all":
                    return
                elif action == "nowperiod":
                    session.anchor = datetime.now(timezone.utc).date()
                else:
                    session.anchor = _shift_period(
                        session.preferences.period,
                        session.anchor,
                        -1 if action == "prevperiod" else 1,
                    )
                session.raw = " ".join(
                    t
                    for t in session.raw.split()
                    if not t.casefold().startswith(("order:", "date:"))
                )
                session.query = None
                await history.save_preferences(user_id, session.preferences)
                await send_batch(callback.message, user_id, session)
            return
        elif action in {"all", "images", "videos"}:
            session.preferences = replace(
                session.preferences, media_filter=cast(MediaFilter, action)
            )
        elif action in {"site", "favcount", "random"}:
            session.preferences = replace(
                session.preferences, order=cast(SearchOrder, action), period="all"
            )
            session.raw = " ".join(
                t for t in session.raw.split() if not t.casefold().startswith("order:")
            )
        elif action in {"count1", "count5", "count10"}:
            session.count = int(action[5:])
            session.preferences = replace(session.preferences, count=session.count)
        else:
            return
        await history.save_preferences(user_id, session.preferences)
        session.progress = "Настройки сохранены. Нажми «Ещё» или отправь новый запрос."
        await update_panel(callback.message, user_id, session)

    return router
