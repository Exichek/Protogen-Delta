"""Проверить альбомы, настройки, отмену и повторную отправку без конвертации."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMediaGroup
from aiogram.types import CallbackQuery, Message
from test_e621 import (
    _JPEG,
    _call_message,
    _Client,
    _message,
    _post,
    _Response,
    _Session,
)

import protogen_delta.handlers.e621 as handler
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.e621_history import (
    CachedMedia,
    E621HistoryRepository,
    E621Preferences,
    MediaFilter,
    SearchOrder,
)
from protogen_delta.services.e621 import E621Client, E621Error, E621Post, E621QueryError
from protogen_delta.services.e621_media import E621MediaService, PreparedMedia


def _delivery(kind: str = "photo") -> Message:
    result = Mock(spec=Message)
    result.video = result.animation = result.document = None
    result.photo = []
    asset = SimpleNamespace(file_id="telegram-file-" + kind)
    if kind == "photo":
        result.photo = [asset]
    else:
        setattr(result, kind, asset)
    return cast(Message, result)


def _button(raw: Mock, action: str, *, user_id: int = 7) -> CallbackQuery:
    markup = (
        raw.panel.edit_text.await_args.kwargs["reply_markup"]
        if raw.panel.edit_text.await_args
        else raw.answer.await_args.kwargs["reply_markup"]
    )
    buttons = [b for row in markup.inline_keyboard for b in row]
    buttons.extend(
        b
        for row in raw.answer.await_args.kwargs["reply_markup"].inline_keyboard
        for b in row
    )
    data = next(
        b.callback_data for b in buttons if b.callback_data.endswith(":" + action)
    )
    callback = Mock(spec=CallbackQuery)
    callback.from_user = Mock(id=user_id)
    callback.data = data
    callback.message = raw
    callback.answer = AsyncMock()
    return cast(CallbackQuery, callback)


def test_preferences_and_cache_survive_restart_and_are_isolated(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = E621HistoryRepository(tmp_path)
        pref = E621Preferences("images", "random", 10)
        await repo.save_preferences(7, pref)
        await repo.cache_media(100, "asset", CachedMedia("file", "mp4"), 1)
        fresh = E621HistoryRepository(tmp_path)
        assert await fresh.preferences(7) == pref
        assert await fresh.preferences(8) == E621Preferences()
        assert await fresh.cached_media(100, "asset") == CachedMedia("file", "mp4")
        assert await fresh.cached_media(200, "asset") is None
        await fresh.forget_media(100, "asset")
        assert await fresh.cached_media(100, "asset") is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "pref",
    [
        E621Preferences(count=0),
        E621Preferences(count=11),
        E621Preferences(order=cast(SearchOrder, "bad")),
        E621Preferences(media_filter=cast(MediaFilter, "bad")),
    ],
)
def test_invalid_preferences_are_not_written(
    tmp_path: Path, pref: E621Preferences
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(E621HistoryRepository(tmp_path).save_preferences(7, pref))


@pytest.mark.parametrize("kind,excluded", [("images", "webm"), ("videos", "jpg")])
def test_media_filter_preserves_explicit_order_and_query_disjunction(
    kind: str, excluded: str
) -> None:
    pref = E621Preferences(media_filter=cast(MediaFilter, kind), order="random")
    query = handler._query("~dragon ~wolf order:favcount", pref, "adult")
    assert "~dragon ~wolf" in query.tags
    assert "order:favcount" in query.tags and "order:random" not in query.tags
    assert "-type:" + excluded in query.tags
    assert "order:random" in handler._query("dragon", pref, "adult").tags


@pytest.mark.parametrize(
    "pref,query",
    [(E621Preferences("images"), "type:gif"), (E621Preferences("videos"), "type:png")],
)
def test_conflicting_explicit_format_gets_explanation(
    pref: E621Preferences, query: str
) -> None:
    with pytest.raises(E621QueryError, match="Выбери"):
        handler._query(query, pref, "adult")


def test_caption_is_compact_escaped_and_does_not_break_long_source() -> None:
    post = replace(
        _post(), artists=("<author>&",), sources=("https://example.org/?a=1&b=2",)
    )
    caption = handler._caption(post)
    assert "&lt;author&gt;&amp;" in caption and "a=1&amp;b=2" in caption
    assert len(caption) < 1024
    assert "Источник" not in handler._caption(
        replace(post, sources=("https://example.org/" + "x" * 700,))
    )


def test_mixed_album_has_one_panel_and_reuses_files_between_accounts(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = cast(Mock, _Client([_post(1), _post(2, "mp4"), _post(3)]))
        client.download = AsyncMock(return_value=_JPEG)
        history = E621HistoryRepository(tmp_path)
        router = handler.create_e621_router(
            cast(E621Client, client), history, UserStateStore(), bot_id=100
        )
        first, raw = _message("/e6 dragon order:favcount count:3")
        raw.answer_media_group.side_effect = lambda items: [
            _delivery("video" if p.type == "video" else "photo") for p in items
        ]
        await _call_message(router, 0, first)
        items = raw.answer_media_group.await_args.args[0]
        assert [p.type for p in items] == ["photo", "video", "photo"]
        assert items[1].supports_streaming
        assert [p.caption.split(" ·")[0] for p in items] == [
            "1. e621 #1",
            "2. e621 #2",
            "3. e621 #3",
        ]
        raw.answer_photo.assert_not_awaited()
        raw.panel.delete.assert_awaited_once()
        assert "Готово: 3" in raw.answer.await_args.args[0]
        assert client.download.await_count == 3
        second, other = _message("/e6 dragon order:favcount count:3", user_id=8)
        await _call_message(router, 0, second)
        assert all(
            isinstance(p.media, str)
            for p in other.answer_media_group.await_args.args[0]
        )
        assert client.download.await_count == 3
        assert await history.seen_ids(7) == await history.seen_ids(8) == {1, 2, 3}

    asyncio.run(scenario())


@pytest.mark.parametrize("mode,ids", [("images", {1, 3}), ("videos", {2})])
def test_saved_filter_is_enforced_even_if_provider_ignores_it(
    tmp_path: Path, mode: str, ids: set[int]
) -> None:
    async def scenario() -> None:
        history = E621HistoryRepository(tmp_path)
        await history.save_preferences(
            7, E621Preferences(media_filter=cast(MediaFilter, mode), count=10)
        )
        client = _Client(
            [_post(1), _post(2, "mp4"), _post(3), replace(_post(4), rating="e")]
        )
        router = handler.create_e621_router(
            cast(E621Client, client), history, UserStateStore()
        )
        message, raw = _message()
        await _call_message(router, 0, message)
        assert await history.seen_ids(7) == ids
        assert "rating:s" in client.queries[0].tags
        assert "-type:" in client.queries[0].tags
        assert raw.answer_document.await_count == 0

    asyncio.run(scenario())


def test_panel_changes_persist_and_buttons_override_explicit_sort(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        history = E621HistoryRepository(tmp_path)
        client = _Client([_post(1), _post(2), _post(3)])
        router = handler.create_e621_router(
            cast(E621Client, client), history, UserStateStore()
        )
        message, raw = _message("/e6 dragon order:favcount")
        await _call_message(router, 0, message)
        for action in (
            "settings",
            "count5",
            "random",
            "images",
            "all",
            "videos",
            "site",
            "favcount",
            "count10",
            "count1",
            "images",
        ):
            await router.callback_query.handlers[1].callback(_button(raw, action))
        assert await history.preferences(7) == E621Preferences("images", "favcount", 1)
        await router.callback_query.handlers[1].callback(_button(raw, "next"))
        assert "order:favcount" in client.queries[-1].tags
        assert "-type:mp4" in client.queries[-1].tags
        callback = _button(raw, "close")
        await router.callback_query.handlers[1].callback(callback)
        raw.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
        await _call_message(router, 0, _message("/e6")[0])

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "case",
    ["owner", "malformed", "stale", "no_message", "unknown", "next_empty", "stop_idle"],
)
def test_panel_rejects_foreign_and_stale_buttons_without_search(
    tmp_path: Path, case: str
) -> None:
    async def scenario() -> None:
        client = _Client([])
        router = handler.create_e621_router(
            cast(E621Client, client), E621HistoryRepository(tmp_path), UserStateStore()
        )
        message, raw = _message("/e6")
        await _call_message(router, 0, message)
        cb = cast(Mock, _button(raw, "next"))
        if case == "owner":
            cb.from_user.id = 8
        elif case == "malformed":
            cb.data = "e6:ctl:7"
        elif case == "stale":
            cb.data = "e6:ctl:7:wrong:next"
        elif case == "no_message":
            cb.message = None
        elif case == "unknown":
            cb.data = cb.data.rsplit(":", 1)[0] + ":unknown"
        elif case == "stop_idle":
            cb.data = cb.data.rsplit(":", 1)[0] + ":stop"
        await router.callback_query.handlers[1].callback(cb)
        assert not client.queries
        cb.answer.assert_awaited_once()
        if case in {"owner", "malformed", "stale", "no_message"}:
            assert cb.answer.await_args.kwargs["show_alert"]
        if case == "next_empty":
            assert "Укажи запрос" in raw.panel.edit_text.await_args.args[0]

    asyncio.run(scenario())


def test_cancel_stops_download_and_busy_queries_do_not_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()

        async def waiting(*args: object, **kwargs: object) -> PreparedMedia:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("cancel did not stop preparation")

        monkeypatch.setattr(E621MediaService, "prepare", waiting)
        history = E621HistoryRepository(tmp_path)
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post()])), history, UserStateStore()
        )
        message, raw = _message()
        task = asyncio.create_task(_call_message(router, 0, message))
        await asyncio.wait_for(entered.wait(), 1)
        second, other = _message()
        await _call_message(router, 0, second)
        assert "ещё готовится" in other.answer.await_args.args[0]
        await router.callback_query.handlers[1].callback(_button(raw, "images"))
        await router.callback_query.handlers[1].callback(_button(raw, "stop"))
        await asyncio.wait_for(task, 1)
        assert "остановлен" in raw.panel.edit_text.await_args.args[0]
        assert await history.seen_ids(7) == set()

    asyncio.run(scenario())


def test_age_change_during_preparation_prevents_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = UserStateStore()
    states.get(7).content_mode = "adult"

    async def prepare(
        self: E621MediaService, post: object, **kwargs: object
    ) -> PreparedMedia:
        states.get(7).content_mode = "soft"
        return PreparedMedia(replace(_post(), rating="e"), "key", "jpg", _JPEG)

    monkeypatch.setattr(E621MediaService, "prepare", prepare)
    history = E621HistoryRepository(tmp_path)
    router = handler.create_e621_router(
        cast(E621Client, _Client([replace(_post(), rating="e")])), history, states
    )
    message, raw = _message()
    asyncio.run(_call_message(router, 0, message))
    raw.answer_photo.assert_not_awaited()
    assert "Возрастной режим" in raw.panel.edit_text.await_args.args[0]
    assert asyncio.run(history.seen_ids(7)) == set()


def test_invalid_cached_file_is_downloaded_again_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = cast(Mock, _Client([_post(1), _post(2)]))
        client.download = AsyncMock(return_value=_JPEG)
        history = E621HistoryRepository(tmp_path)
        service = E621MediaService(cast(E621Client, client), history, 100)
        for post in client.posts:
            p = await service.prepare(post, album=True)
            await service.remember(p, _delivery())
        client.download.reset_mock()
        router = handler.create_e621_router(
            cast(E621Client, client), history, UserStateStore(), bot_id=100
        )
        message, raw = _message("/e6 dragon count:2")
        raw.answer_media_group.side_effect = [
            TelegramBadRequest(
                SendMediaGroup(chat_id=7, media=[]), "wrong file identifier"
            ),
            [_delivery(), _delivery()],
        ]
        await _call_message(router, 0, message)
        assert raw.answer_media_group.await_count == 2
        assert client.download.await_count == 2
        assert await history.seen_ids(7) == {1, 2}

    asyncio.run(scenario())


def test_failed_upload_does_not_mark_seen(tmp_path: Path) -> None:
    history = E621HistoryRepository(tmp_path)
    router = handler.create_e621_router(
        cast(E621Client, _Client([_post(1), _post(2)])), history, UserStateStore()
    )
    message, raw = _message("/e6 dragon count:2")
    raw.answer_media_group.side_effect = TelegramBadRequest(
        SendMediaGroup(chat_id=7, media=[]), "bad media"
    )
    asyncio.run(_call_message(router, 0, message))
    assert asyncio.run(history.seen_ids(7)) == set()
    assert "не принял" in raw.panel.edit_text.await_args.args[0]


def test_timeout_sends_prepared_part_and_allows_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async def prepare(
            self: E621MediaService, post: E621Post, **kwargs: object
        ) -> PreparedMedia:
            if post.post_id == 2:
                await asyncio.sleep(1)
            return PreparedMedia(post, "key", "jpg", _JPEG)

        monkeypatch.setattr(E621MediaService, "prepare", prepare)
        monkeypatch.setattr(handler, "_BATCH_SECONDS", 0.1)
        history = E621HistoryRepository(tmp_path)
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post(1), _post(2)])), history, UserStateStore()
        )
        message, raw = _message("/e6 dragon count:2")
        await _call_message(router, 0, message)
        raw.answer_photo.assert_awaited_once()
        assert await history.seen_ids(7) == {1}
        assert "лимита времени" in raw.answer.await_args.args[0]

    asyncio.run(scenario())


def test_album_buffers_are_flushed_in_small_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(handler, "_BUFFER_BYTES", 1)
    history = E621HistoryRepository(tmp_path)
    router = handler.create_e621_router(
        cast(E621Client, _Client([_post(i) for i in range(1, 6)])),
        history,
        UserStateStore(),
    )
    message, raw = _message("/e6 dragon count:5")
    asyncio.run(_call_message(router, 0, message))
    assert [len(c.args[0]) for c in raw.answer_media_group.await_args_list] == [2, 2]
    raw.answer_photo.assert_awaited_once()
    assert asyncio.run(history.seen_ids(7)) == {1, 2, 3, 4, 5}


def test_media_preparation_deduplicates_concurrent_work_and_uses_cache(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = cast(Mock, _Client([]))
        client.download = AsyncMock(return_value=_JPEG)
        history = E621HistoryRepository(tmp_path)
        service = E621MediaService(cast(E621Client, client), history, 100)
        first, second = await asyncio.gather(
            service.prepare(_post(), album=True), service.prepare(_post(), album=False)
        )
        assert first.content is second.content and client.download.await_count == 1
        await service.remember(first, _delivery())
        cached = await service.prepare(_post(), album=False)
        assert cached.content == "telegram-file-photo" and cached.size == 0
        await service.invalidate(cached)
        await service.prepare(_post(), album=True)
        assert client.download.await_count == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["video", "animation", "document", "photo", "none"])
def test_cache_extracts_actual_telegram_file_id(tmp_path: Path, kind: str) -> None:
    async def scenario() -> None:
        history = E621HistoryRepository(tmp_path)
        service = E621MediaService(cast(E621Client, _Client([])), history, 1)
        prepared = PreparedMedia(_post(), "asset", "mp4", _JPEG)
        await service.remember(prepared, _delivery(kind))
        cached = await history.cached_media(1, "asset")
        assert (cached.file_id if cached else None) == (
            None if kind == "none" else "telegram-file-" + kind
        )
        await service.remember(replace(prepared, notice="preview"), _delivery())

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["download", "conversion", "no_preview"])
def test_album_fallback_is_explicit_and_not_cached_forever(
    tmp_path: Path, failure: str
) -> None:
    async def scenario() -> None:
        client = cast(Mock, _Client([]))
        client.download = AsyncMock(
            side_effect=lambda url, **kwargs: (
                _JPEG
                if "sample-" in url
                else (_ for _ in ()).throw(E621Error("unavailable"))
            )
        )
        service = E621MediaService(
            cast(E621Client, client), E621HistoryRepository(tmp_path)
        )
        post = _post(ext="gif")
        if failure == "conversion":
            client.download.side_effect = None
            client.download.return_value = _JPEG
            setattr(
                service.converter,
                "convert",
                AsyncMock(side_effect=E621Error("conversion")),
            )
        if failure == "no_preview":
            post = replace(post, sample_url=None)
            with pytest.raises(E621Error, match="превью"):
                await service.prepare(post, album=True)
            return
        prepared = await service.prepare(post, album=True)
        assert prepared.kind == "jpg" and "Превью" in prepared.notice
        await service.remember(prepared, _delivery())
        assert await service.history.cached_media(0, prepared.key) is None
        assert not service._ready

    asyncio.run(scenario())


def test_content_length_stops_oversized_download_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(200, b"large")
    setattr(response, "headers", {"Content-Length": "100"})
    cast(Mock, response.content).iter_chunked = Mock(
        side_effect=AssertionError("body should not be read")
    )
    client = E621Client("TestBot/1.0", request_interval=0)
    monkeypatch.setattr(client, "_session", lambda timeout: _Session(response))
    with pytest.raises(E621Error, match="лимит"):
        asyncio.run(client.download("https://example.org/file", max_bytes=10))


def test_missing_file_does_not_discard_other_prepared_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def prepare(
        self: E621MediaService, post: E621Post, **kwargs: object
    ) -> PreparedMedia:
        if post.post_id == 2:
            raise E621Error("no file")
        return PreparedMedia(post, str(post.post_id), "jpg", _JPEG)

    monkeypatch.setattr(E621MediaService, "prepare", prepare)
    history = E621HistoryRepository(tmp_path)
    router = handler.create_e621_router(
        cast(E621Client, _Client([_post(i) for i in range(1, 4)])),
        history,
        UserStateStore(),
    )
    message, raw = _message("/e6 dragon count:3")
    asyncio.run(_call_message(router, 0, message))
    assert asyncio.run(history.seen_ids(7)) == {1, 3}
    assert "Недоступных файлов: 1" in raw.answer.await_args.args[0]


def test_ram_cache_is_bounded_and_evicted_asset_downloads_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import protogen_delta.services.e621_media as media_module

    monkeypatch.setattr(media_module, "_RAM_BYTES", 2 * len(_JPEG) - 1)

    async def scenario() -> None:
        client = cast(Mock, _Client([]))
        client.download = AsyncMock(return_value=_JPEG)
        service = E621MediaService(
            cast(E621Client, client), E621HistoryRepository(tmp_path)
        )
        for post in (_post(1), _post(2), _post(1)):
            await service.prepare(post, album=True)
            assert sum(p.size for p in service._ready.values()) <= 2 * len(_JPEG) - 1
        assert client.download.await_count == 3

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "reason", ["message is not modified", "message to edit not found", "can't edit"]
)
def test_panel_edit_errors_are_handled_without_corrupting_history(
    tmp_path: Path, reason: str
) -> None:
    async def scenario() -> None:
        from aiogram.methods import EditMessageText

        router = handler.create_e621_router(
            cast(E621Client, _Client([_post()])),
            E621HistoryRepository(tmp_path),
            UserStateStore(),
        )
        message, raw = _message("/e6")
        await _call_message(router, 0, message)
        raw.panel.edit_text.side_effect = [
            TelegramBadRequest(EditMessageText(text="panel"), reason),
            None,
        ]
        cb = _button(raw, "images")
        if reason == "can't edit":
            with pytest.raises(TelegramBadRequest):
                await router.callback_query.handlers[1].callback(cb)
        else:
            await router.callback_query.handlers[1].callback(cb)
            assert raw.answer.await_count == (2 if "not found" in reason else 1)

    asyncio.run(scenario())


def test_old_panel_that_cannot_be_deleted_does_not_block_new_query(
    tmp_path: Path,
) -> None:
    from aiogram.methods import DeleteMessage

    async def scenario() -> None:
        history = E621HistoryRepository(tmp_path)
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post(1), _post(2)])), history, UserStateStore()
        )
        message, raw = _message()
        raw.panel.delete.side_effect = TelegramBadRequest(
            DeleteMessage(chat_id=7, message_id=1), "can't delete"
        )
        await _call_message(router, 0, message)
        await _call_message(router, 0, message)
        assert await history.seen_ids(7) == {1, 2}

    asyncio.run(scenario())


@pytest.mark.parametrize("fail_upload", [False, True])
def test_timeout_with_partial_upload_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_upload: bool
) -> None:
    async def scenario() -> None:
        async def prepare(
            self: E621MediaService, post: E621Post, **kwargs: object
        ) -> PreparedMedia:
            if post.post_id == 2 or not fail_upload:
                raise TimeoutError
            return PreparedMedia(post, "key", "jpg", _JPEG)

        monkeypatch.setattr(E621MediaService, "prepare", prepare)
        history = E621HistoryRepository(tmp_path)
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post(1), _post(2)])), history, UserStateStore()
        )
        message, raw = _message("/e6 dragon count:2")
        raw.answer_photo.side_effect = TelegramBadRequest(
            SendMediaGroup(chat_id=7, media=[]), "bad media"
        )
        await _call_message(router, 0, message)
        assert await history.seen_ids(7) == set()
        assert (
            "подготовленную часть" if fail_upload else "лимита времени"
        ) in raw.panel.edit_text.await_args.args[0]

    asyncio.run(scenario())


def test_simultaneous_requests_cannot_replace_session_before_lock(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        history = E621HistoryRepository(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        original = history.preferences

        async def preferences(user_id: int) -> E621Preferences:
            entered.set()
            await release.wait()
            return await original(user_id)

        setattr(history, "preferences", preferences)
        client = _Client([_post()])
        router = handler.create_e621_router(
            cast(E621Client, client), history, UserStateStore()
        )
        first, raw = _message()
        task = asyncio.create_task(_call_message(router, 0, first))
        await entered.wait()
        second, other = _message("/e6 wolf")
        await _call_message(router, 0, second)
        release.set()
        await asyncio.wait_for(task, 1)
        assert len(client.queries) == 1 and "dragon" in client.queries[0].tags
        assert "ещё готовится" in other.answer.await_args.args[0]
        raw.answer_photo.assert_awaited_once()

    asyncio.run(scenario())


def test_fifth_search_page_is_not_lost(tmp_path: Path) -> None:
    async def scenario() -> None:
        history = E621HistoryRepository(tmp_path)
        for i in range(1, 5):
            await history.mark_seen(7, i, i)
        client = cast(Mock, _Client([]))
        client.search = AsyncMock(side_effect=lambda query, *, page: [_post(page)])
        router = handler.create_e621_router(
            cast(E621Client, client), history, UserStateStore()
        )
        message, raw = _message()
        await _call_message(router, 0, message)
        assert "#5" in raw.answer_photo.await_args.kwargs["caption"]
        assert client.search.await_count == 5

    asyncio.run(scenario())
