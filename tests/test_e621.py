"""Тесты поиска артов e621/e926 и истории показов."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Router
from aiogram.types import CallbackQuery, Message

import protogen_delta.handlers.e621 as handler_module
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.e621 import E621_USAGE, create_e621_router
from protogen_delta.repositories.e621_history import E621HistoryRepository
from protogen_delta.services.e621 import (
    E621_BASE_URL,
    E926_BASE_URL,
    E621Client,
    E621Error,
    E621Post,
    E621Query,
    E621QueryError,
    _parse_post,
    looks_like_e621_query,
    normalize_e621_query,
)
from protogen_delta.services.telegram_video import TelegramVideoConverter


def _post(post_id: int = 42, ext: str = "jpg") -> E621Post:
    return E621Post(
        post_id=post_id,
        rating="e",
        file_url=f"https://static.example/{post_id}.{ext}",
        file_ext=ext,
        file_size=100,
        sample_url=f"https://static.example/sample-{post_id}.jpg",
        preview_url=None,
        fav_count=12,
        score=7,
        artists=("artist_one",),
        tags=frozenset({"dragon"}),
        sources=("https://artist.example/work",),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dragon order:favcount", True),
        ("dragon rating:e", True),
        ("привет как дела", False),
        ("/e6 dragon", False),
        (None, False),
    ],
)
def test_looks_like_e621_query(text: str | None, expected: bool) -> None:
    assert looks_like_e621_query(text) is expected


def test_normalize_query_uses_safe_e926_until_adult_mode() -> None:
    query = normalize_e621_query("dragon rating:e order:favcount", "soft")
    assert query.base_url == E926_BASE_URL
    assert "rating:s" in query.tags
    assert "rating:e" not in query.tags
    assert "cub" not in query.tags


def test_normalize_query_keeps_adult_host_and_rating() -> None:
    query = normalize_e621_query("dragon rating:e", "adult")
    assert query.base_url == E621_BASE_URL
    assert "rating:e" in query.tags
    assert "dragon" in query.tags


@pytest.mark.parametrize(
    ("query", "match"),
    [
        ("", "хотя бы один"),
        ("young dragon", "несовершеннолетними"),
        ("dragon limit:100", "limit:"),
        ("dragon status:deleted", "Удалённые"),
        ("dragon rating:nope", "Допустимые рейтинги"),
        ("https://e621.net/posts/1", "не ссылка"),
        (" ".join(["tag"] * 41), "слишком длинный"),
    ],
)
def test_normalize_query_rejects_invalid_input(query: str, match: str) -> None:
    with pytest.raises(E621QueryError, match=match):
        normalize_e621_query(query, "adult")


def test_post_parsing_and_media_fallback() -> None:
    post = _parse_post(
        {
            "id": 9,
            "rating": "s",
            "file": {"url": "file", "ext": "png", "size": 30_000_000},
            "sample": {"url": "sample"},
            "preview": {"url": "preview"},
            "fav_count": 3,
            "score": {"total": 2},
            "tags": {"artist": ["someone"]},
            "sources": ["source"],
        }
    )
    assert post.media_url == "sample"
    assert post.page_url.endswith("/posts/9")
    assert post.artists == ("someone",)


def test_video_uses_original_above_old_image_limit() -> None:
    post = E621Post(
        post_id=10,
        rating="e",
        file_url="video",
        file_ext="webm",
        file_size=30_000_000,
        sample_url="sample",
        preview_url="preview",
        fav_count=1,
        score=1,
        artists=(),
        tags=frozenset(),
        sources=(),
    )

    assert post.media_url == "video"


def test_e621_client_validates_configuration() -> None:
    with pytest.raises(ValueError, match="USER_AGENT"):
        E621Client(" ")
    with pytest.raises(ValueError, match="отрицательным"):
        E621Client("agent", request_interval=-1)


class _Content:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def read(self, amount: int) -> bytes:
        return self.data[:amount]

    async def iter_chunked(self, amount: int) -> Any:
        del amount
        yield self.data


class _Response:
    def __init__(self, status: int, data: bytes) -> None:
        self.status = status
        self.content = _Content(data)

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.request: tuple[str, dict[str, Any]] | None = None

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.request = (url, kwargs)
        return self.response

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def test_client_search_parses_response_and_sends_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {
            "posts": [
                {
                    "id": 1,
                    "file": {"url": "file", "ext": "jpg", "size": 10},
                    "sample": {},
                    "preview": {},
                    "score": {"total": 4},
                    "tags": {"artist": []},
                }
            ]
        }
    ).encode()
    session = _Session(_Response(200, body))
    client = E621Client("TestBot/1.0 (owner)", request_interval=0)
    monkeypatch.setattr(client, "_session", lambda timeout: session)

    posts = asyncio.run(client.search(E621Query("dragon", E621_BASE_URL)))

    assert posts[0].post_id == 1
    assert session.request is not None
    assert session.request[1]["headers"]["User-Agent"] == "TestBot/1.0 (owner)"


@pytest.mark.parametrize("status", [429, 500])
def test_client_search_reports_http_errors(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    client = E621Client("agent", request_interval=0)
    monkeypatch.setattr(
        client, "_session", lambda timeout: _Session(_Response(status, b""))
    )
    with pytest.raises(E621Error):
        asyncio.run(client.search(E621Query("dragon", E621_BASE_URL)))


def test_client_search_rejects_invalid_json_and_filters_blocked_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = E621Client("agent", request_interval=0)
    monkeypatch.setattr(
        client, "_session", lambda timeout: _Session(_Response(200, b"not-json"))
    )
    with pytest.raises(E621Error, match="неожиданный"):
        asyncio.run(client.search(E621Query("dragon", E621_BASE_URL)))

    body = json.dumps(
        {
            "posts": [
                {
                    "id": 2,
                    "file": {},
                    "tags": {"general": ["dragon", "young"]},
                }
            ]
        }
    ).encode()
    monkeypatch.setattr(
        client, "_session", lambda timeout: _Session(_Response(200, body))
    )
    assert asyncio.run(client.search(E621Query("dragon", E621_BASE_URL))) == []


def test_client_download_returns_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = E621Client("agent", request_interval=0)
    monkeypatch.setattr(
        client, "_session", lambda timeout: _Session(_Response(200, b"abc"))
    )
    assert asyncio.run(client.download("https://static.example/a.jpg")) == b"abc"
    with pytest.raises(ValueError, match="больше нуля"):
        asyncio.run(client.download("https://static.example/a.jpg", max_bytes=0))


def test_client_download_reports_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = E621Client("agent", request_interval=0)
    monkeypatch.setattr(
        client, "_session", lambda timeout: _Session(_Response(404, b""))
    )
    with pytest.raises(E621Error, match="HTTP 404"):
        asyncio.run(client.download("https://static.example/missing.jpg"))


def test_history_persists_and_prunes_old_posts(tmp_path: Path) -> None:
    history = E621HistoryRepository(tmp_path, max_posts_per_user=2)
    asyncio.run(history.mark_seen(5, 10, 1.0))
    asyncio.run(history.mark_seen(5, 11, 2.0))
    asyncio.run(history.mark_seen(5, 12, 3.0))
    assert asyncio.run(history.seen_ids(5)) == {11, 12}
    assert asyncio.run(history.seen_ids(6)) == set()


def test_history_rejects_non_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="больше нуля"):
        E621HistoryRepository(tmp_path, max_posts_per_user=0)


class _Client:
    def __init__(self, posts: list[E621Post]) -> None:
        self.posts = posts
        self.queries: list[E621Query] = []

    async def search(
        self, query: E621Query, *, page: int = 1, limit: int = 20
    ) -> list[E621Post]:
        del page, limit
        self.queries.append(query)
        return self.posts

    async def download(self, url: str, *, max_bytes: int) -> bytes:
        del url, max_bytes
        return b"image"


class _ErrorClient(_Client):
    async def search(
        self, query: E621Query, *, page: int = 1, limit: int = 20
    ) -> list[E621Post]:
        del query, page, limit
        raise E621Error("Временная ошибка.")


async def _call_message(router: Router, index: int, message: Message) -> None:
    await router.message.handlers[index].callback(message)


def _message(text: str = "/e6 dragon", user_id: int = 7) -> tuple[Message, Mock]:
    message = Mock(spec=Message)
    message.text = text
    message.from_user = Mock(id=user_id)
    message.answer = AsyncMock()
    message.answer_photo = AsyncMock()
    message.answer_animation = AsyncMock()
    message.answer_video = AsyncMock()
    message.answer_document = AsyncMock()
    return cast(Message, message), message


def test_command_without_tags_shows_usage(tmp_path: Path) -> None:
    message, raw = _message("/e6")
    router = create_e621_router(
        cast(E621Client, _Client([])), E621HistoryRepository(tmp_path), UserStateStore()
    )
    asyncio.run(_call_message(router, 0, message))
    raw.answer.assert_awaited_once_with(E621_USAGE)


def test_command_sends_post_and_remembers_it(tmp_path: Path) -> None:
    message, raw = _message()
    history = E621HistoryRepository(tmp_path)
    client = _Client([_post()])
    states = UserStateStore()
    states.get(7).content_mode = "adult"
    router = create_e621_router(cast(E621Client, client), history, states)

    asyncio.run(_call_message(router, 0, message))

    raw.answer_photo.assert_awaited_once()
    assert asyncio.run(history.seen_ids(7)) == {42}
    assert client.queries[0].base_url == E621_BASE_URL
    markup = raw.answer_photo.await_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].text == "🔄 Ещё"


def test_direct_query_uses_safe_mode(tmp_path: Path) -> None:
    message, raw = _message("dragon order:favcount")
    client = _Client([_post()])
    router = create_e621_router(
        cast(E621Client, client), E621HistoryRepository(tmp_path), UserStateStore()
    )
    asyncio.run(_call_message(router, 1, message))
    raw.answer_photo.assert_awaited_once()
    assert client.queries[0].base_url == E926_BASE_URL


def test_animation_and_video_use_matching_telegram_methods(
    tmp_path: Path,
) -> None:
    for post_id, ext, method in [
        (1, "gif", "answer_animation"),
        (2, "mp4", "answer_video"),
        (3, "webm", "answer_document"),
    ]:
        message, raw = _message(user_id=post_id)
        router = create_e621_router(
            cast(E621Client, _Client([_post(post_id, ext)])),
            E621HistoryRepository(tmp_path / str(post_id)),
            UserStateStore(),
        )
        asyncio.run(_call_message(router, 0, message))
        getattr(raw, method).assert_awaited_once()


def test_query_error_is_shown_to_user(tmp_path: Path) -> None:
    message, raw = _message("/e6 young")
    router = create_e621_router(
        cast(E621Client, _Client([])), E621HistoryRepository(tmp_path), UserStateStore()
    )
    asyncio.run(_call_message(router, 0, message))
    assert "несовершеннолетними" in raw.answer.await_args.args[0]


def test_search_error_and_empty_result_are_shown(tmp_path: Path) -> None:
    for client, expected in [
        (_ErrorClient([]), "Временная"),
        (_Client([]), "не нашлось"),
    ]:
        message, raw = _message()
        router = create_e621_router(
            cast(E621Client, client),
            E621HistoryRepository(tmp_path / expected),
            UserStateStore(),
        )
        asyncio.run(_call_message(router, 0, message))
        assert expected in raw.answer.await_args.args[0]


def _callback(data: str, user_id: int, message: Message | None = None) -> CallbackQuery:
    callback = Mock(spec=CallbackQuery)
    callback.data = data
    callback.from_user = Mock(id=user_id)
    callback.message = message
    callback.answer = AsyncMock()
    return cast(CallbackQuery, callback)


def test_next_button_continues_search_and_rejects_other_users(tmp_path: Path) -> None:
    message, raw = _message()
    router = create_e621_router(
        cast(E621Client, _Client([_post()])),
        E621HistoryRepository(tmp_path),
        UserStateStore(),
    )
    asyncio.run(_call_message(router, 0, message))

    foreign = _callback("e6:next:7", 8, message)
    asyncio.run(router.callback_query.handlers[0].callback(foreign))
    cast(AsyncMock, foreign.answer).assert_awaited_once_with(
        "Эта кнопка предназначена не тебе.", show_alert=True
    )

    owner = _callback("e6:next:7", 7, message)
    asyncio.run(router.callback_query.handlers[0].callback(owner))
    cast(AsyncMock, owner.answer).assert_awaited_once_with()
    assert "не нашлось" in raw.answer.await_args.args[0]


def test_stale_or_malformed_next_button_shows_alert(tmp_path: Path) -> None:
    router = create_e621_router(
        cast(E621Client, _Client([])), E621HistoryRepository(tmp_path), UserStateStore()
    )
    callback = _callback("e6:next:broken", 0)
    asyncio.run(router.callback_query.handlers[0].callback(callback))
    cast(AsyncMock, callback.answer).assert_awaited_once_with(
        "Поиск устарел. Запусти /e6 ещё раз.", show_alert=True
    )


def test_message_without_sender_is_ignored(tmp_path: Path) -> None:
    message, raw = _message()
    raw.from_user = None
    router = create_e621_router(
        cast(E621Client, _Client([])), E621HistoryRepository(tmp_path), UserStateStore()
    )
    asyncio.run(_call_message(router, 0, message))
    raw.answer.assert_not_awaited()


def test_caption_handles_missing_optional_metadata() -> None:
    post = E621Post(1, "x", None, "jpg", 0, None, None, 0, 0, (), frozenset(), ())
    assert post.media_url is None
    assert "не указан" in handler_module._caption(post)


@pytest.mark.parametrize(
    "raw", ["dragon order:favcount count:3", "dragon order:favcount 3"]
)
def test_batch_keeps_sort_order_and_remembers_each_post(
    tmp_path: Path, raw: str
) -> None:
    client = _Client([_post(1), _post(2), _post(3), _post(4)])
    history = E621HistoryRepository(tmp_path)
    states = UserStateStore()
    states.get(7).content_mode = "adult"
    router = create_e621_router(cast(E621Client, client), history, states)
    message, mock = _message("/e6 " + raw)
    asyncio.run(_call_message(router, 0, message))
    assert mock.answer_photo.await_count == 3
    assert [
        call.kwargs["caption"].split(" ·")[0]
        for call in mock.answer_photo.await_args_list
    ] == ["e621 #1", "e621 #2", "e621 #3"]
    assert "count:" not in client.queries[0].tags
    assert asyncio.run(history.seen_ids(7)) == {1, 2, 3}


@pytest.mark.parametrize(
    "raw", ["dragon count:0", "dragon count:11", "dragon count:2 count:3"]
)
def test_batch_rejects_invalid_count(raw: str) -> None:
    with pytest.raises(E621QueryError, match="Количество"):
        handler_module._split_count(raw)


def test_webm_is_converted_and_sent_as_streaming_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    convert = AsyncMock(return_value=b"mp4")
    monkeypatch.setattr(TelegramVideoConverter, "convert", convert)
    client = _Client([_post(1, "webm")])
    router = create_e621_router(
        cast(E621Client, client), E621HistoryRepository(tmp_path), UserStateStore()
    )
    message, mock = _message()
    asyncio.run(_call_message(router, 0, message))
    assert mock.answer_video.await_args.kwargs["supports_streaming"] is True
    assert mock.answer_video.await_args.args[0].filename.endswith(".mp4")
    mock.answer_document.assert_not_awaited()


def test_oversized_video_prefers_mp4_alternate_over_preview(tmp_path: Path) -> None:
    post = replace(
        _post(1, "webm"),
        file_size=100_000_000,
        mp4_urls=("https://static.example/720.mp4",),
    )
    client = _Client([post])
    router = create_e621_router(
        cast(E621Client, client), E621HistoryRepository(tmp_path), UserStateStore()
    )
    message, mock = _message()
    asyncio.run(_call_message(router, 0, message))
    mock.answer_video.assert_awaited_once()
    mock.answer_photo.assert_not_awaited()


def test_parse_mp4_alternates() -> None:
    post = _parse_post(
        {
            "id": 1,
            "sample": {
                "alternates": {
                    "720p": {
                        "urls": [
                            "https://static.example/a.webm",
                            "https://static.example/a.mp4",
                        ]
                    }
                }
            },
        }
    )
    assert post.mp4_urls == ("https://static.example/a.mp4",)
    assert post.webm_urls == ("https://static.example/a.webm",)


def test_oversized_original_uses_convertible_webm_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = replace(
        _post(1, "webm"),
        file_size=100_000_000,
        webm_urls=("https://static.example/720.webm",),
    )
    convert = AsyncMock(return_value=b"mp4")
    monkeypatch.setattr(TelegramVideoConverter, "convert", convert)
    router = create_e621_router(
        cast(E621Client, _Client([post])),
        E621HistoryRepository(tmp_path),
        UserStateStore(),
    )
    message, mock = _message()
    asyncio.run(_call_message(router, 0, message))
    mock.answer_video.assert_awaited_once()
    mock.answer_photo.assert_not_awaited()


def test_oversized_video_fallback_is_labeled_as_preview(tmp_path: Path) -> None:
    post = replace(_post(1, "webm"), file_size=100_000_000)
    client = _Client([post])
    router = create_e621_router(
        cast(E621Client, client), E621HistoryRepository(tmp_path), UserStateStore()
    )
    message, mock = _message()
    asyncio.run(_call_message(router, 0, message))
    assert "отправлено превью" in mock.answer_photo.await_args.kwargs["caption"]


def test_unavailable_mp4_alternate_falls_back_to_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    post = replace(_post(1, "webm"), mp4_urls=("https://static.example/a.mp4",))
    client = _Client([post])
    monkeypatch.setattr(
        client,
        "download",
        AsyncMock(side_effect=[E621Error("unavailable"), b"invalid webm"]),
    )
    router = create_e621_router(
        cast(E621Client, client), E621HistoryRepository(tmp_path), UserStateStore()
    )
    message, mock = _message()
    asyncio.run(_call_message(router, 0, message))
    mock.answer_document.assert_awaited_once()
    assert "исходный WebM" in mock.answer_document.await_args.kwargs["caption"]


def test_old_next_button_rechecks_changed_age_mode(tmp_path: Path) -> None:
    states = UserStateStore()
    states.get(7).content_mode = "adult"
    client = _Client([_post(1), _post(2)])
    router = create_e621_router(
        cast(E621Client, client), E621HistoryRepository(tmp_path), states
    )
    message, _ = _message()
    asyncio.run(_call_message(router, 0, message))
    states.get(7).content_mode = "soft"
    callback = _callback("e6:next:7", 7, message)
    asyncio.run(router.callback_query.handlers[0].callback(callback))
    assert client.queries[-1].base_url == E926_BASE_URL
