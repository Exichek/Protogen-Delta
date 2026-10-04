"""Проверки размеров фотографий Telegram и календарного Popular."""

import asyncio
import sqlite3
from dataclasses import replace
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image
from test_e621 import _JPEG, _call_message, _Client, _message, _post
from test_e621_albums import _button

import protogen_delta.handlers.e621 as handler
import protogen_delta.services.telegram_photo as photo
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.e621_history import (
    E621HistoryRepository,
    E621Preferences,
    PopularPeriod,
)
from protogen_delta.services.e621 import E621Client, E621Error
from protogen_delta.services.e621_media import E621MediaService


def _bytes(image: Image.Image, kind: str = "PNG") -> bytes:
    stream = BytesIO()
    image.save(stream, kind)
    return stream.getvalue()


@pytest.mark.parametrize("size", [(1536, 13001), (10, 10000), (10000, 10), (600, 400)])
def test_photo_fits_telegram_without_cropping(size: tuple[int, int]) -> None:
    data = photo.prepare_photo(_bytes(Image.new("RGB", size, "red")))
    with Image.open(BytesIO(data)) as result:
        assert result.format == "JPEG" and result.mode == "RGB"
        assert sum(result.size) <= 10000
        assert max(result.size) / min(result.size) <= 20
        assert len(data) <= 9 * 1024**2
        pixel = result.getpixel((result.width // 2, result.height // 2))
        assert isinstance(pixel, tuple) and pixel[0] > 240 and pixel[1] < 10


def test_photo_preserves_ordinary_jpeg_and_flattens_transparency() -> None:
    assert photo.prepare_photo(_JPEG) == _JPEG
    data = photo.prepare_photo(_bytes(Image.new("RGBA", (60, 40)), "WEBP"))
    with Image.open(BytesIO(data)) as result:
        assert result.getpixel((30, 20)) == (255, 255, 255)


def test_photo_applies_exif_orientation() -> None:
    image = Image.new("RGB", (60, 40))
    exif = image.getexif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, "JPEG", exif=exif)
    with Image.open(BytesIO(photo.prepare_photo(stream.getvalue()))) as result:
        assert result.size == (40, 60)
        assert not result.getexif().get(274)


@pytest.mark.parametrize("case", ["corrupt", "gif", "pixels", "bytes"])
def test_photo_rejects_invalid_or_unbounded_input(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _JPEG
    if case == "corrupt":
        data = b"not an image"
    elif case == "gif":
        data = _bytes(Image.new("RGB", (20, 20)), "GIF")
    elif case == "pixels":
        monkeypatch.setattr(photo, "_MAX_PIXELS", 1)
    else:
        monkeypatch.setattr(photo, "_MAX_BYTES", 1)
    with pytest.raises(E621Error):
        photo.prepare_photo(data)


def test_photo_compresses_large_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(photo, "_MAX_BYTES", 30000)
    image = Image.effect_noise((200, 200), 100).convert("RGB")
    assert len(photo.prepare_photo(_bytes(image))) <= 30000


def test_corrupt_original_uses_valid_sample(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = cast(Mock, _Client([]))
        client.download = AsyncMock(side_effect=[b"corrupt", _JPEG])
        service = E621MediaService(
            cast(E621Client, client), E621HistoryRepository(tmp_path)
        )
        result = await service.prepare(_post(), album=True)
        assert result.kind == "jpg" and result.notice and result.content == _JPEG
        assert client.download.await_args.args[0] == _post().sample_url

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "period,anchor,start,end",
    [
        ("day", date(2026, 10, 4), date(2026, 10, 4), date(2026, 10, 4)),
        ("week", date(2026, 10, 4), date(2026, 9, 28), date(2026, 10, 4)),
        ("month", date(2024, 2, 17), date(2024, 2, 1), date(2024, 2, 29)),
        ("week", date(2026, 1, 1), date(2025, 12, 29), date(2026, 1, 4)),
    ],
)
def test_popular_calendar_and_filters(
    period: PopularPeriod, anchor: date, start: date, end: date
) -> None:
    assert handler._popular_dates(period, anchor) == (start, end)
    query = handler._query(
        "dragon order:favcount date:2020-01-01",
        E621Preferences("images", "random", 3, period),
        "soft",
        anchor,
    )
    assert f"date:>={start}" in query.tags
    assert f"date:<{end + timedelta(days=1)}" in query.tags
    assert "order:score" in query.tags and "rating:s" in query.tags
    assert "-type:webm" in query.tags
    assert "favcount" not in query.tags and "2020" not in query.tags


@pytest.mark.parametrize(
    "period,anchor,delta,expected",
    [
        ("month", date(2026, 1, 31), -1, date(2025, 12, 1)),
        ("month", date(2026, 12, 31), 1, date(2027, 1, 1)),
        ("week", date(2026, 10, 4), -1, date(2026, 9, 27)),
        ("day", date(2026, 10, 4), 1, date(2026, 10, 5)),
    ],
)
def test_popular_navigation(
    period: PopularPeriod, anchor: date, delta: int, expected: date
) -> None:
    assert handler._shift_period(period, anchor, delta) == expected


def test_reopening_panel_posts_below_new_command_and_expires_old_buttons(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post()])),
            E621HistoryRepository(tmp_path),
            UserStateStore(),
        )
        first, raw = _message("/e6 dragon count:4")
        await _call_message(router, 0, first)
        old_button = _button(raw, "next")
        old_edits = raw.panel.edit_text.await_count
        new, newer = _message("/e6")
        await _call_message(router, 0, new)
        assert newer.answer.await_count == 1
        assert "До 4" in newer.answer.await_args.args[0]
        assert raw.panel.edit_text.await_count == old_edits
        await router.callback_query.handlers[1].callback(old_button)
        cast(Mock, old_button.answer).assert_awaited_once_with(
            "Панель устарела. Открой /e6 заново.", show_alert=True
        )

    asyncio.run(scenario())


def test_popular_menu_browses_without_tags_and_preserves_preferences(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        history = E621HistoryRepository(tmp_path)
        client = cast(Mock, _Client([]))
        client.search = AsyncMock()
        client.search.side_effect = lambda *a, **kw: [_post(client.search.await_count)]
        router = handler.create_e621_router(
            cast(E621Client, client), history, UserStateStore()
        )
        message, raw = _message("/e6")
        await _call_message(router, 0, message)
        for action in (
            "popular",
            "day",
            "week",
            "month",
            "prevperiod",
            "nextperiod",
            "nowperiod",
            "next",
        ):
            await router.callback_query.handlers[1].callback(_button(raw, action))
        assert client.search.await_count >= 7
        assert all(
            "order:score" in call.args[0].tags for call in client.search.await_args_list
        )
        assert (await history.preferences(7)).period == "month"
        assert (await history.preferences(8)).period == "all"
        await router.callback_query.handlers[1].callback(_button(raw, "favcount"))
        assert (await history.preferences(7)).period == "all"
        await history.save_preferences(7, E621Preferences(period="day"))
        explicit, _ = _message("/e6 dragon order:favcount count:4")
        await _call_message(router, 0, explicit)
        assert "date:" not in client.search.await_args.args[0].tags
        assert "order:favcount" in client.search.await_args.args[0].tags

    asyncio.run(scenario())


def test_old_preferences_schema_migrates_without_reset(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "e621.db") as connection:
        connection.execute(
            "CREATE TABLE search_preferences (user_id INTEGER PRIMARY KEY, "
            "media_filter TEXT NOT NULL, search_order TEXT NOT NULL, result_count INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO search_preferences VALUES (7, 'images', 'random', 5)"
        )

    async def scenario() -> None:
        repo = E621HistoryRepository(tmp_path)
        assert await repo.preferences(7) == E621Preferences("images", "random", 5)
        await repo.save_preferences(7, E621Preferences("images", "random", 5, "week"))
        fresh = E621HistoryRepository(tmp_path)
        assert (await fresh.preferences(7)).period == "week"
        with pytest.raises(ValueError):
            await fresh.save_preferences(
                7, replace(E621Preferences(), period=cast(PopularPeriod, "bad"))
            )

    asyncio.run(scenario())


def test_bare_command_during_preparation_keeps_active_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()

        async def waiting(*args: object, **kwargs: object) -> None:
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(E621MediaService, "prepare", waiting)
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post()])),
            E621HistoryRepository(tmp_path),
            UserStateStore(),
        )
        message, raw = _message("/e6 dragon")
        task = asyncio.create_task(_call_message(router, 0, message))
        await asyncio.wait_for(entered.wait(), 2)
        new, newer = _message("/e6")
        await _call_message(router, 0, new)
        assert "ещё готовится" in newer.answer.await_args.args[0]
        raw.panel.delete.assert_not_awaited()
        await router.callback_query.handlers[1].callback(_button(raw, "stop"))
        await asyncio.wait_for(task, 2)
        assert "остановлен" in raw.panel.edit_text.await_args.args[0]

    asyncio.run(scenario())
