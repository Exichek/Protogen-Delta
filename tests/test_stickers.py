"""Тесты хранения, разметки и контекстного выбора стикеров."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.stickers import create_sticker_admin_router
from protogen_delta.repositories.stickers import (
    StickerEntry,
    StickerRating,
    StickersRepository,
)
from protogen_delta.services.sticker_pack import StickerPackImporter
from protogen_delta.services.stickers import (
    ContextualStickerService,
    has_sticker_request,
    sticker_context_tags,
)


@pytest.mark.parametrize(
    "text",
    [
        "Не отправляй стикеры :D",
        "Не скидывай стикеры, спасибо :) ",
        "Давай без стикеров 😂",
        "Стикеры не нужны, привет",
    ],
)
def test_sticker_opt_out_wins_over_smile_or_greeting(tmp_path: Path, text: str) -> None:
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("greeting", "greeting", "happy", "laugh"))
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot), repository, UserStateStore(), min_replies=1, chance=1
    )
    assert not has_sticker_request(text)
    assert not asyncio.run(service.maybe_send(chat_id=1, user_id=1, context_text=text))
    bot.send_sticker.assert_not_awaited()


def test_quoted_sticker_request_and_emoji_are_not_authors_reaction() -> None:
    text = 'Он написал «пришли стикер :D» и "привет 😊", а я спрашиваю про код.'
    assert not has_sticker_request(text)
    assert not sticker_context_tags(text)
    assert "laugh" in sticker_context_tags("Он написал «привет», ахаха :D")


def _entry(
    unique_id: str,
    *tags: str,
    rating: str = "safe",
) -> StickerEntry:
    return StickerEntry(
        file_id=f"file-{unique_id}",
        file_unique_id=unique_id,
        tags=tuple(tags),
        rating=cast(StickerRating, rating),
        emoji="🙂",
        set_name="delta_pack",
    )


def test_sticker_repository_adds_updates_and_removes(tmp_path: Path) -> None:
    """Разметка должна переживать новый экземпляр репозитория."""
    repository = StickersRepository(tmp_path)
    assert repository.get_all() == []
    assert repository.upsert(_entry("one", "playful")) is True
    assert repository.upsert(_entry("one", "angry", rating="adult")) is False

    restored = StickersRepository(tmp_path)
    assert restored.get_all() == [_entry("one", "angry", rating="adult")]
    assert restored.remove("missing") is False
    assert restored.remove("one") is True
    assert restored.get_all() == []


def test_sticker_repository_rejects_invalid_json(tmp_path: Path) -> None:
    """Повреждённая структура не должна тихо становиться пустым паком."""
    (tmp_path / "stickers.json").write_text(
        '{"STICKERS":[{"file_id":"x"}]}', encoding="utf-8"
    )
    with pytest.raises(TypeError, match="STICKERS"):
        StickersRepository(tmp_path).get_all()


def test_contextual_sticker_obeys_gap_cooldown_and_no_repeat(tmp_path: Path) -> None:
    """Стикеры должны быть редкими и не повторяться подряд при наличии выбора."""
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("one", "playful"))
    repository.upsert(_entry("two", "playful"))
    states = UserStateStore()
    states.get(42).mood = "playful"
    bot = AsyncMock(spec=Bot)
    now = [100.0]
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=1,
        cooldown_seconds=60,
        min_replies=2,
        clock=lambda: now[0],
        random_value=lambda: 0,
        choose=lambda entries: entries[0],
    )

    async def scenario() -> None:
        assert await service.maybe_send(chat_id=7, user_id=42) is False
        assert await service.maybe_send(chat_id=7, user_id=42) is True
        assert await service.maybe_send(chat_id=7, user_id=42) is False
        assert await service.maybe_send(chat_id=7, user_id=42) is False
        now[0] += 61
        assert await service.maybe_send(chat_id=7, user_id=42) is True

    asyncio.run(scenario())

    assert bot.send_sticker.await_args_list[0].kwargs["sticker"] == "file-one"
    assert bot.send_sticker.await_args_list[1].kwargs["sticker"] == "file-two"


def test_contextual_sticker_filters_adult_rating(tmp_path: Path) -> None:
    """Adult-стикер разрешён только после выбора взрослого режима."""
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("adult", "horny", rating="adult"))
    states = UserStateStore()
    state = states.get(42)
    state.mood = "horny"
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=1,
        cooldown_seconds=0,
        min_replies=1,
        random_value=lambda: 0,
    )

    async def scenario() -> None:
        state.content_mode = "soft"
        assert await service.maybe_send(chat_id=7, user_id=42) is False
        state.content_mode = "adult"
        assert await service.maybe_send(chat_id=7, user_id=42) is True

    asyncio.run(scenario())
    bot.send_sticker.assert_awaited_once_with(chat_id=7, sticker="file-adult")


def test_contextual_sticker_uses_rp_tag_and_probability_gate(tmp_path: Path) -> None:
    """RP может выбрать свой тег, а нулевая вероятность отключает реакции."""
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("rp", "rp"))
    states = UserStateStore()
    states.get(42).roleplay_active = True
    bot = AsyncMock(spec=Bot)
    disabled = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=0,
        cooldown_seconds=0,
        min_replies=1,
        random_value=lambda: 0,
    )
    enabled = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=1,
        cooldown_seconds=0,
        min_replies=1,
        random_value=lambda: 0,
    )

    async def scenario() -> None:
        assert await disabled.maybe_send(chat_id=7, user_id=42) is False
        assert await enabled.maybe_send(chat_id=7, user_id=42) is True

    asyncio.run(scenario())


def test_contextual_sticker_ignores_send_error(tmp_path: Path) -> None:
    """Ошибка Telegram при дополнительной реакции не ломает текстовый ответ."""
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("one", "neutral"))
    bot = AsyncMock(spec=Bot)
    bot.send_sticker.side_effect = TelegramBadRequest(
        method=Mock(), message="sticker invalid"
    )
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        UserStateStore(),
        chance=1,
        cooldown_seconds=0,
        min_replies=1,
        random_value=lambda: 0,
    )

    result = asyncio.run(service.maybe_send(chat_id=7, user_id=42))

    assert result is False


def test_contextual_sticker_validates_configuration(tmp_path: Path) -> None:
    """Невозможные вероятности и интервалы должны отклоняться при запуске."""
    bot = cast(Bot, AsyncMock(spec=Bot))
    repository = StickersRepository(tmp_path)
    states = UserStateStore()
    with pytest.raises(ValueError, match="chance"):
        ContextualStickerService(bot, repository, states, chance=2)
    with pytest.raises(ValueError, match="cooldown"):
        ContextualStickerService(bot, repository, states, cooldown_seconds=-1)
    with pytest.raises(ValueError, match="min_replies"):
        ContextualStickerService(bot, repository, states, min_replies=0)


def _admin_message(
    text: str,
    *,
    user_id: int = 1,
    sticker: object | None = None,
) -> tuple[Message, AsyncMock, AsyncMock]:
    message = Mock(spec=Message)
    message.text = text
    message.from_user = SimpleNamespace(id=user_id)
    message.answer = AsyncMock()
    message.answer_sticker = AsyncMock()
    message.reply_to_message = (
        SimpleNamespace(sticker=sticker) if sticker is not None else None
    )
    return cast(Message, message), message.answer, message.answer_sticker


async def _call(router: Router, message: Message) -> None:
    await router.message.handlers[0].callback(message)


def _answer_text(answer: AsyncMock) -> str:
    call = answer.await_args
    assert call is not None
    return cast(str, call.args[0])


def test_admin_can_tag_list_test_and_remove_sticker(tmp_path: Path) -> None:
    """Полный цикл разметки должен работать reply-командами."""
    repository = StickersRepository(tmp_path)
    router = create_sticker_admin_router(repository, frozenset({1}))
    sticker = SimpleNamespace(
        file_id="file-one",
        file_unique_id="one",
        emoji="😼",
        set_name="delta_pack",
    )

    async def scenario() -> None:
        add, answer, _ = _admin_message(
            "/stickers add playful rp rating:adult", sticker=sticker
        )
        await _call(router, add)
        assert "Добавил" in _answer_text(answer)

        listing, list_answer, _ = _admin_message("/stickers list")
        await _call(router, listing)
        assert "adult: 1" in _answer_text(list_answer)
        assert "playful: 1" in _answer_text(list_answer)

        testing, _, answer_sticker = _admin_message("/stickers test playful")
        await _call(router, testing)
        answer_sticker.assert_awaited_once_with("file-one")

        remove, remove_answer, _ = _admin_message("/stickers remove", sticker=sticker)
        await _call(router, remove)
        assert "удалён" in _answer_text(remove_answer)

    asyncio.run(scenario())
    assert repository.get_all() == []


def test_sticker_admin_rejects_non_admin_and_bad_add(tmp_path: Path) -> None:
    """Разметка закрыта от пользователей и проверяет теги."""
    router = create_sticker_admin_router(StickersRepository(tmp_path), frozenset({1}))

    async def scenario() -> None:
        denied, denied_answer, _ = _admin_message("/stickers list", user_id=2)
        await _call(router, denied)
        assert "нет доступа" in _answer_text(denied_answer)

        invalid, invalid_answer, _ = _admin_message("/stickers add плохой тег")
        await _call(router, invalid)
        assert "Ответь" in _answer_text(invalid_answer)

    asyncio.run(scenario())


def test_pack_import_matches_ids_preserves_tags_and_skips_unknown(
    tmp_path: Path,
) -> None:
    """Перестановка и добавление новых стикеров не должны менять их смысл."""
    repository = StickersRepository(tmp_path)
    known_id = "AgAD6rAAAhoQCUo"
    repository.upsert(_entry(known_id, "custom", rating="adult"))
    bot = AsyncMock(spec=Bot)
    bot.get_sticker_set.return_value = SimpleNamespace(
        name="delta_sticksss",
        stickers=[
            SimpleNamespace(file_unique_id="unknown", file_id="new", emoji="🙂"),
            SimpleNamespace(file_unique_id=known_id, file_id="fresh", emoji="😴"),
            SimpleNamespace(
                file_unique_id="AgAD_qUAAi5dCUo", file_id="greeting", emoji="👋"
            ),
        ],
    )
    importer = StickerPackImporter(cast(Bot, bot), repository)
    first = asyncio.run(importer.sync())
    assert (first.matched, first.added, first.unknown) == (2, 1, 1)
    second = asyncio.run(importer.sync())
    assert second.added == 0
    entries = StickersRepository(tmp_path).get_all()
    assert len(entries) == 2
    assert entries[0].file_id == "fresh"
    assert entries[0].tags == ("custom",)
    assert entries[0].rating == "adult"
    assert entries[0].name == "boring"
    assert entries[1].tags == ("greeting",)
    assert entries[1].name == "hi_sticker"
    assert repository.remove(known_id) is True
    asyncio.run(importer.sync())
    assert len(repository.get_all()) == 1
    repository.upsert(_entry(known_id, "custom"))
    asyncio.run(importer.sync())
    assert len(repository.get_all()) == 2


def test_sticker_prefers_specific_context_to_generic_mood(tmp_path: Path) -> None:
    """Запрос про кофе должен выбрать кофе, даже при игривом настроении."""
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("happy", "playful"))
    repository.upsert(_entry("coffee", "coffee"))
    states = UserStateStore()
    states.get(42).mood = "playful"
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=1,
        min_replies=1,
        cooldown_seconds=0,
        random_value=lambda: 0,
        choose=lambda items: items[0],
    )
    assert (
        asyncio.run(
            service.maybe_send(chat_id=7, user_id=42, context_text="Нужен кофе")
        )
        is True
    )
    bot.send_sticker.assert_awaited_once_with(chat_id=7, sticker="file-coffee")


def test_active_rp_alone_does_not_send_adult_sticker(tmp_path: Path) -> None:
    """Даже взрослому пользователю RP само по себе не повод для adult-реакции."""
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("adult", "rp", rating="adult"))
    states = UserStateStore()
    states.get(42).content_mode = "adult"
    states.get(42).roleplay_active = True
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=1,
        min_replies=1,
        cooldown_seconds=0,
        random_value=lambda: 0,
    )
    assert asyncio.run(service.maybe_send(chat_id=7, user_id=42)) is False
    bot.send_sticker.assert_not_awaited()


def test_admin_pack_import_reports_results(tmp_path: Path) -> None:
    """Импорт доступен только администратору, ошибки Telegram понятны."""
    importer = AsyncMock(spec=StickerPackImporter)
    importer.sync.return_value = SimpleNamespace(matched=19, added=19, unknown=0)
    router = create_sticker_admin_router(
        StickersRepository(tmp_path),
        frozenset({1}),
        cast(StickerPackImporter, importer),
    )

    async def scenario() -> None:
        denied, _, _ = _admin_message("/stickers import", user_id=2)
        await _call(router, denied)
        importer.sync.assert_not_awaited()
        message, answer, _ = _admin_message("/stickers import")
        await _call(router, message)
        assert "19" in _answer_text(answer)
        importer.sync.side_effect = TelegramBadRequest(method=Mock(), message="missing")
        failed, failure_answer, _ = _admin_message("/stickers import")
        await _call(router, failed)
        assert "позже" in _answer_text(failure_answer)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "restriction",
    [
        "age_restricted",
        "delta_reference_restricted",
        "roleplay_character_restricted",
        "character",
    ],
)
def test_reference_age_marker_also_filters_adult_stickers(
    tmp_path: Path, restriction: str
) -> None:
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("restricted-reaction", "playful", rating="adult"))
    states = UserStateStore()
    state = states.get(42)
    state.content_mode = "adult"
    if restriction == "character":
        state.roleplay_character = "Ребёнок"
    else:
        setattr(state, restriction, True)
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot), repository, states, min_replies=1, chance=1
    )
    assert not service.is_available(42)
    assert "нет доступных" in service.capabilities_context(42)
    assert not asyncio.run(
        service.maybe_send(chat_id=42, user_id=42, context_text="uwu")
    )
    bot.send_sticker.assert_not_awaited()
    repository.upsert(_entry("ordinary-reaction", "playful"))
    assert service.is_available(42)
    assert asyncio.run(service.maybe_send(chat_id=42, user_id=42, context_text="uwu"))
    assert bot.send_sticker.await_args is not None
    assert bot.send_sticker.await_args.kwargs["sticker"] == "file-ordinary-reaction"
