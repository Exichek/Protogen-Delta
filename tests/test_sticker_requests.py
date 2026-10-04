"""Прямая просьба о стикере работает отдельно от редких фоновых реакций."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Message

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.stickers import StickerEntry, StickersRepository
from protogen_delta.services.response_engine import ResponseEngine
from protogen_delta.services.stickers import (
    ContextualStickerService,
    has_sticker_request,
)


@pytest.mark.parametrize(
    "text",
    ["а у тебя стикеры есть?", "Есть стикеры?", "Покажи стикер", "скинь стикер :D"],
)
def test_sticker_request_intent(text: str) -> None:
    assert has_sticker_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "не присылай стикеры",
        "не хочу стикеров",
        "Отключи стикеры",
        "Опиши этот стикер",
        "Как добавить стикер?",
        "Вот `покажи стикер` в коде",
        "https://example.org/покажи-стикер",
        "Стикеры смешные",
        "Покажи время",
    ],
)
def test_sticker_discussion_is_not_send_request(text: str) -> None:
    assert not has_sticker_request(text)


def _service(tmp_path: Path) -> tuple[ContextualStickerService, AsyncMock]:
    repo = StickersRepository(tmp_path)
    repo.upsert(StickerEntry("safe-one", "one", ("happy",)))
    repo.upsert(StickerEntry("safe-two", "two", ("happy",)))
    repo.upsert(StickerEntry("adult", "adult", ("rp",), rating="adult"))
    bot = AsyncMock(spec=Bot)
    return ContextualStickerService(cast(Bot, bot), repo, UserStateStore()), bot


def test_request_bypasses_gap_probability_but_has_short_cooldown(
    tmp_path: Path,
) -> None:
    service, bot = _service(tmp_path)
    now = [100.0]
    service._clock = lambda: now[0]
    service._random_value = lambda: 1.0
    service._choose = lambda entries: entries[0]

    async def scenario() -> None:
        assert await service.maybe_send(
            chat_id=7, user_id=42, context_text="а у тебя стикеры есть?"
        )
        assert not await service.maybe_send(
            chat_id=7, user_id=42, context_text="Покажи стикер"
        )
        now[0] += 11
        assert await service.maybe_send(
            chat_id=7, user_id=42, context_text="Покажи стикер"
        )
        assert not await service.maybe_send(
            chat_id=7, user_id=42, context_text="Спасибо :D"
        )

    asyncio.run(scenario())
    assert [call.kwargs["sticker"] for call in bot.send_sticker.await_args_list] == [
        "safe-one",
        "safe-two",
    ]


@pytest.mark.parametrize("empty", [False, True])
def test_disabled_or_empty_pack_reports_unavailability(
    tmp_path: Path, empty: bool
) -> None:
    service, bot = _service(tmp_path)
    if empty:
        for entry in service._repository.get_all():
            service._repository.remove(entry.file_unique_id)
    else:
        service._chance = 0
    assert "отключена или нет" in service.capabilities_context(42)
    assert not asyncio.run(
        service.maybe_send(chat_id=7, user_id=42, context_text="Покажи стикер")
    )
    bot.send_sticker.assert_not_awaited()


def test_text_handler_supplies_capability_and_delivers_requested_sticker(
    tmp_path: Path,
) -> None:
    service, bot = _service(tmp_path)
    engine = AsyncMock(spec=ResponseEngine)
    message = SimpleNamespace(
        text="а у тебя стикеры есть?",
        caption=None,
        sticker=None,
        chat=SimpleNamespace(id=7),
        from_user=SimpleNamespace(id=42),
        answer=AsyncMock(),
    )

    async def respond(user_id: int, text: str, deliver: Any, **kwargs: Any) -> None:
        assert "доступны 2" in kwargs["trusted_input_context"]
        assert "Не отрицай" in kwargs["trusted_input_context"]
        await deliver("Да, есть реакции из моего пака.")

    engine.respond_and_deliver.side_effect = respond
    router = create_text_router(cast(ResponseEngine, engine), sticker_service=service)
    asyncio.run(router.message.handlers[0].callback(cast(Message, message)))
    message.answer.assert_awaited_once()
    bot.send_sticker.assert_awaited_once()
