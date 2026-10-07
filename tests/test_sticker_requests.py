"""Прямая просьба о стикере работает отдельно от редких фоновых реакций."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.handlers.start import create_start_router
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.stickers import StickerEntry, StickersRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.deepseek import DeepSeekService
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


STICKER_QUESTION = "Как работают твои стикеры говоришь? :D"
STICKER_EXPLANATION = (
    "У меня есть набор из 19 Telegram-стикеров. "
    "Если хочешь посмотреть — скажи, я попробую показать один."
)


@pytest.mark.parametrize(
    "text",
    [
        "скинь какой-то",
        "покажи какой-нибудь",
        "давай пример",
        "ещё один",
        "скинь",
        "Ну пришли один пожалуйста :D",
        "Дай мне любой 🙂",
    ],
)
def test_followup_request_requires_immediate_sticker_context(text: str) -> None:
    assert not has_sticker_request(text)
    assert has_sticker_request(
        text, previous_user_message=STICKER_QUESTION, previous_reply=STICKER_EXPLANATION
    )
    assert not has_sticker_request(text, previous_reply=STICKER_EXPLANATION)
    assert not has_sticker_request(text, previous_user_message=STICKER_QUESTION)


@pytest.mark.parametrize(
    "text",
    [
        "не надо",
        "не скидывай",
        "давай пример кода",
        "скинь музыку",
        "покажи время",
        "давай без стикеров",
        "а как это работает?",
        "Вот `скинь один`",
        "«скинь какой-то»",
        "скинь https://example.org/video.mp4",
        "скинь " + " " * 150,
    ],
)
def test_followup_does_not_turn_other_requests_into_stickers(text: str) -> None:
    assert not has_sticker_request(
        text, previous_user_message=STICKER_QUESTION, previous_reply=STICKER_EXPLANATION
    )


def test_followup_context_is_per_user_and_clears_with_topic_or_reset(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    state = service._user_states.get(42)
    state.history.append(ConversationTurn(STICKER_QUESTION, STICKER_EXPLANATION))
    assert service.is_request(42, "скинь какой-то")
    assert not service.is_request(43, "скинь какой-то")
    state.history.append(ConversationTurn("А какая погода?", "Дождь."))
    assert not service.is_request(42, "скинь какой-то")
    state.history.append(ConversationTurn(STICKER_QUESTION, STICKER_EXPLANATION))
    state.reset_context()
    assert not service.is_request(42, "скинь какой-то")


@pytest.mark.parametrize("outcome", ["sent", "disabled", "cooldown", "telegram_error"])
def test_screenshot_followup_sends_directly_or_reports_actual_failure(
    tmp_path: Path, outcome: str
) -> None:
    service, bot = _service(tmp_path)
    state = service._user_states.get(42)
    state.history.append(ConversationTurn(STICKER_QUESTION, STICKER_EXPLANATION))
    service._random_value = lambda: 1.0
    if outcome == "disabled":
        service._chance = 0
    elif outcome == "telegram_error":
        bot.send_sticker.side_effect = TelegramBadRequest(
            method=Mock(), message="failed"
        )
    engine = AsyncMock(spec=ResponseEngine)
    message = SimpleNamespace(
        text="скинь какой-то",
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=7),
        answer=AsyncMock(),
    )
    router = create_text_router(cast(ResponseEngine, engine), sticker_service=service)

    async def scenario() -> None:
        if outcome == "cooldown":
            assert await service.maybe_send(
                chat_id=7, user_id=42, context_text="Покажи стикер"
            )
            bot.send_sticker.reset_mock()
        await router.message.handlers[0].callback(cast(Message, message))

    asyncio.run(scenario())
    engine.respond_and_deliver.assert_not_awaited()
    assert bot.send_sticker.await_count == int(outcome in {"sent", "telegram_error"})
    if outcome == "sent":
        message.answer.assert_not_awaited()
        assert bot.send_sticker.await_args.kwargs["sticker"].startswith("safe-")
    else:
        message.answer.assert_awaited_once()
        assert "не получилось" in message.answer.await_args.args[0] or (
            "отключены" in message.answer.await_args.args[0]
        )


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


def test_text_handler_delivers_requested_sticker_without_llm(
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

    router = create_text_router(cast(ResponseEngine, engine), sticker_service=service)
    asyncio.run(router.message.handlers[0].callback(cast(Message, message)))
    message.answer.assert_not_awaited()
    engine.respond_and_deliver.assert_not_awaited()
    bot.send_sticker.assert_awaited_once()


@pytest.mark.parametrize("disabled", [False, True])
def test_text_request_reports_unavailable_or_cooldown(
    tmp_path: Path, disabled: bool
) -> None:
    service, bot = _service(tmp_path)
    if disabled:
        service._chance = 0
    else:
        asyncio.run(
            service.maybe_send(chat_id=7, user_id=42, context_text="Покажи стикер")
        )
    engine = AsyncMock(spec=ResponseEngine)
    message = SimpleNamespace(
        text="Покажи стикер",
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=7),
        answer=AsyncMock(),
    )
    router = create_text_router(cast(ResponseEngine, engine), sticker_service=service)
    asyncio.run(router.message.handlers[0].callback(cast(Message, message)))
    message.answer.assert_awaited_once()
    engine.respond_and_deliver.assert_not_awaited()


def test_regular_text_supplies_actual_capabilities(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    engine = AsyncMock(spec=ResponseEngine)
    message = SimpleNamespace(
        text="Как дела?",
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=7, type="private"),
        forward_origin=None,
    )
    router = create_text_router(cast(ResponseEngine, engine), sticker_service=service)
    asyncio.run(router.message.handlers[0].callback(cast(Message, message)))
    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert "доступны 2" in call.kwargs["trusted_input_context"]
    assert "Не отрицай" in call.kwargs["trusted_input_context"]


def test_first_greeting_selects_hello_and_respects_cooldown(tmp_path: Path) -> None:
    service, bot = _service(tmp_path)
    service._repository.upsert(StickerEntry("hello", "hello", ("greeting",)))
    service._random_value = lambda: 1.0
    now = [100.0]
    service._clock = lambda: now[0]

    async def scenario() -> None:
        assert await service.maybe_send(
            chat_id=7, user_id=42, context_text="Привет, хочу кофе"
        )
        assert not await service.maybe_send(
            chat_id=7, user_id=42, context_text="Привет"
        )
        now[0] += 181
        assert await service.maybe_send(chat_id=7, user_id=42, context_text="Привет")

    asyncio.run(scenario())
    assert all(c.kwargs["sticker"] == "hello" for c in bot.send_sticker.await_args_list)
    assert bot.send_sticker.await_count == 2


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_start_sends_sticker_after_greeting_and_keeps_age_prompt(
    tmp_path: Path, repeat: bool, failure: bool
) -> None:
    users = UsersRepository(tmp_path)
    if repeat:
        users.add(42)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Живое приветствие"
    service, bot = _service(tmp_path)
    service._repository.upsert(StickerEntry("hello", "hello", ("greeting",)))
    events: list[str] = []

    async def send_sticker(**kwargs: object) -> None:
        events.append("sticker")
        assert kwargs["sticker"] == "hello"
        if failure:
            raise RuntimeError("sticker unavailable")

    async def answer(text: str, **kwargs: object) -> None:
        events.append("text")

    bot.send_sticker.side_effect = send_sticker
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=42),
        chat=SimpleNamespace(id=7),
        answer=AsyncMock(side_effect=answer),
    )
    router = create_start_router(
        users, cast(DeepSeekService, model), "prompt", sticker_service=service
    )
    asyncio.run(router.message.handlers[0].callback(cast(Message, message)))
    assert events == ["text", "sticker", "text"]
