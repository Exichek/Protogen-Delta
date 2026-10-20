"""Повод, частота и подтверждение нативных реакций без Telegram и модели."""

import asyncio
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Chat, Message, User

from protogen_delta.core.reaction_receipt import (
    reaction_receipt,
    reaction_receipt_scope,
    with_reaction_receipt,
)
from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.reply_outcome import error_reply
from protogen_delta.core.user_state import UserState, UserStateStore
from protogen_delta.handlers.delivery import create_reply_delivery
from protogen_delta.services.message_reactions import (
    MessageReactionService,
    choose_message_reaction,
)
from protogen_delta.services.stickers import ContextualStickerService


def _state() -> UserState:
    state = UserState()
    state.emotions.warmth = state.emotions.playfulness = 0.4
    state.emotions.memory.joy = state.emotions.memory.curiosity = 0.4
    return state


def _message(number: int = 10, **changes: Any) -> Message:
    message = Message(
        message_id=number,
        date=datetime(2026, 10, 16, tzinfo=timezone.utc),
        chat=Chat(id=42, type="private"),
        from_user=User(id=42, is_bot=False, first_name="Тест"),
        text="Обнимаю тебя ❤",
    )
    return message.model_copy(update=changes)


def _service(**options: Any) -> tuple[MessageReactionService, AsyncMock]:
    bot = AsyncMock(spec=Bot)
    bot.set_message_reaction.return_value = True
    states = UserStateStore()
    states.get(42).emotions.warmth = 0.4
    states.get(43).emotions.warmth = 0.4
    return MessageReactionService(cast(Bot, bot), states, **options), bot


@pytest.mark.parametrize(
    "text,reply,expected",
    [
        ("Обнимаю тебя ❤", "И я тебя обнимаю.", "❤"),
        ("Ахаха, вот это номер!", "Ахаха, да!", "🤣"),
        ("Ура, у меня получилось!", "Круто, поздравляю!", "🔥"),
        ("Спасибо, ты молодец", "Спасибо, мне приятно 😊", "👍"),
        ("Интересная загадка, что думаешь?", "Надо подумать, интересно.", "🤔"),
        ("Мне грустно, спасибо что рядом", "Жаль, обнимаю.", "😢"),
        ("Меня уволили 😂", "Очень жаль, сочувствую.", "😢"),
        ("Не получилось, но ахаха", "Вот это номер!", None),
        ("Не люблю, когда меня обнимают", "Понял.", None),
        ("Не смешно 😂", "Ахаха", None),
        ("Не обнимай ❤", "Хорошо", None),
        ("Без реакций, спасибо ❤", "Спасибо 😊", None),
        ("Не ставь реакции ❤", "Хорошо", None),
        ("Перестань реагировать ❤", "Хорошо", None),
        ("Посмотри код в репозитории", "Готово, проверил.", None),
        ("Он написал «обнимаю ❤»", "Понял.", None),
        ("Он написал “ахаха 😂”", "Ахаха", None),
        ('```python\nprint("Обнимаю ❤")\n```', "Спасибо 😊", None),
        ("> Обнимаю ❤\nЭто цитата", "Спасибо 😊", None),
        ("https://example.org/❤", "Спасибо 😊", None),
    ],
)
def test_reaction_needs_current_unquoted_reason(
    text: str, reply: str, expected: str | None
) -> None:
    assert choose_message_reaction(text, reply, _state()) == expected


def test_emotions_change_selection_but_do_not_create_a_reason() -> None:
    state = UserState()
    assert choose_message_reaction("Обнимаю ❤", "Обнимаю", state) is None
    state.emotions.warmth = 0.4
    assert choose_message_reaction("Обнимаю ❤", "Обнимаю", state) == "❤"
    state.emotions.irritation = 0.8
    assert choose_message_reaction("Обнимаю ❤", "Обнимаю", state) is None
    state.emotions.irritation = 0
    state.mood = "horny"
    assert choose_message_reaction("Обнимаю ❤", "Обнимаю", state) is None
    state.mood = "angry"
    assert choose_message_reaction("Обнимаю ❤", "Обнимаю", state) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"chat": Chat(id=-42, type="group")},
        {"chat": Chat(id=43, type="private")},
        {"from_user": None},
        {"from_user": User(id=43, is_bot=False, first_name="Другой")},
        {"from_user": User(id=42, is_bot=True, first_name="Бот")},
        {"forward_origin": object()},
        {"message_id": 0},
        {"text": "/reset ❤"},
        {"text": None},
    ],
)
def test_scope_and_non_dialogue_messages_do_not_react(changes: dict[str, Any]) -> None:
    service, bot = _service(chance=1)
    assert asyncio.run(service.maybe_react(_message(**changes), "Обнимаю", 42)) is None
    bot.set_message_reaction.assert_not_awaited()


def test_batch_and_application_error_do_not_react() -> None:
    service, bot = _service(chance=1)

    async def run() -> None:
        token = current_reply_sources.set((ReplySource("❤", "❤", True, True),) * 2)
        try:
            assert await service.maybe_react(_message(), "Обнимаю", 42) is None
        finally:
            current_reply_sources.reset(token)
        with error_reply(True):
            assert await service.maybe_react(_message(), "Ошибка", 42) is None

    asyncio.run(run())
    bot.set_message_reaction.assert_not_awaited()


def test_cooldown_gap_duplicate_and_per_chat_isolation() -> None:
    now = [1000.0]
    service, bot = _service(chance=1, clock=lambda: now[0])

    async def run() -> None:
        assert await service.maybe_react(_message(), "Обнимаю", 42) == "❤"
        assert await service.maybe_react(_message(), "Обнимаю", 42) is None
        assert await service.maybe_react(_message(11), "Обнимаю", 42) is None
        other = _message(
            chat=Chat(id=43, type="private"),
            from_user=User(id=43, is_bot=False, first_name="Другой"),
        )
        assert await service.maybe_react(other, "Обнимаю", 43) == "❤"
        now[0] += 121
        assert await service.maybe_react(_message(12), "Обнимаю", 42) is None
        assert await service.maybe_react(_message(13), "Обнимаю", 42) == "❤"

    asyncio.run(run())
    assert bot.set_message_reaction.await_count == 3
    arguments = bot.set_message_reaction.await_args.kwargs
    assert arguments["message_id"] == 13 and arguments["is_big"] is False
    assert [item.emoji for item in arguments["reaction"]] == ["❤"]


def test_parallel_delivery_has_one_attempt() -> None:
    service, bot = _service(chance=1)

    async def run() -> None:
        results = await asyncio.gather(
            *(service.maybe_react(_message(), "Обнимаю", 42) for _ in range(5))
        )
        assert results.count("❤") == 1

    asyncio.run(run())
    bot.set_message_reaction.assert_awaited_once()


@pytest.mark.parametrize("failure", ["api", "timeout", "false"])
def test_failed_attempt_is_not_confirmation_and_not_retried(failure: str) -> None:
    service, bot = _service(chance=1)
    if failure == "api":
        bot.set_message_reaction.side_effect = TelegramAPIError(Mock(), "unavailable")
    elif failure == "timeout":
        bot.set_message_reaction.side_effect = TimeoutError
    else:
        bot.set_message_reaction.return_value = False

    async def run() -> None:
        assert await service.maybe_react(_message(), "Обнимаю", 42) is None
        assert await service.maybe_react(_message(11), "Обнимаю", 42) is None

    asyncio.run(run())
    bot.set_message_reaction.assert_awaited_once()


@pytest.mark.parametrize("chance,draw", [(0.0, 0.0), (0.3, 0.4)])
def test_disabled_or_random_skip(chance: float, draw: float) -> None:
    service, bot = _service(chance=chance, random_value=lambda: draw)
    assert asyncio.run(service.maybe_react(_message(), "Обнимаю", 42)) is None
    bot.set_message_reaction.assert_not_awaited()


@pytest.mark.parametrize("value", [-1.0, 2.0, float("nan"), float("inf")])
def test_invalid_chance(value: float) -> None:
    with pytest.raises(ValueError):
        _service(chance=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_cooldown(value: float) -> None:
    with pytest.raises(ValueError):
        _service(cooldown_seconds=value)


@pytest.mark.parametrize("outcome", ["sent", "unavailable", "failure", "text_failed"])
def test_delivery_records_only_real_reaction_and_avoids_sticker(outcome: str) -> None:
    message = Mock(spec=Message)
    message.chat = Mock(id=42, type="private")
    message.text, message.caption, message.sticker = "Обнимаю ❤", None, None
    message.answer = AsyncMock()
    reactions = AsyncMock(spec=MessageReactionService)
    reactions.maybe_react.return_value = "❤" if outcome == "sent" else None
    stickers = AsyncMock(spec=ContextualStickerService)
    if outcome == "failure":
        reactions.maybe_react.side_effect = RuntimeError("unexpected")
    if outcome == "text_failed":
        message.answer.side_effect = TelegramAPIError(Mock(), "unavailable")
    deliver = create_reply_delivery(
        cast(Message, message),
        None,
        cast(ContextualStickerService, stickers),
        reaction_service=cast(MessageReactionService, reactions),
        user_id=42,
    )

    async def run() -> None:
        with reaction_receipt_scope():
            if outcome == "text_failed":
                with pytest.raises(TelegramAPIError):
                    await deliver("Обнимаю")
                reactions.maybe_react.assert_not_awaited()
                return
            await deliver("Обнимаю")
            assert reaction_receipt.get() == ("❤" if outcome == "sent" else None)
            assert ("Подтверждённое" in with_reaction_receipt("Ответ")) is (
                outcome == "sent"
            )
            assert stickers.maybe_send.await_count == (0 if outcome == "sent" else 1)
        assert reaction_receipt.get() is None

    asyncio.run(run())
