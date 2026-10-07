"""Регрессии пропущенных реакций и неуместных сообщений после молчания."""

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.methods import SendMessage
from aiogram.types import Message

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.delivery import create_reply_delivery
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.stickers import StickerEntry, StickersRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger
from protogen_delta.services.stickers import (
    ContextualStickerService,
    sticker_context_tags,
    sticker_reply_tags,
)


@pytest.mark.parametrize(
    "text", ["Класс :D", "😂", "🤣", "лол", "Спасибо :Д", "xD", "Хд"]
)
def test_laughter_markers_match(text: str) -> None:
    assert "laugh" in sticker_context_tags(text)


@pytest.mark.parametrize("text", [":)", ":-)", "😊", ")))", "ясно)))"])
def test_smiles_match_happy(text: str) -> None:
    assert "happy" in sticker_context_tags(text)


@pytest.mark.parametrize("text", [";)", ":3", "UwU", "OwO", "😜"])
def test_playful_faces_do_not_imply_intimate_context(text: str) -> None:
    tags = sticker_context_tags(text)
    assert "playful" in tags
    assert not {"horny", "oral", "rimming", "rp"}.intersection(tags)


@pytest.mark.parametrize(
    "text",
    ["`:D`", "```python\nprint(':)')\n```", "https://example.com/:D", "print(foo())"],
)
def test_code_and_urls_do_not_become_reactions(text: str) -> None:
    assert sticker_context_tags(text) == set()


def test_reply_reaction_ignores_quoted_topics() -> None:
    assert sticker_reply_tags("В статье написано: он улыбнулся и рассмеялся.") == set()
    assert sticker_reply_tags("*улыбнулся* Готово!") == {"happy"}
    assert sticker_reply_tags("*рассмеялся* Ха!") == {"laugh"}
    assert sticker_reply_tags("*смутился* Упс.") == {"oops"}


def test_explicit_sticker_survives_random_gate_but_obeys_intervals(
    tmp_path: Path,
) -> None:
    repository = StickersRepository(tmp_path)
    repository.upsert(StickerEntry("file-laugh", "laugh", ("laugh",), "safe"))
    bot = AsyncMock(spec=Bot)
    now = [100.0]
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        UserStateStore(),
        chance=0.15,
        min_replies=2,
        cooldown_seconds=60,
        random_value=lambda: 0.99,
        clock=lambda: now[0],
    )

    async def scenario() -> None:
        assert not await service.maybe_send(chat_id=7, user_id=42, context_text=":D")
        assert await service.maybe_send(chat_id=7, user_id=42, context_text=":D")
        assert not await service.maybe_send(chat_id=7, user_id=42, context_text=":D")
        assert not await service.maybe_send(chat_id=7, user_id=42, context_text=":D")
        now[0] += 61
        assert await service.maybe_send(
            chat_id=7, user_id=42, reply_text="*рассмеялся*"
        )

    asyncio.run(scenario())
    assert bot.send_sticker.await_count == 2


def test_disabled_sticker_reactions_stay_disabled(tmp_path: Path) -> None:
    repository = StickersRepository(tmp_path)
    repository.upsert(StickerEntry("file-happy", "happy", ("happy",), "safe"))
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot), repository, UserStateStore(), chance=0, min_replies=1
    )
    assert not asyncio.run(
        service.maybe_send(chat_id=7, user_id=42, context_text="Спасибо")
    )
    bot.send_sticker.assert_not_awaited()


def test_generic_mood_still_uses_probability(tmp_path: Path) -> None:
    repository = StickersRepository(tmp_path)
    repository.upsert(StickerEntry("file-playful", "playful", ("playful",), "safe"))
    states = UserStateStore()
    states.get(42).mood = "playful"
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=0.15,
        min_replies=1,
        random_value=lambda: 0.99,
    )
    assert not asyncio.run(
        service.maybe_send(chat_id=7, user_id=42, context_text="Поясни DNS")
    )
    bot.send_sticker.assert_not_awaited()


def test_sticker_emoji_and_reply_reach_selection() -> None:
    message = Mock(spec=Message)
    message.chat = SimpleNamespace(id=7, type="private")
    message.text = message.caption = None
    message.sticker = SimpleNamespace(emoji="😂")
    message.answer = AsyncMock()
    service = AsyncMock(spec=ContextualStickerService)
    reply = "*рассмеялся* Вот это да."
    asyncio.run(
        create_reply_delivery(
            cast(Message, message),
            None,
            cast(ContextualStickerService, service),
            user_id=42,
        )(reply)
    )
    assert service.maybe_send.await_args.kwargs["context_text"].strip() == "😂"
    assert service.maybe_send.await_args.kwargs["reply_text"] == reply


def _messenger(
    repository: MemoriesRepository, bot: Any, model: Any
) -> ProactiveMessenger:
    return ProactiveMessenger(
        bot=cast(Bot, bot),
        deepseek=cast(DeepSeekService, model),
        repository=repository,
        system_prompt="personality",
        config=ProactiveConfig(idle_seconds=20, cooldown_seconds=100),
        clock=lambda: 40.0,
        local_datetime=lambda: datetime(2026, 10, 4, 12),
    )


def test_proactive_omits_grievances_and_disables_tools(tmp_path: Path) -> None:
    repository = MemoriesRepository(tmp_path)
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Привет! Могу помочь с чем-нибудь, если захочешь."

    async def scenario() -> None:
        await repository.note_user_activity(7, 10)
        await repository.remember(7, "grievance", "старый конфликт", 12)
        await repository.remember(7, "topic", "нет настроения", 13)
        await repository.remember(7, "topic", "обсуждали фетиши", 14)
        await repository.remember(7, "topic", "Python", 11)
        assert await _messenger(repository, bot, model).run_once() == 1

    asyncio.run(scenario())
    args = model.chat.await_args.kwargs
    assert args["tool_names"] == ()
    assert "старый конфликт" not in args["user_message"]
    assert "нет настроения" not in args["user_message"]
    assert "фетиши" not in args["user_message"]
    assert "Python" in args["user_message"]
    assert "0 дней назад" in args["user_message"]
    assert "не означает плохое настроение" in args["system_prompt"]


@pytest.mark.parametrize(
    "text",
    [
        "Скучаю по нашим вечерам. Ты сегодня опять без настроения?",
        "Почему ты не отвечаешь?",
        "Опять пропал!",
        "Найдёшь, чем себя занять?",
    ],
)
def test_proactive_replaces_pressure(tmp_path: Path, text: str) -> None:
    repository = MemoriesRepository(tmp_path)
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = text

    async def scenario() -> None:
        await repository.note_user_activity(7, 10)
        assert await _messenger(repository, bot, model).run_once() == 1

    asyncio.run(scenario())
    assert bot.send_message.await_args.args[1] == (
        "Привет! Если захочешь поболтать или разобрать что-нибудь, я рядом."
    )


def test_proactive_enforces_actual_length(tmp_path: Path) -> None:
    repository = MemoriesRepository(tmp_path)
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Привет!\n\n" + "Слово " * 80

    async def scenario() -> None:
        await repository.note_user_activity(7, 10)
        assert await _messenger(repository, bot, model).run_once() == 1

    asyncio.run(scenario())
    text = bot.send_message.await_args.args[1]
    assert len(text) <= 240 and "\n" not in text and text.endswith("…")


@pytest.mark.parametrize("change", ["activity", "disable", "delete"])
def test_proactive_does_not_send_after_state_changes(
    tmp_path: Path, change: str
) -> None:
    repository = MemoriesRepository(tmp_path)
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)

    async def generate(**kwargs: Any) -> str:
        if change == "activity":
            await repository.note_user_activity(7, 39)
        elif change == "disable":
            await repository.set_proactive(7, False, 39)
        else:
            await repository.delete_user(7)
        return "Привет!"

    model.chat.side_effect = generate

    async def scenario() -> None:
        await repository.note_user_activity(7, 10)
        assert await _messenger(repository, bot, model).run_once() == 0

    asyncio.run(scenario())
    bot.send_message.assert_not_awaited()


@pytest.mark.parametrize("reason", ["missing", "blocked", "other"])
def test_unreachable_proactive_recipient_does_not_repeat_generation(
    tmp_path: Path, reason: str
) -> None:
    repository = MemoriesRepository(tmp_path)
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = "Привет!"
    method = SendMessage(chat_id=7, text="hello")
    bot.send_message.side_effect = (
        TelegramForbiddenError(method, "bot was blocked")
        if reason == "blocked"
        else TelegramBadRequest(
            method,
            "Bad Request: chat not found" if reason == "missing" else "invalid markup",
        )
    )

    async def scenario() -> None:
        await repository.note_user_activity(7, 10)
        messenger = _messenger(repository, bot, model)
        assert await messenger.run_once() == 0
        assert await repository.proactive_enabled(7)
        assert await messenger.run_once() == 0
        assert model.chat.await_count == (2 if reason == "other" else 1)
        await repository.note_user_activity(7, 15)
        bot.send_message.side_effect = None
        assert await messenger.run_once() == 1
        await repository.set_proactive(7, False, 15)
        await repository.suspend_proactive_delivery(7)
        await repository.note_user_activity(7, 15)
        assert not await repository.proactive_enabled(7)
        assert await messenger.run_once() == 0

    asyncio.run(scenario())


def test_old_engagement_schema_migrates_without_losing_preferences(
    tmp_path: Path,
) -> None:
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(tmp_path / "memories.db")) as connection, connection:
        connection.execute(
            "CREATE TABLE engagement (user_id INTEGER PRIMARY KEY, "
            "proactive_enabled INTEGER NOT NULL DEFAULT 1, "
            "last_user_message_at REAL NOT NULL DEFAULT 0, "
            "last_proactive_at REAL, unanswered_count INTEGER NOT NULL DEFAULT 0)"
        )
        connection.execute("INSERT INTO engagement VALUES (7, 0, 10, NULL, 0)")

    async def scenario() -> None:
        repository = MemoriesRepository(tmp_path)
        assert not await repository.proactive_enabled(7)
        await repository.suspend_proactive_delivery(7)
        await repository.set_proactive(7, True, 15)
        fresh = MemoriesRepository(tmp_path)
        assert await fresh.proactive_enabled(7)
        assert (
            len(
                await fresh.due_candidates(
                    now=40, idle_seconds=20, cooldown_seconds=100, limit=10
                )
            )
            == 1
        )

    asyncio.run(scenario())
