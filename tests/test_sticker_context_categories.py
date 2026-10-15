"""Контекстная разметка, настоящий сбой 404 и неизменность возрастного фильтра."""

import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.reply_outcome import error_reply, reply_error
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.delivery import create_reply_delivery
from protogen_delta.handlers.failure_reply import send_failure_reply
from protogen_delta.repositories.stickers import (
    StickerEntry,
    StickersRepository,
    parse_sticker_entries,
)
from protogen_delta.services.deepseek import DeepSeekTimeoutError
from protogen_delta.services.sticker_pack import StickerPackImporter
from protogen_delta.services.sticker_reaction_context import reaction_context
from protogen_delta.services.stickers import ContextualStickerService


def catalog() -> list[StickerEntry]:
    return parse_sticker_entries(
        json.loads(
            Path("src/protogen_delta/config/data/delta_stickers.json").read_text(
                "utf-8"
            )
        )
    )


@pytest.mark.parametrize(
    ("name", "contexts"),
    [
        ("hi_sticker", ("greeting",)),
        ("hi_sticker2", ("greeting",)),
        ("hi_sticker3", ("greeting",)),
        ("harmful_stiker", ("refusal",)),
        ("harmful_stiker2", ("agreement",)),
        ("okay_sticker", ("agreement",)),
        ("error", ("error",)),
        ("boring", ("uncertain",)),
        ("oooops", ("uncertain",)),
        ("beep", ("joy",)),
        ("love", ("affection",)),
        ("rofl_whezeze", ("laugh",)),
        ("toster_sushi", ("everyday", "joy")),
    ],
)
def test_user_pictures_match_explicit_contexts(
    name: str, contexts: tuple[str, ...]
) -> None:
    entry = next(item for item in catalog() if item.name == name)
    assert entry.context_categories == contexts


@pytest.mark.parametrize(
    ("tags", "reply", "expected"),
    [
        (set(), "Нет, не хочу", "refusal"),
        (set(), "Не согласен с этим", "refusal"),
        (set(), "Да, согласен", "agreement"),
        ({"okay"}, "", "agreement"),
        (set(), "Не уверен, нужно проверить", "uncertain"),
        ({"confused"}, "", "uncertain"),
        ({"greeting", "happy"}, "", "greeting"),
        ({"love"}, "", "affection"),
        ({"laugh"}, "", "laugh"),
        ({"happy"}, "", "joy"),
        (set(), "Ошибка в коде считается отдельно", "everyday"),
        (set(), "Он сказал «Нет, не хочу»", "everyday"),
        (set(), "```\nНет, не хочу\n```", "everyday"),
    ],
)
def test_reaction_intent_does_not_confuse_quote_with_own_refusal(
    tags: set[str], reply: str, expected: str
) -> None:
    assert reaction_context(tags, reply) == expected


@pytest.mark.parametrize("failure", ["timeout", "empty", "whitespace"])
def test_true_model_error_always_uses_404_but_error_discussion_does_not(
    tmp_path: Path,
    failure: str,
) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        repo = StickersRepository(tmp_path)
        repo.import_entries(catalog())
        bot = AsyncMock(spec=Bot)
        service = ContextualStickerService(
            cast(Bot, bot),
            repo,
            engine._user_states,
            chance=0.001,
            min_replies=100,
            cooldown_seconds=999999,
            random_value=lambda: 1,
        )
        event = Mock(spec=Message)
        event.from_user = SimpleNamespace(id=TEST_USER_ID)
        event.chat = SimpleNamespace(id=TEST_USER_ID, type="private")
        event.text = "Разбери ситуацию"
        event.caption = None
        event.sticker = None
        event.answer = AsyncMock()
        if failure == "timeout":
            model.chat.side_effect = DeepSeekTimeoutError("synthetic")
        else:
            model.chat.return_value = "" if failure == "empty" else "   "
        for text in ("Первый вопрос", "Другой вопрос"):
            await engine.respond_and_deliver(
                TEST_USER_ID,
                text,
                create_reply_delivery(
                    cast(Message, event), None, service, user_id=TEST_USER_ID
                ),
            )
        expected = next(item.file_id for item in catalog() if item.name == "error")
        assert bot.send_sticker.await_count == 2
        assert all(
            call.kwargs["sticker"] == expected
            for call in bot.send_sticker.await_args_list
        )
        assert not reply_error.get()
        assert not engine._user_states.get(TEST_USER_ID).history
        assert engine._user_states.get(TEST_USER_ID).reply_count == 0
        assert not await service.maybe_send(
            chat_id=TEST_USER_ID,
            user_id=TEST_USER_ID,
            context_text="Как исправить ошибку 404?",
            reply_text="Сначала проверим код",
        )
        assert bot.send_sticker.await_count == 2

    asyncio.run(run())


def test_category_prevents_f_becoming_error_or_no_becoming_random(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = StickersRepository(tmp_path)
        repo.import_entries(catalog())
        bot = AsyncMock(spec=Bot)
        states = UserStateStore()
        service = ContextualStickerService(
            cast(Bot, bot),
            repo,
            states,
            chance=1,
            min_replies=1,
            cooldown_seconds=0,
            choose=lambda entries: entries[0],
        )
        assert await service.maybe_send(chat_id=1, user_id=1, reply_text="Да, согласен")
        sent = bot.send_sticker.await_args
        assert sent is not None
        assert sent.kwargs["sticker"] == next(
            item.file_id for item in catalog() if item.name == "harmful_stiker2"
        )
        assert await service.maybe_send(
            chat_id=1, user_id=1, reply_text="Не хочу сейчас"
        )
        assert bot.send_sticker.await_args.kwargs["sticker"] == next(
            item.file_id for item in catalog() if item.name == "harmful_stiker"
        )
        assert await service.maybe_send(
            chat_id=1,
            user_id=1,
            context_text="Ты не хочешь?",
            reply_text="Посмотрю задачу",
        )
        sent_id = bot.send_sticker.await_args.kwargs["sticker"]
        assert "everyday" in next(
            item.context_categories for item in catalog() if item.file_id == sent_id
        )

    asyncio.run(run())


def test_offline_context_migration_preserves_manual_rating_and_tags(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = StickersRepository(tmp_path)
        known = next(item for item in catalog() if item.name == "harmful_stiker2")
        repo.upsert(replace(known, contexts=(), tags=("custom",)))
        adult = next(item for item in catalog() if item.rating == "adult")
        repo.upsert(replace(adult, contexts=()))
        bot = AsyncMock(spec=Bot)
        bot.get_sticker_set.side_effect = TelegramBadRequest(Mock(), "synthetic")
        with pytest.raises(TelegramBadRequest):
            await StickerPackImporter(cast(Bot, bot), repo).sync()
        entries = repo.get_all()
        assert entries[0].tags == ("custom",) and entries[0].contexts == ("agreement",)
        assert entries[1].rating == "adult" and entries[1].contexts == ("adult",)
        repo.upsert(replace(entries[0], contexts=("everyday",)))
        repo.import_contexts(catalog())
        assert repo.get_all()[0].contexts == ("everyday",)

    asyncio.run(run())


@pytest.mark.parametrize("contexts", [["unknown"], ["adult"], "joy", [1]])
def test_invalid_context_metadata_rejected(contexts: object) -> None:
    raw = asdict(catalog()[0])
    raw["contexts"] = contexts
    with pytest.raises(TypeError):
        parse_sticker_entries({"STICKERS": [raw]})


def test_failed_error_sticker_does_not_retry_the_text(tmp_path: Path) -> None:
    async def run() -> None:
        repo = StickersRepository(tmp_path)
        repo.import_entries(catalog())
        bot = AsyncMock(spec=Bot)
        bot.send_sticker.side_effect = TelegramBadRequest(Mock(), "synthetic")
        service = ContextualStickerService(cast(Bot, bot), repo, UserStateStore())
        event = Mock(spec=Message)
        event.from_user = SimpleNamespace(id=1)
        event.chat = SimpleNamespace(id=1)
        event.answer = AsyncMock()
        await send_failure_reply(cast(Message, event), "Ошибка чтения", service)
        event.answer.assert_awaited_once_with("Ошибка чтения")
        bot.send_sticker.assert_awaited_once()
        disabled = ContextualStickerService(
            cast(Bot, bot), repo, UserStateStore(), chance=0
        )
        with error_reply(True):
            assert not await disabled.send_error(chat_id=1, user_id=1)
        assert not reply_error.get()

    asyncio.run(run())


def test_protogen_glossary_is_scoped_and_reaches_visual_passes() -> None:
    for name in (
        "appearance_observation",
        "appearance_verification",
        "conversation_visual",
        "furry_species_reference",
        "personality/protogen_lore",
    ):
        prompt = load_prompt(name)
        assert "боковой дисплей визора" in prompt and "404" in prompt
        assert "другие виды" in prompt
