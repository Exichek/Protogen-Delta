"""Рейтинги и нейтральные поводы для пяти дополнительных реакций."""

import asyncio
import json
from importlib.resources import files
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.stickers import (
    StickersRepository,
    parse_sticker_entries,
)
from protogen_delta.services.stickers import (
    ContextualStickerService,
    sticker_context_tags,
)


def test_new_sticker_manifest_has_requested_ratings_and_unique_ids() -> None:
    raw = json.loads(
        files("protogen_delta.config")
        .joinpath("data/delta_stickers.json")
        .read_text("utf-8")
    )
    entries = parse_sticker_entries(raw)
    by_name = {entry.name: entry for entry in entries}
    assert len(entries) == len({entry.file_unique_id for entry in entries})
    for name in ("boop_2", "bite"):
        assert by_name[name].rating == "safe"
    for name in ("good_boy", "ur_my_pet", "kiss"):
        assert by_name[name].rating == "adult"
        assert "horny" not in by_name[name].tags
    assert "you_suck" not in by_name and "you_suck2" not in by_name


@pytest.mark.parametrize(
    "name,text,mode,sent",
    [
        ("boop_2", "Буп!", "soft", True),
        ("bite", "Кусь", "soft", True),
        ("good_boy", "Похвали меня", "adult", True),
        ("ur_my_pet", "Обними меня", "adult", True),
        ("kiss", "Поцелуй", "adult", True),
        ("good_boy", "Похвали меня", "soft", False),
        ("ur_my_pet", "Обними меня", "soft", False),
        ("kiss", "Поцелуй", "soft", False),
        ("kiss", "Продолжим", "adult", False),
    ],
)
def test_new_reactions_need_matching_topic_and_rating(
    tmp_path: Path, name: str, text: str, mode: str, sent: bool
) -> None:
    raw = json.loads(
        files("protogen_delta.config")
        .joinpath("data/delta_stickers.json")
        .read_text("utf-8")
    )
    entry = next(entry for entry in parse_sticker_entries(raw) if entry.name == name)
    repository = StickersRepository(tmp_path)
    repository.upsert(entry)
    states = UserStateStore()
    state = states.get(42)
    state.content_mode = "adult" if mode == "adult" else "soft"
    state.roleplay_active = True
    state.mood = "horny"
    state.emotions.arousal = 1.0
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(
        cast(Bot, bot), repository, states, min_replies=1, chance=1
    )
    assert (
        asyncio.run(service.maybe_send(chat_id=42, user_id=42, context_text=text))
        is sent
    )
    assert bot.send_sticker.await_count == int(sent)


def test_gesture_words_in_code_and_quotes_are_not_reaction_requests() -> None:
    assert not sticker_context_tags('`kiss` и «буп», "укусить"')
    assert "kiss" not in sticker_context_tags(
        "Нужна целая таблица; используй целую строку"
    )
