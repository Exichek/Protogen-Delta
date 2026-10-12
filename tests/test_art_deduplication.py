"""Повторные Telegram-ID не создают дубль; массовый импорт пишет один раз."""

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Chat, Document, Update
from test_art_channels import _photo
from test_art_management import message as command

from protogen_delta.handlers.art import create_art_router
from protogen_delta.repositories.images import ImagesRepository


def test_unique_identity_survives_restart_and_keeps_old_code(tmp_path: Path) -> None:
    images = ImagesRepository(tmp_path)
    assert images.add("old-id")
    assert not images.add("old-id", file_unique_id="same-photo")
    restored = ImagesRepository(tmp_path)
    assert not restored.add("new-id", file_unique_id="same-photo")
    assert restored.get_all() == ["old-id"]
    assert restored.add("other", kind="document", file_unique_id="other-photo")
    assert restored.get_kind("other") == "document"
    assert restored.remove("old-id")
    assert restored.add("new-id", file_unique_id="same-photo")
    assert restored.get_all() == ["other", "new-id"]
    metadata = json.loads((tmp_path / "images.json").read_text("utf-8"))
    assert metadata["UNIQUE_IDS"] == {
        "other": "other-photo",
        "new-id": "same-photo",
    }


def test_batch_import_writes_once_and_duplicate_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = ImagesRepository(tmp_path)
    images.get_all()
    write = Mock(wraps=images._storage._save)
    monkeypatch.setattr(images._storage, "_save", write)
    ids = [f"file-{index}" for index in range(100)]
    assert images.add_many(ids + ids) == 100
    write.assert_called_once()
    write.reset_mock()
    assert images.add_many(ids) == 0
    write.assert_not_called()
    assert ImagesRepository(tmp_path).get_all() == ids


def test_channel_reposts_and_addimage_reply_deduplicate(tmp_path: Path) -> None:
    images = ImagesRepository(tmp_path)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_art_router(images, -100, frozenset({1})))
    bot = Bot("123456:test-token")
    channel = Chat(id=-100, type="channel")

    async def scenario() -> None:
        original = _photo(channel, "original")
        assert original.photo
        duplicate = original.model_copy(
            update={
                "photo": [
                    original.photo[-1].model_copy(update={"file_id": "new-identifier"})
                ]
            }
        )
        doc = original.model_copy(
            update={
                "photo": None,
                "document": Document(
                    file_id="document",
                    file_unique_id="doc-unique",
                    mime_type="image/png",
                ),
            }
        )
        for index, message in enumerate([original, duplicate, doc]):
            await dispatcher.feed_update(
                bot, Update(update_id=index, channel_post=message)
            )
        # Команда в ответ на повторный документ использует тот же путь дедупликации.
        reply = command("/addimage")
        assert doc.document is not None
        reply.reply_to_message = doc.model_copy(
            update={"document": doc.document.model_copy(update={"file_id": "new-doc"})}
        )
        await dispatcher.sub_routers[0].message.handlers[4].callback(reply)
        call = cast(AsyncMock, reply.answer).await_args
        assert call is not None
        assert "Добавлено артов: 0" in call.args[0]
        assert ImagesRepository(tmp_path).get_all() == ["original", "document"]
        await dispatcher.storage.close()
        await bot.session.close()

    asyncio.run(scenario())
