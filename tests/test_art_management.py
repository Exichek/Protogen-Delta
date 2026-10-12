"""Разрешения на загрузку артов и сохраняемые группы."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import Message

from protogen_delta.handlers.art import create_art_router
from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.images import ImagesRepository


def message(text: str = "", user: int = 1) -> Message:
    msg = Mock(spec=Message)
    msg.text = text
    msg.from_user = SimpleNamespace(id=user)
    msg.chat = SimpleNamespace(id=-100)
    msg.photo = []
    msg.document = None
    msg.reply_to_message = None
    msg.answer = AsyncMock()
    msg.answer_photo = AsyncMock()
    msg.answer_document = AsyncMock()
    return cast(Message, msg)


def test_sources_persist_and_validate(tmp_path: Path) -> None:
    sources = ArtSourcesRepository(tmp_path, -100)
    assert sources.get_all() == [-100]
    assert sources.change(-200, add=True)
    assert not sources.change(-200, add=True)
    assert sources.change(-100, add=False)
    assert not sources.change(-100, add=False)
    assert ArtSourcesRepository(tmp_path, -300).get_all() == [-200]
    with pytest.raises(ValueError):
        sources.change(1, add=True)


def test_art_commands_and_admin_only_ingestion(tmp_path: Path) -> None:
    images = ImagesRepository(tmp_path)
    sources = ArtSourcesRepository(tmp_path, -100)
    router = create_art_router(images, -100, frozenset({1}), sources)

    async def call(index: int, msg: Message) -> None:
        await router.message.handlers[index].callback(msg)

    async def scenario() -> None:
        for index, text in [(3, "/artchat add -200"), (4, "/addimage secret")]:
            await call(index, message(text, user=2))
        assert sources.get_all() == [-100]
        assert images.get_all() == []
        await call(3, message("/artchat add -200"))
        await call(3, message("/artchat list"))
        await call(3, message("/artchat remove -200"))
        await call(3, message("/artchat bad"))
        await call(3, message("/artchat add nonsense"))
        await call(4, message("/addimage a,b a"))
        assert images.get_all() == ["a", "b"]
        await call(4, message("/addimage"))
        photo = message(user=2)
        photo.photo = [SimpleNamespace(file_id="unauthorized")]  # type: ignore[list-item]
        await call(0, photo)
        assert "unauthorized" not in images.get_all()
        photo.from_user = SimpleNamespace(id=1)  # type: ignore[assignment]
        await call(0, photo)
        assert "unauthorized" in images.get_all()
        document = message()
        document.document = SimpleNamespace(file_id="document", mime_type="image/png")  # type: ignore[assignment]
        await call(1, document)
        assert images.get_kind("document") == "document"
        assert images.get_kind("a") == "photo"
        for file_id in ["a", "b", "unauthorized"]:
            images.remove(file_id)
        await call(2, message())
        assert images.get_all() == ["document"]
        images.remove("document")
        assert images.get_kind("document") == "photo"

    asyncio.run(scenario())
