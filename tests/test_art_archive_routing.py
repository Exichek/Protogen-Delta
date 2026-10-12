"""Источник коллекции перехватывает вложения до разговорных роутеров."""

import asyncio
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher, Router
from aiogram.client.session.base import BaseSession
from aiogram.types import (
    Animation,
    Chat,
    Document,
    Message,
    MessageEntity,
    PhotoSize,
    Sticker,
    Update,
    User,
    Video,
    Voice,
)

from protogen_delta.handlers.art import create_art_router
from protogen_delta.handlers.documents import create_document_router
from protogen_delta.handlers.media import create_media_router
from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.services.response_engine import ResponseEngine


def _message(chat: Chat, **values: object) -> Message:
    return Message.model_validate(
        {
            "message_id": 1,
            "date": datetime.now(timezone.utc),
            "chat": chat,
            "from_user": User(id=1, is_bot=False, first_name="Admin"),
            **values,
        }
    )


def test_archive_content_never_reaches_media_or_conversation(tmp_path: Path) -> None:
    images = ImagesRepository(tmp_path)
    sources = ArtSourcesRepository(tmp_path, -100)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_art_router(images, -100, frozenset({1}), sources))
    engine = AsyncMock(spec=ResponseEngine)
    downloader = AsyncMock(spec=Bot)
    dispatcher.include_router(
        create_media_router(cast(ResponseEngine, engine), cast(Bot, downloader))
    )
    dispatcher.include_router(
        create_document_router(cast(ResponseEngine, engine), cast(Bot, downloader))
    )
    fallback = AsyncMock()
    fallback_router = Router()

    @fallback_router.message()
    async def normal_conversation(message: Message) -> None:
        await fallback(message)

    dispatcher.include_router(fallback_router)
    session = AsyncMock(spec=BaseSession)
    bot = Bot("123456:test-token", session=cast(BaseSession, session))
    chat = Chat(id=-100, type="supergroup")

    async def scenario() -> None:
        posts = [
            _message(chat, text="Посмотри на мои арты @test_bot"),
            _message(
                chat,
                video=Video(
                    file_id="video", file_unique_id="v", width=40, height=40, duration=1
                ),
            ),
            _message(
                chat,
                animation=Animation(
                    file_id="gif", file_unique_id="g", width=40, height=40, duration=1
                ),
                caption="Привет @test_bot, посмотри на GIF",
            ),
            _message(
                chat,
                sticker=Sticker(
                    file_id="sticker",
                    file_unique_id="s",
                    type="regular",
                    width=40,
                    height=40,
                    is_animated=False,
                    is_video=False,
                ),
            ),
            _message(
                chat, voice=Voice(file_id="voice", file_unique_id="vo", duration=1)
            ),
            _message(
                chat,
                document=Document(
                    file_id="pdf", file_unique_id="p", mime_type="application/pdf"
                ),
            ),
            _message(
                chat,
                document=Document(
                    file_id="webm", file_unique_id="w", mime_type="video/webm"
                ),
            ),
            _message(
                chat,
                photo=[
                    PhotoSize(
                        file_id="unauthorized", file_unique_id="u", width=40, height=40
                    )
                ],
                from_user=User(id=2, is_bot=False, first_name="Other"),
            ),
            _message(
                chat,
                photo=[
                    PhotoSize(file_id="photo", file_unique_id="ph", width=40, height=40)
                ],
                media_group_id="album",
            ),
            _message(
                chat,
                document=Document(
                    file_id="png", file_unique_id="pn", mime_type="image/png"
                ),
                media_group_id="album",
            ),
        ]
        for index, message in enumerate(posts):
            await dispatcher.feed_update(bot, Update(update_id=index, message=message))
        assert images.get_all() == ["photo", "png"]
        assert images.get_kind("png") == "document"
        downloader.download.assert_not_awaited()
        engine.respond_and_deliver.assert_not_awaited()
        fallback.assert_not_awaited()
        session.assert_not_awaited()

        command = _message(
            chat,
            text="/artchat list",
            entities=[MessageEntity(type="bot_command", offset=0, length=8)],
        )
        await dispatcher.feed_update(bot, Update(update_id=11, message=command))
        session.assert_awaited_once()
        call = session.await_args
        assert call is not None
        assert "В коллекции: 2" in call.args[1].text
        session.reset_mock()
        command = _message(
            chat,
            text="/randomart",
            entities=[MessageEntity(type="bot_command", offset=0, length=10)],
        )
        await dispatcher.feed_update(bot, Update(update_id=12, message=command))
        session.assert_awaited_once()
        engine.respond_and_deliver.assert_not_awaited()
        sources.change(-100, add=False)
        await dispatcher.feed_update(
            bot, Update(update_id=13, message=_message(chat, text="Обычный разговор"))
        )
        fallback.assert_awaited_once()
        assert images.count() == 2
        await dispatcher.storage.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "chat", [Chat(id=42, type="private"), Chat(id=-200, type="supergroup")]
)
def test_other_chats_still_reach_media_analysis(tmp_path: Path, chat: Chat) -> None:
    dispatcher = Dispatcher()
    dispatcher.include_router(
        create_art_router(ImagesRepository(tmp_path), -100, frozenset({1}))
    )
    engine = AsyncMock(spec=ResponseEngine)
    downloader = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: BytesIO) -> BytesIO:
        destination.write(b"\xff\xd8\xfftest")
        return destination

    downloader.download.side_effect = download
    dispatcher.include_router(
        create_media_router(cast(ResponseEngine, engine), cast(Bot, downloader))
    )
    session = AsyncMock(spec=BaseSession)
    bot = Bot("123456:test-token", session=cast(BaseSession, session))

    async def scenario() -> None:
        message = _message(
            chat,
            photo=[PhotoSize(file_id="photo", file_unique_id="p", width=40, height=40)],
        )
        await dispatcher.feed_update(bot, Update(update_id=1, message=message))
        downloader.download.assert_awaited_once()
        engine.respond_and_deliver.assert_awaited_once()
        await dispatcher.storage.close()

    asyncio.run(scenario())
