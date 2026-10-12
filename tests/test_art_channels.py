"""Настоящие channel_post проходят через Dispatcher без автора и без ответа."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Document, Message, PhotoSize, Update, User

from protogen_delta.handlers.art import create_art_router
from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.images import ImagesRepository


def _photo(chat: Chat, file_id: str, **kwargs: object) -> Message:
    return Message.model_validate(
        {
            "message_id": 1,
            "date": datetime.now(timezone.utc),
            "chat": chat,
            "photo": [
                PhotoSize(
                    file_id="small", file_unique_id="small", width=100, height=100
                ),
                PhotoSize(
                    file_id=file_id,
                    file_unique_id=file_id,
                    width=1000,
                    height=1000,
                ),
            ],
            **kwargs,
        }
    )


def test_registered_channel_albums_persist_silently_and_survive_disconnect(
    tmp_path: Path,
) -> None:
    images = ImagesRepository(tmp_path)
    sources = ArtSourcesRepository(tmp_path, -100)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_art_router(images, -100, frozenset({1}), sources))
    assert "channel_post" in dispatcher.resolve_used_update_types()
    session = AsyncMock(spec=BaseSession)
    bot = Bot("123456:test-token", session=cast(BaseSession, session))
    channel = Chat(id=-100, type="channel", title="Art collection")

    async def scenario() -> None:
        photos = [
            _photo(channel, "photo-1", sender_chat=channel, media_group_id="album"),
            _photo(channel, "photo-2", sender_chat=channel, media_group_id="album"),
        ]
        document = Message(
            message_id=3,
            date=datetime.now(timezone.utc),
            chat=channel,
            sender_chat=channel,
            document=Document(
                file_id="original-png", file_unique_id="original", mime_type="image/png"
            ),
        )
        assert all(message.from_user is None for message in photos + [document])
        for index, message in enumerate(photos + [document, photos[0]]):
            await dispatcher.feed_update(
                bot, Update(update_id=index, channel_post=message)
            )
        assert images.get_all() == ["photo-1", "photo-2", "original-png"]
        restored = ImagesRepository(tmp_path)
        assert restored.count() == 3
        assert restored.get_kind("original-png") == "document"
        assert restored.get_kind("photo-1") == "photo"
        sources.change(-100, add=False)
        await dispatcher.feed_update(
            bot,
            Update(update_id=5, channel_post=_photo(channel, "after-disconnect")),
        )
        assert images.get_all() == ["photo-1", "photo-2", "original-png"]
        session.assert_not_awaited()
        await dispatcher.storage.close()

    asyncio.run(scenario())


def test_unknown_channel_and_non_image_posts_are_ignored(tmp_path: Path) -> None:
    images = ImagesRepository(tmp_path)
    sources = ArtSourcesRepository(tmp_path, -100)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_art_router(images, -100, frozenset({1}), sources))
    session = AsyncMock(spec=BaseSession)
    bot = Bot("123456:test-token", session=cast(BaseSession, session))
    channel = Chat(id=-100, type="channel")

    async def scenario() -> None:
        posts = [
            _photo(Chat(id=-200, type="channel"), "unregistered"),
            Message(
                message_id=2,
                date=datetime.now(timezone.utc),
                chat=channel,
                text="/artchat add -200",
            ),
            Message(
                message_id=3,
                date=datetime.now(timezone.utc),
                chat=channel,
                document=Document(
                    file_id="video", file_unique_id="video", mime_type="video/mp4"
                ),
            ),
        ]
        for index, message in enumerate(posts):
            await dispatcher.feed_update(
                bot, Update(update_id=index, channel_post=message)
            )
        assert images.count() == 0
        assert sources.get_all() == [-100]
        session.assert_not_awaited()
        await dispatcher.storage.close()

    asyncio.run(scenario())


def test_group_ingestion_still_requires_application_admin(tmp_path: Path) -> None:
    images = ImagesRepository(tmp_path)
    sources = ArtSourcesRepository(tmp_path, -100)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_art_router(images, -100, frozenset({1}), sources))
    session = AsyncMock(spec=BaseSession)
    bot = Bot("123456:test-token", session=cast(BaseSession, session))
    group = Chat(id=-100, type="supergroup")

    async def scenario() -> None:
        messages = [
            _photo(
                group, "non-admin", from_user=User(id=2, is_bot=False, first_name="B")
            ),
            _photo(group, "anonymous", sender_chat=group),
            _photo(group, "admin", from_user=User(id=1, is_bot=False, first_name="A")),
        ]
        for index, message in enumerate(messages):
            await dispatcher.feed_update(bot, Update(update_id=index, message=message))
        assert images.get_all() == ["admin"]
        session.assert_not_awaited()
        await dispatcher.storage.close()

    asyncio.run(scenario())


def test_configured_channel_works_without_dynamic_sources(tmp_path: Path) -> None:
    images = ImagesRepository(tmp_path)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_art_router(images, -100))
    session = AsyncMock(spec=BaseSession)
    bot = Bot("123456:test-token", session=cast(BaseSession, session))

    async def scenario() -> None:
        await dispatcher.feed_update(
            bot,
            Update(
                update_id=1,
                channel_post=_photo(Chat(id=-100, type="channel"), "configured"),
            ),
        )
        assert images.get_all() == ["configured"]
        session.assert_not_awaited()
        await dispatcher.storage.close()

    asyncio.run(scenario())
