"""Тесты обработчиков /start и системы артов."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

import protogen_delta.handlers.art as art_module
import protogen_delta.handlers.start as start_module
from protogen_delta.handlers.art import create_art_router
from protogen_delta.handlers.start import FIRST_START_MESSAGE, create_start_router
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.users import UsersRepository


async def _call_handler(
    router: Router,
    index: int,
    message: Message,
) -> None:
    """Вызвать зарегистрированный message-handler по индексу."""
    handler = router.message.handlers[index]

    await handler.callback(message)


def _create_message_mock() -> tuple[
    Message,
    Mock,
    AsyncMock,
    AsyncMock,
]:
    """Создать Telegram Message с нужными моками методов."""
    message_mock = Mock()

    answer_mock = AsyncMock()
    answer_photo_mock = AsyncMock()

    message_mock.answer = answer_mock
    message_mock.answer_photo = answer_photo_mock

    return (
        cast(Message, message_mock),
        message_mock,
        answer_mock,
        answer_photo_mock,
    )


def test_start_registers_new_user() -> None:
    """Новый пользователь должен сохраниться и получить первое приветствие."""
    users_mock = Mock(spec=UsersRepository)
    users_mock.add.return_value = True

    router = create_start_router(
        users_repository=cast(UsersRepository, users_mock),
        start_messages=["Повторное приветствие"],
    )

    message, raw_message, answer_mock, _ = _create_message_mock()

    raw_message.from_user = SimpleNamespace(id=123)

    asyncio.run(
        _call_handler(
            router,
            0,
            message,
        )
    )

    users_mock.add.assert_called_once_with(123)
    answer_mock.assert_awaited_once_with(FIRST_START_MESSAGE)


def test_start_returns_random_message_for_existing_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Повторный /start должен возвращать одно из обычных приветствий."""
    users_mock = Mock(spec=UsersRepository)
    users_mock.add.return_value = False

    start_messages = [
        "Первое",
        "Второе",
    ]

    monkeypatch.setattr(
        start_module.random,
        "choice",
        lambda values: values[0],
    )

    router = create_start_router(
        users_repository=cast(UsersRepository, users_mock),
        start_messages=start_messages,
    )

    message, raw_message, answer_mock, _ = _create_message_mock()

    raw_message.from_user = SimpleNamespace(id=123)

    asyncio.run(
        _call_handler(
            router,
            0,
            message,
        )
    )

    users_mock.add.assert_called_once_with(123)
    answer_mock.assert_awaited_once_with("Первое")


def test_start_uses_fallback_without_start_messages() -> None:
    """При пустом списке приветствий должен использоваться fallback."""
    users_mock = Mock(spec=UsersRepository)
    users_mock.add.return_value = False

    router = create_start_router(
        users_repository=cast(UsersRepository, users_mock),
        start_messages=[],
    )

    message, raw_message, answer_mock, _ = _create_message_mock()

    raw_message.from_user = SimpleNamespace(id=123)

    asyncio.run(
        _call_handler(
            router,
            0,
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "Я уже запущен :D",
    )


def test_start_ignores_message_without_user() -> None:
    """Сообщение без from_user не должно регистрироваться."""
    users_mock = Mock(spec=UsersRepository)

    router = create_start_router(
        users_repository=cast(UsersRepository, users_mock),
        start_messages=[],
    )

    message, raw_message, answer_mock, _ = _create_message_mock()

    raw_message.from_user = None

    asyncio.run(
        _call_handler(
            router,
            0,
            message,
        )
    )

    users_mock.add.assert_not_called()
    answer_mock.assert_not_awaited()


def test_art_handler_saves_photo_from_allowed_chat() -> None:
    """Фото из разрешённой группы должно добавляться в базу."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.add.return_value = True

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, raw_message, _, _ = _create_message_mock()

    raw_message.chat = SimpleNamespace(id=-100123)
    raw_message.photo = [
        SimpleNamespace(file_id="small-file-id"),
        SimpleNamespace(file_id="large-file-id"),
    ]

    asyncio.run(
        _call_handler(
            router,
            0,
            message,
        )
    )

    images_mock.add.assert_called_once_with(
        "large-file-id",
    )


def test_art_handler_ignores_photo_from_other_chat() -> None:
    """Фото из другого чата не должно попадать в базу."""
    images_mock = Mock(spec=ImagesRepository)

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, raw_message, _, _ = _create_message_mock()

    raw_message.chat = SimpleNamespace(id=-100999)
    raw_message.photo = [
        SimpleNamespace(file_id="file-id"),
    ]

    asyncio.run(
        _call_handler(
            router,
            0,
            message,
        )
    )

    images_mock.add.assert_not_called()


def test_art_handler_ignores_empty_photo() -> None:
    """Пустой список фото не должен сохраняться."""
    images_mock = Mock(spec=ImagesRepository)

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, raw_message, _, _ = _create_message_mock()

    raw_message.chat = SimpleNamespace(id=-100123)
    raw_message.photo = []

    asyncio.run(
        _call_handler(
            router,
            0,
            message,
        )
    )

    images_mock.add.assert_not_called()


def test_art_handler_saves_image_document() -> None:
    """Документ-изображение из разрешённой группы должен сохраняться."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.add.return_value = True

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, raw_message, _, _ = _create_message_mock()

    raw_message.chat = SimpleNamespace(id=-100123)
    raw_message.document = SimpleNamespace(
        file_id="document-file-id",
        mime_type="image/png",
    )

    asyncio.run(
        _call_handler(
            router,
            1,
            message,
        )
    )

    images_mock.add.assert_called_once_with(
        "document-file-id",
    )


def test_art_handler_ignores_non_image_document() -> None:
    """Обычный документ не должен добавляться как арт."""
    images_mock = Mock(spec=ImagesRepository)

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, raw_message, _, _ = _create_message_mock()

    raw_message.chat = SimpleNamespace(id=-100123)
    raw_message.document = SimpleNamespace(
        file_id="document-file-id",
        mime_type="application/pdf",
    )

    asyncio.run(
        _call_handler(
            router,
            1,
            message,
        )
    )

    images_mock.add.assert_not_called()


def test_art_handler_ignores_document_from_other_chat() -> None:
    """Документ из другого чата не должен сохраняться."""
    images_mock = Mock(spec=ImagesRepository)

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, raw_message, _, _ = _create_message_mock()

    raw_message.chat = SimpleNamespace(id=-100999)
    raw_message.document = SimpleNamespace(
        file_id="document-file-id",
        mime_type="image/png",
    )

    asyncio.run(
        _call_handler(
            router,
            1,
            message,
        )
    )

    images_mock.add.assert_not_called()


def test_random_art_reports_empty_database() -> None:
    """При пустой базе /randomart должен сообщить об этом."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = []

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock()

    asyncio.run(
        _call_handler(
            router,
            2,
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "База пустая 😢 сначала добавь арты.",
    )
    answer_photo_mock.assert_not_awaited()


def test_random_art_sends_selected_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Команда /randomart должна отправлять случайный сохранённый арт."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = [
        "file-id-1",
        "file-id-2",
    ]

    monkeypatch.setattr(
        art_module.random,
        "choice",
        lambda values: values[0],
    )

    router = create_art_router(
        images_repository=cast(ImagesRepository, images_mock),
        art_chat_id=-100123,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock()

    asyncio.run(
        _call_handler(
            router,
            2,
            message,
        )
    )

    answer_mock.assert_not_awaited()

    answer_photo_mock.assert_awaited_once_with(
        "file-id-1",
        caption="🎨 Лови артик!",
    )


def test_random_art_handles_telegram_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка отправки арта должна давать пользователю понятный ответ."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = [
        "file-id-1",
    ]

    monkeypatch.setattr(
        art_module.random,
        "choice",
        lambda values: values[0],
    )

    router = create_art_router(
        images_repository=cast(
            ImagesRepository,
            images_mock,
        ),
        art_chat_id=-100123,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock()

    answer_photo_mock.side_effect = TelegramAPIError(
        method=Mock(),
        message="Telegram internal error",
    )

    asyncio.run(
        _call_handler(
            router,
            2,
            message,
        )
    )

    answer_photo_mock.assert_awaited_once_with(
        "file-id-1",
        caption="🎨 Лови артик!",
    )

    answer_mock.assert_awaited_once_with("Не смог отправить арт 😢 попробуй ещё раз.")
