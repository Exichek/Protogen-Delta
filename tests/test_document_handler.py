"""Тесты Telegram-обработчика документов."""

import asyncio
import io
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.types import Message

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.documents import (
    DOCUMENT_DOWNLOAD_ERROR_REPLY,
    DOCUMENT_READ_ERROR_REPLY,
    DOCUMENT_TOO_LARGE_REPLY,
    UNSUPPORTED_DOCUMENT_REPLY,
    create_document_router,
)
from protogen_delta.handlers.text import BUSY_REPLY, RATE_LIMIT_REPLY
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.documents import (
    MAX_DOCUMENT_BYTES,
    ExtractedDocument,
    ExtractedImage,
)
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine

TEST_USER_ID = 123456


def _message(document: Any, caption: str | None = None) -> tuple[Message, AsyncMock]:
    """Создать сообщение с документом и методом ответа."""
    message = Mock(spec=Message)
    message.document = document
    message.caption = caption
    message.from_user = SimpleNamespace(id=TEST_USER_ID)
    message.chat = SimpleNamespace(id=777, type="private")
    message.answer = AsyncMock()
    return cast(Message, message), message.answer


def _router(data: bytes) -> tuple[Any, AsyncMock, AsyncMock]:
    """Создать роутер с заданным содержимым скачиваемого файла."""
    engine = AsyncMock(spec=ResponseEngine)

    async def respond(
        user_id: int,
        text: str,
        deliver: Any,
        **kwargs: Any,
    ) -> None:
        await deliver("Разобрал документ.")

    engine.respond_and_deliver.side_effect = respond
    bot = AsyncMock(spec=Bot)

    async def download(file_id: str, *, destination: io.BytesIO) -> io.BytesIO:
        destination.write(data)
        return destination

    bot.download.side_effect = download
    return (
        create_document_router(
            cast(ResponseEngine, engine),
            cast(Bot, bot),
            native_work=BlockingWorkPool(2),
        ),
        engine,
        bot,
    )


def test_document_caption_and_content_are_kept_separate() -> None:
    """История должна получать вопрос, а модель — отдельный текст вложения."""
    router, engine, _ = _router("Содержимое отчёта".encode())
    document = SimpleNamespace(
        file_id="doc-id",
        file_size=100,
        file_name="report.txt",
        mime_type="text/plain",
    )
    message, answer = _message(document, "Найди главную мысль")

    asyncio.run(router.message.handlers[0].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert call.args[1] == "Найди главную мысль"
    assert call.kwargs["attachment_text"] == "Содержимое отчёта"
    assert call.kwargs["attachment_name"] == "report.txt (TXT)"
    answer.assert_awaited_once_with("Разобрал документ.")


def test_document_without_caption_gets_natural_default_request() -> None:
    """Документ без подписи должен получать просьбу о кратком содержании."""
    router, engine, _ = _router(b"Hello")
    document = SimpleNamespace(
        file_id="doc-id",
        file_size=5,
        file_name="../../notes.md",
        mime_type="text/markdown",
    )
    message, _ = _message(document)

    asyncio.run(router.message.handlers[0].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert '"notes.md"' in call.args[1]
    assert "../" not in call.args[1]


def test_large_document_is_rejected_before_download() -> None:
    """Заведомо большой документ не должен скачиваться."""
    router, engine, bot = _router(b"data")
    document = SimpleNamespace(
        file_id="large-id",
        file_size=MAX_DOCUMENT_BYTES + 1,
        file_name="large.txt",
        mime_type="text/plain",
    )
    message, answer = _message(document)

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_awaited_once_with(DOCUMENT_TOO_LARGE_REPLY)
    bot.download.assert_not_awaited()
    engine.respond_and_deliver.assert_not_awaited()


def test_unsupported_and_broken_documents_get_clear_replies() -> None:
    """Неподдерживаемый и повреждённый форматы должны различаться."""
    unsupported_router, _, _ = _router(b"archive")
    unsupported = SimpleNamespace(
        file_id="zip-id",
        file_size=7,
        file_name="archive.zip",
        mime_type="application/zip",
    )
    unsupported_message, unsupported_answer = _message(unsupported)
    asyncio.run(unsupported_router.message.handlers[0].callback(unsupported_message))
    unsupported_answer.assert_awaited_once_with(UNSUPPORTED_DOCUMENT_REPLY)

    broken_router, _, _ = _router(b"broken")
    broken = SimpleNamespace(
        file_id="pdf-id",
        file_size=6,
        file_name="broken.pdf",
        mime_type="application/pdf",
    )
    broken_message, broken_answer = _message(broken)
    asyncio.run(broken_router.message.handlers[0].callback(broken_message))
    broken_answer.assert_awaited_once_with(DOCUMENT_READ_ERROR_REPLY)


def test_download_and_busy_errors_are_handled() -> None:
    """Сетевой сбой и занятый движок не должны падать наружу."""
    router, engine, bot = _router(b"text")
    document = SimpleNamespace(
        file_id="doc-id",
        file_size=4,
        file_name="note.txt",
        mime_type="text/plain",
    )
    message, answer = _message(document)
    bot.download.side_effect = OSError("network")
    asyncio.run(router.message.handlers[0].callback(message))
    answer.assert_awaited_once_with(DOCUMENT_DOWNLOAD_ERROR_REPLY)

    router, engine, _ = _router(b"text")
    engine.respond_and_deliver.side_effect = ResponseBusyError
    message, answer = _message(document)
    asyncio.run(router.message.handlers[0].callback(message))
    answer.assert_awaited_once_with(BUSY_REPLY)


def test_missing_sender_is_ignored() -> None:
    """Документ без подтверждённого отправителя должен игнорироваться."""
    router, engine, bot = _router(b"text")
    document = SimpleNamespace(
        file_id="doc-id",
        file_size=4,
        file_name="note.txt",
        mime_type="text/plain",
    )
    message, answer = _message(document)
    message.from_user = None
    asyncio.run(router.message.handlers[0].callback(message))
    answer.assert_not_awaited()
    bot.download.assert_not_awaited()
    engine.respond_and_deliver.assert_not_awaited()


def test_document_rate_limit_is_checked_before_download() -> None:
    """Повторный документ пользователя не должен скачиваться во время cooldown."""
    engine = AsyncMock(spec=ResponseEngine)
    bot = AsyncMock(spec=Bot)
    limiter = UserRateLimiter(cooldown_seconds=2.0, clock=lambda: 100.0)
    assert limiter.allow(TEST_USER_ID)
    router = create_document_router(
        cast(ResponseEngine, engine),
        cast(Bot, bot),
        rate_limiter=limiter,
    )
    document = SimpleNamespace(
        file_id="doc-id",
        file_size=4,
        file_name="note.txt",
        mime_type="text/plain",
    )
    message, answer = _message(document)

    asyncio.run(router.message.handlers[0].callback(message))

    answer.assert_awaited_once_with(RATE_LIMIT_REPLY)
    bot.download.assert_not_awaited()


def test_truncated_document_is_marked_for_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Модель должна видеть явную отметку о частичном документе."""
    router, engine, _ = _router(b"text")
    monkeypatch.setattr(
        "protogen_delta.handlers.documents.extract_document",
        lambda *args: ExtractedDocument("часть", "TXT", truncated=True),
    )
    document = SimpleNamespace(
        file_id="doc-id",
        file_size=4,
        file_name="long.txt",
        mime_type="text/plain",
    )
    message, _ = _message(document)

    asyncio.run(router.message.handlers[0].callback(message))

    call = engine.respond_and_deliver.await_args
    assert call is not None
    assert "обработан частично" in call.kwargs["attachment_text"]


def test_scanned_pdf_images_are_passed_to_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Извлечённая страница скана должна попасть в общий vision-запрос."""
    router, engine, _ = _router(b"pdf")
    monkeypatch.setattr(
        "protogen_delta.handlers.documents.extract_document",
        lambda *args: ExtractedDocument(
            "Прочитай скан",
            "PDF (скан)",
            images=(ExtractedImage(b"jpeg", "image/jpeg", "скан страницы 1 PDF"),),
        ),
    )
    document = SimpleNamespace(
        file_id="pdf-id",
        file_size=4,
        file_name="scan.pdf",
        mime_type="application/pdf",
    )
    message, _ = _message(document, "Что написано?")

    asyncio.run(router.message.handlers[0].callback(message))

    call_args = engine.respond_and_deliver.await_args
    assert call_args is not None
    assert call_args.kwargs["images"][0].data == b"jpeg"
    assert call_args.kwargs["images"][0].label == "скан страницы 1 PDF"
