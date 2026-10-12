"""Тесты контекста корреляции логов."""

import logging

from protogen_delta.core.log_context import (
    LogContextFilter,
    bind_log_context,
)
from protogen_delta.core.logging_config import LOG_FORMAT


def _create_record() -> logging.LogRecord:
    """Создать минимальную запись лога."""
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test",
        args=(),
        exc_info=None,
    )


def test_log_context_filter_uses_defaults() -> None:
    """Вне пользовательского запроса идентификаторы должны быть пустыми."""
    record = _create_record()

    result = LogContextFilter().filter(record)

    assert result is True
    assert getattr(record, "user_id") == "-"
    assert getattr(record, "request_id") == "-"


def test_log_context_filter_uses_bound_context() -> None:
    """Фильтр должен добавлять текущие user_id и request_id."""
    with bind_log_context(
        user_id=123456,
        request_id="request-123",
    ):
        record = _create_record()

        LogContextFilter().filter(record)

    assert getattr(record, "user_id") == "123456"
    assert getattr(record, "request_id") == "request-123"


def test_log_context_is_restored_after_request() -> None:
    """Контекст одного запроса не должен протекать в следующий."""
    with bind_log_context(
        user_id=123456,
        request_id="request-123",
    ):
        pass

    record = _create_record()

    LogContextFilter().filter(record)

    assert getattr(record, "user_id") == "-"
    assert getattr(record, "request_id") == "-"


def test_log_format_includes_bound_context() -> None:
    """Форматированный лог должен содержать user_id и request_id."""
    with bind_log_context(
        user_id=123456,
        request_id="request-123",
    ):
        record = _create_record()

        LogContextFilter().filter(record)

        result = logging.Formatter(LOG_FORMAT).format(record)

    assert "user_id=123456" in result
    assert "request_id=request-123" in result
    assert "test: test" in result


def test_log_format_uses_defaults_outside_request() -> None:
    """Системные логи вне запроса должны безопасно использовать заглушки."""
    record = _create_record()

    LogContextFilter().filter(record)

    result = logging.Formatter(LOG_FORMAT).format(record)

    assert "user_id=-" in result
    assert "request_id=-" in result
