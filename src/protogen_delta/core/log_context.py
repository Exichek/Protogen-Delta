"""Контекст корреляции логов пользовательского запроса."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

_user_id_context: ContextVar[str] = ContextVar(
    "log_user_id",
    default="-",
)
_request_id_context: ContextVar[str] = ContextVar(
    "log_request_id",
    default="-",
)


class LogContextFilter(logging.Filter):
    """Добавлять пользовательский контекст в записи логов."""

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        """Добавить user_id и request_id в запись."""
        setattr(
            record,
            "user_id",
            _user_id_context.get(),
        )
        setattr(
            record,
            "request_id",
            _request_id_context.get(),
        )

        return True


@contextmanager
def bind_log_context(
    *,
    user_id: int,
    request_id: str | None = None,
) -> Iterator[str]:
    """Привязать контекст логов к обработке одного запроса."""
    current_request_id = request_id or uuid4().hex[:12]

    user_token = _user_id_context.set(
        str(user_id),
    )
    request_token = _request_id_context.set(
        current_request_id,
    )

    try:
        yield current_request_id
    finally:
        _request_id_context.reset(request_token)
        _user_id_context.reset(user_token)
