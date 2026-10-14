"""Признак подтверждённого сбоя приложения, независимо от текста модели."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

reply_error: ContextVar[bool] = ContextVar("reply_error", default=False)


@contextmanager
def error_reply(enabled: bool) -> Iterator[None]:
    token = reply_error.set(enabled)
    try:
        yield
    finally:
        reply_error.reset(token)
