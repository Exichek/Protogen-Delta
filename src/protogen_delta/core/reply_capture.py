"""Собрать подготовленные части одного хода до обращения к модели."""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

ReplyCapture = Callable[
    [int, str, Callable[[str], Awaitable[None]], dict[str, Any]], None
]
current_reply_capture: ContextVar[ReplyCapture | None] = ContextVar(
    "reply_capture", default=None
)
