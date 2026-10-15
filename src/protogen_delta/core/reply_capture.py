"""Собрать подготовленные части одного хода до обращения к модели."""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

ReplyCapture = Callable[
    [int, str, Callable[[str], Awaitable[None]], dict[str, Any]], None
]
current_reply_capture: ContextVar[ReplyCapture | None] = ContextVar(
    "reply_capture", default=None
)


@dataclass(frozen=True, slots=True)
class ReplySource:
    """Отделить исходную реплику от оформления пачки и чужих пересылок."""

    text: str
    model_text: str
    allow_controls: bool
    allow_facts: bool


current_reply_sources: ContextVar[tuple[ReplySource, ...] | None] = ContextVar(
    "reply_sources", default=None
)
