"""Передать в историю только подтверждённую реакцию текущего ответа."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

reaction_receipt: ContextVar[str | None] = ContextVar("reaction_receipt", default=None)


@contextmanager
def reaction_receipt_scope() -> Iterator[None]:
    token = reaction_receipt.set(None)
    try:
        yield
    finally:
        reaction_receipt.reset(token)


def with_reaction_receipt(reply: str) -> str:
    """Оставить видимый ответ прежним, дополнив его историю фактом действия."""
    emoji = reaction_receipt.get()
    if emoji is None:
        return reply
    return (
        reply
        + "\n\n[Подтверждённое действие Дельты: поставил реакцию "
        + emoji
        + " на сообщение пользователя этого хода. Это моя отправленная реакция.]"
    )
