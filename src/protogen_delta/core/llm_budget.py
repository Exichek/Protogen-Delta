"""Привязка расходов модели к пользователю без хранения содержимого сообщений."""

from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, TypeVar

current_llm_user: ContextVar[int | None] = ContextVar("current_llm_user", default=None)
P = ParamSpec("P")
T = TypeVar("T")


class LlmBudgetError(RuntimeError):
    """Безопасное для пользователя сообщение об отказе до обращения к модели."""


class LlmDailyLimitError(LlmBudgetError):
    def __init__(self) -> None:
        super().__init__(
            "Дневной лимит запросов к модели исчерпан. Он обновится в 00:00 по Москве. Профиль и память сохранены."
        )


class LlmUsageUnavailableError(LlmBudgetError):
    def __init__(self) -> None:
        super().__init__("Учёт запросов к модели сейчас недоступен. Попробуй позже.")


@contextmanager
def llm_user(user_id: int | None) -> Iterator[None]:
    token = current_llm_user.set(user_id)
    try:
        yield
    finally:
        current_llm_user.reset(token)


def user_scoped(
    method: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(method)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        user_id = kwargs.get("user_id", args[1] if len(args) > 1 else None)
        if not isinstance(user_id, int) or isinstance(user_id, bool):
            raise ValueError("Нужен Telegram ID владельца запроса")
        with llm_user(user_id):
            return await method(*args, **kwargs)

    return wrapped
