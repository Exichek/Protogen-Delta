"""Привязка расходов модели к пользователю без хранения содержимого сообщений."""

from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, TypeVar

current_llm_user: ContextVar[int | None] = ContextVar("current_llm_user", default=None)
current_llm_chat: ContextVar[int | None] = ContextVar("current_llm_chat", default=None)
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


class LlmGlobalLimitError(LlmBudgetError):
    def __init__(self) -> None:
        super().__init__(
            "Общий дневной лимит запросов бота к модели исчерпан. "
            "Он обновится в 00:00 по Москве. Профиль и память сохранены."
        )


@contextmanager
def llm_user(user_id: int | None, chat_id: int | None = None) -> Iterator[None]:
    token = current_llm_user.set(user_id)
    chat_token = current_llm_chat.set(chat_id)
    try:
        yield
    finally:
        current_llm_user.reset(token)
        current_llm_chat.reset(chat_token)


def user_scoped(
    method: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(method)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        user_id = kwargs.get("user_id", args[1] if len(args) > 1 else None)
        if not isinstance(user_id, int) or isinstance(user_id, bool):
            raise ValueError("Нужен Telegram ID владельца запроса")
        chat_id = kwargs.get("chat_id")
        with llm_user(user_id, chat_id if type(chat_id) is int else None):
            return await method(*args, **kwargs)

    return wrapped
