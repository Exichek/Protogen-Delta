"""Выбор контекста лички или отдельного группового чата."""

from typing import TypedDict


class ChatScopeOptions(TypedDict, total=False):
    chat_id: int


def chat_scope_options(chat_id: int, chat_type: str) -> ChatScopeOptions:
    """Личные вызовы сохраняют прежний API; группы получают явный ключ."""
    if chat_type in {"group", "supergroup", "channel"}:
        return {"chat_id": chat_id}
    return {}
