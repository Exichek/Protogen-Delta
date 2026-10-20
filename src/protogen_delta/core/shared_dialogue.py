"""Явно подтверждённые группы памяти без объединения прав и настроек."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar


class SharedDialogueAccounts:
    def __init__(self, groups: tuple[frozenset[int], ...] = ()) -> None:
        self.groups = groups
        self._members = {owner: group for group in groups for owner in group}
        self._locks = {group: asyncio.Lock() for group in groups}
        self._held: ContextVar[frozenset[frozenset[int]]] = ContextVar(
            "shared_dialogue_locks", default=frozenset()
        )

    def members(self, owner: int) -> frozenset[int]:
        return self._members.get(owner, frozenset({owner}))

    def key(self, owner: int) -> int:
        return min(self.members(owner))

    @asynccontextmanager
    async def coordinate(self, owner: int) -> AsyncIterator[None]:
        group = self._members.get(owner)
        held = self._held.get()
        if group is None or group in held:
            yield
            return
        async with self._locks[group]:
            token = self._held.set(held | {group})
            try:
                yield
            finally:
                self._held.reset(token)
