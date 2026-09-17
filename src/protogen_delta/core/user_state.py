"""Состояние отдельных пользователей во время работы приложения."""

import asyncio
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Хранить один завершённый ход диалога."""

    user_message: str
    assistant_message: str


@dataclass(slots=True)
class UserState:
    """Хранить изменяемое состояние одного пользователя."""

    mood: str = "playful"
    reply_count: int = 0
    history: deque[ConversationTurn] = field(
        default_factory=deque,
    )
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
        compare=False,
    )

    def register_reply(self) -> None:
        """Увеличить счётчик ответов этому пользователю."""
        self.reply_count += 1


class UserStateStore:
    """Хранить runtime-состояние пользователей текущего процесса."""

    def __init__(
        self,
        history_limit: int = 8,
    ) -> None:
        """Создать пустое хранилище пользовательских состояний."""
        if history_limit <= 0:
            raise ValueError("history_limit должен быть больше нуля")

        self._states: dict[int, UserState] = {}
        self._history_limit = history_limit

    def get(self, user_id: int) -> UserState:
        """Получить состояние пользователя или создать новое."""
        state = self._states.get(user_id)

        if state is None:
            state = UserState(
                history=deque(
                    maxlen=self._history_limit,
                )
            )
            self._states[user_id] = state

        return state

    def remove(self, user_id: int) -> bool:
        """Удалить runtime-состояние пользователя."""
        return self._states.pop(user_id, None) is not None

    @property
    def tracked_users_count(self) -> int:
        """Вернуть количество состояний в памяти."""
        return len(self._states)
