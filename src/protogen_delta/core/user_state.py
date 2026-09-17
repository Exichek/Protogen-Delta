"""Состояние отдельных пользователей во время работы приложения."""

from dataclasses import dataclass


@dataclass(slots=True)
class UserState:
    """Хранить изменяемое состояние одного пользователя."""

    mood: str = "playful"
    reply_count: int = 0

    def register_reply(self) -> None:
        """Увеличить счётчик ответов этому пользователю."""
        self.reply_count += 1


class UserStateStore:
    """Хранить runtime-состояние пользователей текущего процесса."""

    def __init__(self) -> None:
        """Создать пустое хранилище пользовательских состояний."""
        self._states: dict[int, UserState] = {}

    def get(self, user_id: int) -> UserState:
        """Получить состояние пользователя или создать новое."""
        state = self._states.get(user_id)

        if state is None:
            state = UserState()
            self._states[user_id] = state

        return state

    def remove(self, user_id: int) -> bool:
        """Удалить runtime-состояние пользователя."""
        return self._states.pop(user_id, None) is not None

    @property
    def tracked_users_count(self) -> int:
        """Вернуть количество состояний в памяти."""
        return len(self._states)
