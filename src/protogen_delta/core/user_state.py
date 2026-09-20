"""Состояние отдельных пользователей во время работы приложения."""

import asyncio
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Protocol


def _clamp_unit(value: float) -> float:
    """Ограничить числовое состояние диапазоном от 0.0 до 1.0."""
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Хранить один завершённый ход диалога."""

    user_message: str
    assistant_message: str


@dataclass(slots=True)
class EmotionalState:
    """Хранить накопленное эмоциональное состояние Дельты."""

    warmth: float = 0.0
    irritation: float = 0.0
    playfulness: float = 0.0
    arousal: float = 0.0

    def adjust(
        self,
        *,
        warmth: float = 0.0,
        irritation: float = 0.0,
        playfulness: float = 0.0,
        arousal: float = 0.0,
    ) -> None:
        """Изменить эмоциональные показатели с ограничением диапазона."""
        self.warmth = _clamp_unit(self.warmth + warmth)
        self.irritation = _clamp_unit(self.irritation + irritation)
        self.playfulness = _clamp_unit(
            self.playfulness + playfulness,
        )
        self.arousal = _clamp_unit(self.arousal + arousal)


@dataclass(slots=True)
class RelationshipState:
    """Хранить накопленное отношение Дельты к пользователю."""

    familiarity: float = 0.0
    trust: float = 0.0
    affection: float = 0.0
    resentment: float = 0.0

    def adjust(
        self,
        *,
        familiarity: float = 0.0,
        trust: float = 0.0,
        affection: float = 0.0,
        resentment: float = 0.0,
    ) -> None:
        """Изменить показатели отношений с ограничением диапазона."""
        self.familiarity = _clamp_unit(
            self.familiarity + familiarity,
        )
        self.trust = _clamp_unit(self.trust + trust)
        self.affection = _clamp_unit(
            self.affection + affection,
        )
        self.resentment = _clamp_unit(
            self.resentment + resentment,
        )


@dataclass(frozen=True, slots=True)
class PersistentUserState:
    """Хранить долгоживущую часть пользовательского состояния."""

    emotions: EmotionalState
    relationship: RelationshipState


class UserStatePersistence(Protocol):
    """Описывать хранилище долгоживущего пользовательского состояния."""

    def load(
        self,
        user_id: int,
    ) -> PersistentUserState | None:
        """Загрузить долгоживущее состояние пользователя."""
        ...

    def save(
        self,
        user_id: int,
        *,
        emotions: EmotionalState,
        relationship: RelationshipState,
    ) -> None:
        """Сохранить долгоживущее состояние пользователя."""
        ...


@dataclass(slots=True)
class UserState:
    """Хранить изменяемое состояние одного пользователя."""

    mood: str = "neutral"
    reply_count: int = 0
    history: deque[ConversationTurn] = field(
        default_factory=deque,
    )
    emotions: EmotionalState = field(
        default_factory=EmotionalState,
    )
    relationship: RelationshipState = field(
        default_factory=RelationshipState,
    )
    last_accessed_at: float = field(
        default=0.0,
        repr=False,
        compare=False,
    )
    active_operations: int = field(
        default=0,
        repr=False,
        compare=False,
    )
    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
        compare=False,
    )

    def register_reply(self) -> None:
        """Увеличить счётчик ответов этому пользователю."""
        self.reply_count += 1

    def reset_context(self) -> None:
        """Сбросить контекст диалога пользователя к начальному состоянию."""
        self.mood = "neutral"
        self.reply_count = 0
        self.history.clear()


class UserStateStore:
    """Хранить runtime-состояние пользователей текущего процесса."""

    def __init__(
        self,
        history_limit: int = 8,
        retention_seconds: float = 86400.0,
        clock: Callable[[], float] | None = None,
        persistence: UserStatePersistence | None = None,
    ) -> None:
        """Настроить историю и время хранения неактивных состояний."""
        if history_limit <= 0:
            raise ValueError("history_limit должен быть больше нуля")

        if retention_seconds <= 0:
            raise ValueError("retention_seconds должен быть больше нуля")

        if not isfinite(retention_seconds):
            raise ValueError("retention_seconds должен быть конечным числом")

        self._states: OrderedDict[int, UserState] = OrderedDict()
        self._history_limit = history_limit
        self._retention_seconds = retention_seconds
        self._clock = clock or monotonic
        self._persistence = persistence

    def get(self, user_id: int) -> UserState:
        """Получить состояние пользователя или создать новое."""
        now = self._clock()

        self._remove_stale_states(now)

        state = self._states.get(user_id)

        if state is None:
            persistent_state = None

            if self._persistence is not None:
                persistent_state = self._persistence.load(user_id)

            state = UserState(
                history=deque(
                    maxlen=self._history_limit,
                ),
                emotions=(
                    persistent_state.emotions
                    if persistent_state is not None
                    else EmotionalState()
                ),
                relationship=(
                    persistent_state.relationship
                    if persistent_state is not None
                    else RelationshipState()
                ),
                last_accessed_at=now,
            )

            self._states[user_id] = state
        else:
            state.last_accessed_at = now
            self._states.move_to_end(user_id)

        return state

    @asynccontextmanager
    async def use(
        self,
        user_id: int,
    ) -> AsyncIterator[UserState]:
        """Безопасно предоставить состояние для пользовательской операции."""
        state = self.get(user_id)
        state.active_operations += 1

        try:
            async with state.lock:
                try:
                    yield state
                finally:
                    if self._persistence is not None:
                        self._persistence.save(
                            user_id,
                            emotions=state.emotions,
                            relationship=state.relationship,
                        )
        finally:
            state.last_accessed_at = self._clock()
            self._states.move_to_end(user_id)
            state.active_operations -= 1

    def remove(self, user_id: int) -> bool:
        """Безопасно удалить состояние пользователя, если оно не используется."""
        state = self._states.get(user_id)

        if state is None:
            return False

        if self._is_state_in_use(state):
            return False

        del self._states[user_id]

        return True

    @property
    def tracked_users_count(self) -> int:
        """Вернуть количество состояний в памяти."""
        return len(self._states)

    @staticmethod
    def _is_state_in_use(state: UserState) -> bool:
        """Проверить, выполняется ли операция с состоянием."""
        return state.active_operations > 0 or state.lock.locked()

    def _remove_stale_states(
        self,
        now: float,
    ) -> None:
        """Удалить давно неиспользуемые состояния."""
        for user_id, state in list(self._states.items()):
            if now - state.last_accessed_at < self._retention_seconds:
                break

            if self._is_state_in_use(state):
                continue

            del self._states[user_id]
