"""Состояние отдельных пользователей во время работы приложения."""

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic, time
from typing import Literal, Protocol

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.conversation_gate import ConversationGate

logger = logging.getLogger(__name__)

StateKey = int | tuple[int, int]

ContentMode = Literal["unselected", "soft", "adult"]

_SECONDS_PER_HOUR = 3600.0
_WARMTH_DECAY_PER_HOUR = 0.01
_IRRITATION_DECAY_PER_HOUR = 0.02
_PLAYFULNESS_DECAY_PER_HOUR = 0.02
_AROUSAL_DECAY_PER_HOUR = 0.02


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

    def decay(
        self,
        elapsed_seconds: float,
    ) -> None:
        """Ослабить краткосрочные эмоции пропорционально прошедшему времени."""
        if elapsed_seconds <= 0.0:
            return

        elapsed_hours = elapsed_seconds / _SECONDS_PER_HOUR

        self.adjust(
            warmth=-_WARMTH_DECAY_PER_HOUR * elapsed_hours,
            irritation=-_IRRITATION_DECAY_PER_HOUR * elapsed_hours,
            playfulness=-_PLAYFULNESS_DECAY_PER_HOUR * elapsed_hours,
            arousal=-_AROUSAL_DECAY_PER_HOUR * elapsed_hours,
        )


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
    emotions_updated_at: float
    roleplay_active: bool = False
    roleplay_configuration: str = "male"
    roleplay_character: str = ""
    roleplay_fetishes: tuple[str, ...] = ()
    roleplay_preferences: str = ""
    roleplay_boundaries: str = ""
    delta_appearance: str = ""
    content_mode: ContentMode = "unselected"
    history: tuple[ConversationTurn, ...] = ()
    history_updated_at: float = 0.0


class UserStatePersistenceError(RuntimeError):
    """Ошибка чтения или сохранения долгоживущего состояния."""


class UserStatePersistence(Protocol):
    """Описывать хранилище долгоживущего пользовательского состояния."""

    async def load(
        self,
        user_id: StateKey,
    ) -> PersistentUserState | None:
        """Загрузить долгоживущее состояние пользователя."""
        ...

    async def save(
        self,
        user_id: StateKey,
        *,
        emotions: EmotionalState,
        relationship: RelationshipState,
        emotions_updated_at: float,
        roleplay_active: bool,
        roleplay_configuration: str = "male",
        roleplay_character: str = "",
        roleplay_fetishes: tuple[str, ...] = (),
        roleplay_preferences: str = "",
        roleplay_boundaries: str = "",
        delta_appearance: str = "",
        content_mode: ContentMode = "unselected",
        history: tuple[ConversationTurn, ...] | None = None,
        history_updated_at: float = 0.0,
        history_expires_before: float | None = None,
    ) -> None:
        """Сохранить долгоживущее состояние пользователя."""
        ...

    async def delete(
        self,
        user_id: StateKey,
    ) -> None:
        """Удалить долгоживущее состояние пользователя."""
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
    roleplay_active: bool = False
    roleplay_configuration: str = "male"
    roleplay_character: str = ""
    roleplay_fetishes: tuple[str, ...] = ()
    roleplay_preferences: str = ""
    roleplay_boundaries: str = ""
    delta_appearance: str = ""
    content_mode: ContentMode = "unselected"
    history_updated_at: float = 0.0
    emotions_updated_at: float = field(
        default=0.0,
        repr=False,
        compare=False,
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
    coordinating_operations: int = field(default=0, repr=False, compare=False)
    conversation_gate: ConversationGate = field(
        default_factory=ConversationGate, repr=False, compare=False
    )
    persistence_loaded: bool = field(
        default=False,
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
        self.history_updated_at = 0.0
        self.emotions.arousal = 0.0
        self.roleplay_active = False
        self.roleplay_fetishes = ()

    def reset_all(self) -> None:
        """Полностью сбросить пользовательское состояние."""
        self.reset_context()
        self.emotions = EmotionalState()
        self.relationship = RelationshipState()
        self.roleplay_configuration = "male"
        self.roleplay_character = ""
        self.roleplay_preferences = ""
        self.roleplay_boundaries = ""
        self.delta_appearance = ""
        self.content_mode = "unselected"


class UserStateStore:
    """Хранить runtime-состояние пользователей текущего процесса."""

    def __init__(
        self,
        history_limit: int = 8,
        retention_seconds: float = 86400.0,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        persistence: UserStatePersistence | None = None,
        history_ttl_seconds: float = 7 * 86400,
    ) -> None:
        """Настроить историю и время хранения неактивных состояний."""
        if history_limit <= 0:
            raise ValueError("history_limit должен быть больше нуля")

        if retention_seconds <= 0:
            raise ValueError("retention_seconds должен быть больше нуля")

        if not isfinite(retention_seconds):
            raise ValueError("retention_seconds должен быть конечным числом")

        if not isfinite(history_ttl_seconds) or history_ttl_seconds <= 0:
            raise ValueError(
                "history_ttl_seconds должен быть положительным конечным числом"
            )
        self._history_ttl_seconds = history_ttl_seconds
        self._states: OrderedDict[StateKey, UserState] = OrderedDict()
        self._history_limit = history_limit
        self._retention_seconds = retention_seconds
        self._clock = clock or monotonic
        self._wall_clock = wall_clock or time
        self._persistence = persistence

    def get(self, user_id: StateKey) -> UserState:
        """Получить runtime-состояние пользователя или создать новое."""
        now = self._clock()

        self._remove_stale_states(now)

        state = self._states.get(user_id)

        if state is None:
            state = UserState(
                history=deque(
                    maxlen=self._history_limit,
                ),
                emotions_updated_at=(
                    self._wall_clock() if self._persistence is None else 0.0
                ),
                last_accessed_at=now,
                persistence_loaded=self._persistence is None,
            )

            self._states[user_id] = state
        else:
            state.last_accessed_at = now
            self._states.move_to_end(user_id)

        return state

    @asynccontextmanager
    async def use(
        self,
        user_id: StateKey,
    ) -> AsyncIterator[UserState]:
        """Безопасно предоставить состояние для пользовательской операции."""
        async with self._coordinate(user_id, exclusive=isinstance(user_id, int)):
            async with self._use_state(user_id) as state:
                yield state

    @asynccontextmanager
    async def _coordinate(
        self, key: StateKey, *, exclusive: bool
    ) -> AsyncIterator[UserState]:
        user_id = key if isinstance(key, int) else key[1]
        private = self.get(user_id)
        # Очередь и активные чаты удерживают один и тот же gate в кеше.
        private.coordinating_operations += 1
        gate = private.conversation_gate
        try:
            async with gate.exclusive() if exclusive else gate.shared():
                yield private
        finally:
            private.last_accessed_at = self._clock()
            self._states.move_to_end(user_id)
            private.coordinating_operations -= 1

    @asynccontextmanager
    async def _use_state(self, user_id: StateKey) -> AsyncIterator[UserState]:
        state = self.get(user_id)
        state.active_operations += 1

        try:
            async with state.lock:
                await self._load_persistent_state(
                    user_id,
                    state,
                )

                self._decay_emotions(state)
                if (
                    state.history_updated_at > 0
                    and self._wall_clock() - state.history_updated_at
                    >= self._history_ttl_seconds
                ):
                    state.history.clear()
                    state.history_updated_at = 0.0
                original_history = tuple(state.history)

                try:
                    yield state
                finally:
                    if tuple(state.history) != original_history:
                        state.history_updated_at = (
                            self._wall_clock() if state.history else 0.0
                        )
                    if self._persistence is not None:
                        try:
                            await finish_operation(
                                self._persistence.save(
                                    user_id,
                                    emotions=state.emotions,
                                    relationship=state.relationship,
                                    emotions_updated_at=state.emotions_updated_at,
                                    roleplay_active=state.roleplay_active,
                                    roleplay_configuration=state.roleplay_configuration,
                                    roleplay_character=state.roleplay_character,
                                    roleplay_fetishes=state.roleplay_fetishes,
                                    roleplay_preferences=state.roleplay_preferences,
                                    roleplay_boundaries=state.roleplay_boundaries,
                                    delta_appearance=state.delta_appearance,
                                    content_mode=state.content_mode,
                                    history=tuple(state.history),
                                    history_updated_at=state.history_updated_at,
                                    history_expires_before=self._wall_clock()
                                    - self._history_ttl_seconds,
                                )
                            )
                        except UserStatePersistenceError:
                            logger.exception(
                                "Не удалось сохранить состояние пользователя %s",
                                user_id,
                            )
        finally:
            state.last_accessed_at = self._clock()
            self._states.move_to_end(user_id)
            state.active_operations -= 1

    def get_conversation(self, user_id: int, chat_id: int | None = None) -> UserState:
        """Получить только состояние этого чата, с общим возрастным режимом."""
        private = self.get(user_id)
        if chat_id is None:
            return private
        state = self.get((chat_id, user_id))
        state.content_mode = private.content_mode
        return state

    def mark_history_updated(self, state: UserState) -> None:
        """Учесть доставленный ход, даже если заполненная история не изменилась."""
        state.history_updated_at = self._wall_clock()

    @asynccontextmanager
    async def use_conversation(
        self, user_id: int, chat_id: int | None = None
    ) -> AsyncIterator[UserState]:
        """Блокировать только этот чат; общие изменения ждут все активные чаты."""
        async with self._coordinate(user_id, exclusive=False) as private:
            # Первое восстановление возраста короткое и однократное. После него
            # группам не нужен lock личного диалога на время генерации/доставки.
            if not private.persistence_loaded:
                async with private.lock:
                    await self._load_persistent_state(user_id, private)
            key: StateKey = user_id if chat_id is None else (chat_id, user_id)
            async with self._use_state(key) as state:
                if chat_id is not None:
                    state.content_mode = private.content_mode
                yield state

    def remove(self, user_id: StateKey) -> bool:
        """Безопасно удалить состояние пользователя, если оно не используется."""
        state = self._states.get(user_id)

        if state is None:
            return False

        if self._is_state_in_use(state):
            return False

        del self._states[user_id]

        return True

    async def reset_user(
        self,
        user_id: StateKey,
        *,
        cleanup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Полностью забыть состояние конкретного пользователя."""
        async with self._coordinate(user_id, exclusive=isinstance(user_id, int)):
            await self._reset_state(user_id, cleanup)

    async def _reset_state(
        self, user_id: StateKey, cleanup: Callable[[], Awaitable[None]] | None
    ) -> None:
        state = self.get(user_id)
        state.active_operations += 1

        try:
            async with state.lock:
                await finish_operation(self._reset_locked(user_id, state, cleanup))
        finally:
            state.last_accessed_at = self._clock()
            self._states.move_to_end(user_id)
            state.active_operations -= 1

    async def _reset_locked(
        self,
        user_id: StateKey,
        state: UserState,
        cleanup: Callable[[], Awaitable[None]] | None,
    ) -> None:
        """Завершить удаление и сброс RAM вместе, даже при отмене ожидающего запроса."""
        if self._persistence is not None:
            await self._persistence.delete(user_id)
        if isinstance(user_id, int):
            for key, conversation in self._states.items():
                if isinstance(key, tuple) and key[1] == user_id:
                    conversation.reset_all()
                    conversation.emotions_updated_at = self._wall_clock()
                    conversation.persistence_loaded = True
        state.reset_all()
        state.emotions_updated_at = self._wall_clock()
        state.persistence_loaded = True
        # Связанные хранилища очищаются под тем же барьером. Callback не должен
        # повторно входить в UserStateStore; обновление меню выполняется позже.
        if cleanup is not None:
            await cleanup()

    @property
    def tracked_users_count(self) -> int:
        """Вернуть количество состояний в памяти."""
        return len(self._states)

    async def _load_persistent_state(
        self,
        user_id: StateKey,
        state: UserState,
    ) -> None:
        """Однократно восстановить долгоживущее состояние под lock пользователя."""
        if self._persistence is None or state.persistence_loaded:
            return

        persistent_state = await self._persistence.load(user_id)

        if persistent_state is not None:
            state.emotions = persistent_state.emotions
            state.relationship = persistent_state.relationship
            state.emotions_updated_at = persistent_state.emotions_updated_at
            state.roleplay_active = persistent_state.roleplay_active
            state.roleplay_configuration = persistent_state.roleplay_configuration
            state.roleplay_character = persistent_state.roleplay_character
            state.roleplay_fetishes = persistent_state.roleplay_fetishes
            state.roleplay_preferences = persistent_state.roleplay_preferences
            state.roleplay_boundaries = persistent_state.roleplay_boundaries
            state.delta_appearance = persistent_state.delta_appearance
            state.content_mode = persistent_state.content_mode
            state.history = deque(persistent_state.history, maxlen=self._history_limit)
            state.history_updated_at = persistent_state.history_updated_at
        else:
            state.emotions_updated_at = self._wall_clock()

        state.persistence_loaded = True

    def _decay_emotions(
        self,
        state: UserState,
    ) -> None:
        """Ослабить эмоции согласно реально прошедшему времени."""
        now = self._wall_clock()
        elapsed_seconds = now - state.emotions_updated_at

        if elapsed_seconds <= 0.0:
            return

        state.emotions.decay(elapsed_seconds)
        state.emotions_updated_at = now

    @staticmethod
    def _is_state_in_use(state: UserState) -> bool:
        """Проверить, выполняется ли операция с состоянием."""
        return (
            state.active_operations > 0
            or state.coordinating_operations > 0
            or state.lock.locked()
        )

    def _remove_stale_states(
        self,
        now: float,
    ) -> None:
        """Удалить давно неиспользуемые состояния."""
        stale_ids = []
        for user_id, state in self._states.items():
            if now - state.last_accessed_at < self._retention_seconds:
                break

            if self._is_state_in_use(state):
                continue

            stale_ids.append(user_id)
        for user_id in stale_ids:
            del self._states[user_id]
