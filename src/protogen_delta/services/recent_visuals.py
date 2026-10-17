"""Короткий повторный просмотр исходных изображений без записи в базу."""

import asyncio
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic

from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.deepseek import ImageInput

_VISUAL_REFERENCE = re.compile(
    r"\b(?:(?:на|в|этом?|этой|том|той)\s+(?:фото\w*|картин\w*|"
    r"изображени\w*|скрин\w*|стикер\w*|кадр\w*|снимке|рисунке)|"
    r"пересмотри|присмотрись)\b",
    re.I,
)
_CORRECTION = re.compile(
    r"\b(?:там\s+(?:нет|не)|это\s+не|"
    r"где\s+ты\s+(?:это\s+)?(?:увидел\w*|наш[её]л)|"
    r"ты\s+(?:ошиб\w*|перепутал\w*)|что\s+(?:там|слева|справа))\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class _Visual:
    images: tuple[ImageInput, ...]
    turn: ConversationTurn
    expires: float
    size: int


class RecentVisuals:
    """Хранить ограниченные байты для уточнений в том же личном диалоге."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        ttl: float = 600,
        max_bytes: int = 64 * 1024 * 1024,
        max_item_bytes: int = 20 * 1024 * 1024,
        max_entries: int = 16,
    ) -> None:
        if min(ttl, max_bytes, max_item_bytes, max_entries) <= 0:
            raise ValueError(
                "Ограничения повторного просмотра должны быть положительными"
            )
        self._clock = clock
        self._ttl = ttl
        self._max_bytes = max_bytes
        self._max_item_bytes = max_item_bytes
        self._max_entries = max_entries
        self._entries: OrderedDict[int, _Visual] = OrderedDict()
        self._expiry: dict[int, asyncio.TimerHandle] = {}

    def forget(self, user_id: int) -> None:
        """Удалить временные изображения владельца."""
        self._entries.pop(user_id, None)
        if timer := self._expiry.pop(user_id, None):
            timer.cancel()

    def _prune(self) -> None:
        now = self._clock()
        for owner, entry in tuple(self._entries.items()):
            if now >= entry.expires:
                self.forget(owner)

    def remember(
        self, user_id: int, images: Sequence[ImageInput], turn: ConversationTurn
    ) -> None:
        """Сохранить полный набор после доставки ответа; старый набор заменить."""
        self.forget(user_id)
        self._prune()
        size = sum(len(image.data) for image in images)
        if not images or size > min(self._max_bytes, self._max_item_bytes):
            return
        self._entries[user_id] = _Visual(
            tuple(images), turn, self._clock() + self._ttl, size
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            self._expiry[user_id] = loop.call_later(self._ttl, self.forget, user_id)
        while (
            len(self._entries) > self._max_entries
            or sum(entry.size for entry in self._entries.values()) > self._max_bytes
        ):
            self.forget(next(iter(self._entries)))

    def recall(
        self, user_id: int, history: Sequence[ConversationTurn], text: str
    ) -> tuple[ImageInput, ...]:
        """Повторить только недавнее вложение при прямом визуальном уточнении."""
        self._prune()
        entry = self._entries.get(user_id)
        if entry is None:
            return ()
        # Проверяем сам объект хода: reset, выгрузка состояния или другая
        # сессия с похожим текстом не дают доступа к прежним байтам.
        recent = tuple(history)[-3:]
        position = next(
            (i for i, turn in enumerate(recent) if turn is entry.turn), None
        )
        if (
            position is None
            or any(turn.context_closed or turn.proactive for turn in recent[position:])
            or not (_VISUAL_REFERENCE.search(text) or _CORRECTION.search(text))
        ):
            self.forget(user_id)
            return ()
        return entry.images
