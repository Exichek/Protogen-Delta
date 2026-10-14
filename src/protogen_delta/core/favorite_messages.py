"""Проверенные причины выбора сообщения в избранное."""

from dataclasses import dataclass
from typing import Literal

FavoriteReason = Literal["support", "humor", "shared_success", "interesting_thought"]
FAVORITE_LABELS = {
    "support": "искренняя поддержка",
    "humor": "удачная шутка",
    "shared_success": "совместный успех",
    "interesting_thought": "интересная мысль",
}


@dataclass(frozen=True, slots=True)
class FavoriteMessage:
    id: int
    user_id: int
    text: str
    reason: FavoriteReason
    created_at: float
