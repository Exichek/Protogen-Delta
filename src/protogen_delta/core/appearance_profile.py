"""Ручной профиль характера текущего облика Дельты."""

import json
from dataclasses import asdict, dataclass
from typing import Self

PROFILE_FIELDS = ("personality", "behavior", "preferences", "boundaries")
PROFILE_FIELD_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class AppearanceProfile:
    personality: str = ""
    behavior: str = ""
    preferences: str = ""
    boundaries: str = ""

    @classmethod
    def from_payload(cls, value: object) -> Self:
        if not isinstance(value, dict) or set(value) != set(PROFILE_FIELDS):
            raise ValueError("Нужны характер, поведение, предпочтения и границы.")
        parts: list[str] = []
        for name in PROFILE_FIELDS:
            text = value[name]
            if not isinstance(text, str) or len(text) > PROFILE_FIELD_LIMIT:
                raise ValueError("Каждое поле профиля — текст до 1000 знаков.")
            parts.append(text.strip())
        return cls(*parts)

    @classmethod
    def decode(cls, value: str) -> Self | None:
        if not value:
            return cls()
        if len(value) > 16384:
            return None
        try:
            return cls.from_payload(json.loads(value))
        except ValueError, TypeError, RecursionError:
            return None

    def encode(self) -> str:
        values = asdict(self)
        return json.dumps(values, ensure_ascii=False) if any(values.values()) else ""
