"""Сохраняемое настроение персонажа и причины его изменения."""

import json
import math
from dataclasses import asdict, dataclass, field

EMOTION_LABELS = {
    "joy": "радость",
    "curiosity": "интерес",
    "sadness": "грусть",
    "anxiety": "тревога",
    "confidence": "уверенность",
    "tiredness": "усталость",
}
EVENT_LABELS = {
    "support": "тёплое сообщение пользователя",
    "play": "дружеская шутка или игра",
    "progress": "пользователь сообщил о результате",
    "question": "новая тема или вопрос",
    "conflict": "резкая реплика или конфликт",
    "sadness": "пользователь сообщил о грусти",
    "worry": "пользователь сообщил о тревоге",
}


@dataclass(frozen=True, slots=True)
class EmotionalEvent:
    """Причина хранится как проверенный тип события, без текста переписки."""

    kind: str
    at: float
    creator: bool = False


@dataclass(slots=True)
class EmotionalMemory:
    """Сохранять накопление, ограничивая влияние повторяющихся сигналов."""

    joy: float = 0.0
    curiosity: float = 0.0
    sadness: float = 0.0
    anxiety: float = 0.0
    confidence: float = 0.0
    tiredness: float = 0.0
    creator_contacts: int = 0
    events: list[EmotionalEvent] = field(default_factory=list)
    positive_updated_at: float = -1.0

    def adjust(self, **changes: float) -> None:
        for name, delta in changes.items():
            if name not in EMOTION_LABELS or not math.isfinite(delta):
                raise ValueError("Некорректное изменение настроения")
            setattr(self, name, max(0.0, min(1.0, getattr(self, name) + delta)))

    def decay(self, elapsed_seconds: float) -> None:
        """Пауза ослабляет показатели; молчание не является негативным событием."""
        if elapsed_seconds <= 0:
            return
        hours = elapsed_seconds / 3600
        self.adjust(
            joy=-0.002 * hours,
            curiosity=-0.005 * hours,
            sadness=-0.01 * hours,
            anxiety=-0.015 * hours,
            confidence=-0.001 * hours,
            tiredness=-0.02 * hours,
        )

    def observe(self, kinds: tuple[str, ...], at: float, *, creator: bool) -> bool:
        """Вернуть, принят ли новый позитивный сигнал для отношений."""
        if (
            not math.isfinite(at)
            or at < 0
            or at > 253402300799
            or any(kind not in EVENT_LABELS for kind in kinds)
        ):
            raise ValueError("Некорректное эмоциональное событие")
        if creator:
            self.creator_contacts = min(1000000000, self.creator_contacts + 1)
        positive = any(kind in {"support", "play", "progress"} for kind in kinds)
        permitted = positive and (
            self.positive_updated_at < 0 or at - self.positive_updated_at >= 900
        )
        if permitted:
            self.positive_updated_at = at
        for kind in dict.fromkeys(kinds):
            if kind in {"support", "play", "progress"} and not permitted:
                continue
            if kind == "support":
                self.adjust(joy=0.045, confidence=0.015, anxiety=-0.02, sadness=-0.01)
            elif kind == "play":
                self.adjust(joy=0.035, curiosity=0.02, anxiety=-0.01)
            elif kind == "progress":
                self.adjust(joy=0.03, confidence=0.04, curiosity=0.015)
            elif kind == "question":
                self.adjust(curiosity=0.015)
            elif kind == "conflict":
                self.adjust(joy=-0.025, anxiety=0.07, confidence=-0.015)
            elif kind == "sadness":
                self.adjust(sadness=0.025)
            elif kind == "worry":
                self.adjust(anxiety=0.03)
            self.events.append(EmotionalEvent(kind, at, creator))
        self.events = self.events[-8:]
        self.adjust(tiredness=0.001)
        return permitted

    def encode(self) -> str:
        return json.dumps(
            {"version": 1, **asdict(self)}, ensure_ascii=False, separators=(",", ":")
        )

    @classmethod
    def decode(
        cls, value: str, *, legacy_joy: float = 0, legacy_confidence: float = 0
    ) -> "EmotionalMemory":
        """Старые/повреждённые данные сохраняют доступные прежние показатели."""
        fallback = cls(
            joy=max(0.0, min(1.0, legacy_joy)),
            confidence=max(0.0, min(1.0, legacy_confidence)),
        )
        try:
            data = json.loads(value)
            if (
                not isinstance(data, dict)
                or type(data.get("version")) is not int
                or data.get("version") != 1
            ):
                return fallback
            values = {name: data.get(name, 0) for name in EMOTION_LABELS}
            if any(
                type(number) not in {int, float}
                or not math.isfinite(number)
                or not 0 <= number <= 1
                for number in values.values()
            ):
                return fallback
            contacts = data.get("creator_contacts", 0)
            updated = data.get("positive_updated_at", -1)
            if (
                type(contacts) is not int
                or not 0 <= contacts <= 1000000000
                or type(updated) not in {int, float}
                or not math.isfinite(updated)
                or not -1 <= updated <= 253402300799
            ):
                return fallback
            events = []
            raw_events = data.get("events", [])
            if not isinstance(raw_events, list):
                return fallback
            for event in raw_events[-8:]:
                if not isinstance(event, dict) or event.get("kind") not in EVENT_LABELS:
                    return fallback
                at = event.get("at")
                creator = event.get("creator", False)
                if (
                    not isinstance(at, (int, float))
                    or isinstance(at, bool)
                    or not math.isfinite(at)
                    or not 0 <= at <= 253402300799
                    or type(creator) is not bool
                ):
                    return fallback
                events.append(EmotionalEvent(event["kind"], at, creator))
            return cls(
                **values,
                creator_contacts=contacts,
                positive_updated_at=updated,
                events=events,
            )
        except TypeError, ValueError, KeyError:
            return fallback
