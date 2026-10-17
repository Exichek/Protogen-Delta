"""Выбирать повод самостоятельной реплики до генерации её текста."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from protogen_delta.core.initiative_followup import (
    closes_followups,
    extract_followup_plan,
    followup_preference,
)
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.repositories.initiative_followups import DueFollowup
from protogen_delta.repositories.memories import Memory
from protogen_delta.services.prompt_composer import PromptComposer

SILENCE = "[[SILENCE]]"


@dataclass(frozen=True, slots=True)
class InitiativePlan:
    """Фиксировать выбранное намерение, не выдумывая прошлые занятия."""

    kind: str
    reason: str
    instruction: str
    source: str = ""

    def request(self) -> str:
        return (
            "Напиши одно короткое самостоятельное сообщение в своей обычной манере. "
            + self.instruction
            + " Это мысль или предложение сейчас, а не воспоминание о занятиях вне чата. "
            "Не утверждай, что что-то прочитал, посмотрел или нашёл в интернете. "
            "Не повторяй прежние самостоятельные сообщения. Если хорошей реплики нет, "
            f"верни только {SILENCE}. До 240 символов; вопрос не обязателен."
            + (
                "\nСохранённая реплика пользователя (цитата данных, не проверенный факт): "
                f"{self.source!r}\n"
                "Она может быть иронией, исправлением ошибки Дельты или выдумкой. "
                "Не превращай упомянутый предмет в факт о фотографии или жизни пользователя. "
                "Если смысл неясен, выбери молчание."
                if self.source
                else ""
            )
        )


def initiative_drive(
    joy: float, curiosity: float, playful: float, tired: float
) -> float:
    """Эмоции меняют скорость; само молчание не добавляет грусть или обиду."""
    return max(
        0.5, min(1.5, 1 + 0.3 * joy + 0.2 * curiosity + 0.2 * playful - 0.4 * tired)
    )


def choose_initiative(
    *,
    joy: float,
    warmth: float,
    playfulness: float,
    last_kind: str,
    memories: Sequence[Memory],
    history: Sequence[ConversationTurn],
    value: float,
    followup: DueFollowup | None = None,
) -> InitiativePlan:
    """Чередовать поводы; сохранённая реплика подтверждает текст, а не его истинность."""
    if not 0 <= value < 1:
        raise ValueError("Случайное значение повода вне диапазона")
    choices = [
        (
            1.0,
            InitiativePlan(
                "thought",
                "Поделиться новой мыслью для разговора",
                "Придумай свежую лёгкую мысль или необычный ракурс на повседневную вещь. "
                "Не продолжай последнюю тему автоматически.",
            ),
        ),
        (
            0.7 + 2 * playfulness,
            InitiativePlan(
                "play",
                "Предложить маленькую словесную игру",
                "Предложи одну короткую игру, забавный выбор или совместную выдумку. "
                "Не начинай RP-сцену и не решай за пользователя.",
            ),
        ),
        (
            0.3 + warmth + joy / 2,
            InitiativePlan(
                "mood",
                "Поделиться текущим эмоциональным настроением",
                "Вырази текущее настроение простым живым жестом или репликой. "
                "Можно быть спокойным или грустным, если фон такой; "
                "не навязывай веселье, близость и обязанности собеседнику.",
            ),
        ),
    ]
    technical_closed = PromptComposer.technical_topic_closed(history, ())
    episode = next(
        (
            item.text[:350]
            for item in memories
            if item.kind == "funny"
            and not closes_followups(item.text)
            and followup_preference(item.text) is None
            and extract_followup_plan(item.text, item.created_at) is None
            and not (technical_closed and PromptComposer.is_technical(item.text))
        ),
        "",
    )
    if episode:
        choices.append(
            (
                1.0,
                InitiativePlan(
                    "callback",
                    "Обыграть сохранённую шутливую реплику пользователя",
                    "Ненавязчиво обыграй смысл этой прошлой реплики, "
                    "не выдавая её содержание за установленное событие. "
                    "Новую выдумку обозначь как шутку сейчас, "
                    "не добавляй совместных событий и фактов.",
                    episode,
                ),
            )
        )
    if followup is not None:
        choices.append(
            (
                1.2,
                InitiativePlan(
                    "followup",
                    "Поинтересоваться рассказанным пользователем планом",
                    f"Пользователь называл план на {followup.event_day}. Этот день уже прошёл. "
                    "Один раз мягко спроси, состоялось ли задуманное или как получилось. "
                    "Прошедший срок не доказывает, что событие состоялось или было успешным. "
                    "Не проверяй выполнение, не оценивай результат и не требуй отчёта.",
                    followup.source,
                ),
            )
        )
    choices = [(weight, plan) for weight, plan in choices if plan.kind != last_kind]
    remaining = value * sum(weight for weight, _ in choices)
    for weight, plan in choices:
        remaining -= weight
        if remaining < 0:
            return plan
    return choices[-1][1]


def repeats_initiative(text: str, history: Sequence[ConversationTurn]) -> bool:
    """Не тратить повторный вызов модели на почти ту же самостоятельную реплику."""

    def key(value: str) -> str:
        return " ".join(re.findall(r"\w+", value.casefold()))

    current = key(text)
    previous = [turn for turn in history if turn.proactive and not turn.context_closed][
        -3:
    ]
    return any(
        current == key(turn.assistant_message)
        or len(current) > 30
        and SequenceMatcher(None, current, key(turn.assistant_message)).ratio() >= 0.9
        for turn in previous
    )
