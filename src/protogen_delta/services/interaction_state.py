"""Накопление эмоционального состояния и отношений после взаимодействия."""

import re

from protogen_delta.core.user_state import UserState
from protogen_delta.services.insults import InsultType
from protogen_delta.services.mood import MoodType


def apply_interaction_effects(
    user_state: UserState,
    *,
    mood: MoodType | None,
    insult_type: InsultType,
    text: str = "",
    at: float | None = None,
    creator: bool = False,
) -> None:
    """Применить накопительные последствия сообщения."""
    user_state.relationship.adjust(
        familiarity=0.01,
    )

    if mood == "sweet":
        user_state.emotions.adjust(
            warmth=0.10,
            irritation=-0.03,
        )
        user_state.relationship.adjust(
            trust=0.01,
            affection=0.03,
            resentment=-0.01,
        )

    elif mood == "horny":
        user_state.emotions.adjust(
            arousal=0.10,
            playfulness=0.02,
        )

    elif mood == "angry":
        user_state.emotions.adjust(
            irritation=0.08,
            warmth=-0.02,
        )

    elif mood == "playful":
        user_state.emotions.adjust(
            playfulness=0.10,
            warmth=0.03,
        )

    if insult_type == "direct":
        user_state.emotions.adjust(
            irritation=0.10,
            warmth=-0.03,
        )
        user_state.relationship.adjust(
            trust=-0.05,
            affection=-0.03,
            resentment=0.08,
        )

    elif insult_type == "question":
        user_state.emotions.adjust(
            irritation=0.06,
            warmth=-0.02,
        )
        user_state.relationship.adjust(
            trust=-0.03,
            affection=-0.02,
            resentment=0.05,
        )

    if at is not None:
        # Цитаты и код не становятся сообщениями о результате или личном настроении.
        plain = re.sub(
            r"```.*?```|`[^`]*`|«[^»]*»|“[^”]*”|\"[^\"]*\"|(?m:^\s*>[^\n]*)",
            " ",
            text,
            flags=re.S,
        ).casefold()
        kinds: list[str] = []
        if insult_type in {"direct", "question"} or mood == "angry":
            kinds.append("conflict")
        else:
            if mood == "sweet" or re.search(
                r"\b(?:спасибо|благодарю|молодец)\b", plain
            ):
                kinds.append("support")
            if mood == "playful":
                kinds.append("play")
            if re.search(
                r"\b(?:получилось|заработало|исправил|исправили|сделал|готово)\b|тесты.{0,30}прошли",
                plain,
            ):
                kinds.append("progress")
            if "?" in plain or re.search(
                r"\b(?:объясни|проверь|интересно|почему|как работает)\b", plain
            ):
                kinds.append("question")
            if re.search(
                r"\b(?:мне грустно|я расстроен|я расстроился|я расстроилась)\b", plain
            ):
                kinds.append("sadness")
            if re.search(r"\b(?:я тревожусь|я переживаю|мне страшно|я боюсь)\b", plain):
                kinds.append("worry")
        positive = user_state.emotions.memory.observe(tuple(kinds), at, creator=creator)
        if creator and positive:
            user_state.relationship.adjust(trust=0.003, affection=0.005)
