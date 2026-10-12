"""Накопление эмоционального состояния и отношений после взаимодействия."""

from protogen_delta.core.user_state import UserState
from protogen_delta.services.insults import InsultType
from protogen_delta.services.mood import MoodType


def apply_interaction_effects(
    user_state: UserState,
    *,
    mood: MoodType | None,
    insult_type: InsultType,
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
