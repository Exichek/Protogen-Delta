"""Тесты накопительного состояния после взаимодействия."""

import pytest

from protogen_delta.core.user_state import UserState
from protogen_delta.services.interaction_state import (
    apply_interaction_effects,
)


def test_interaction_increases_familiarity() -> None:
    """Каждое взаимодействие должно немного повышать знакомство."""
    state = UserState()

    apply_interaction_effects(
        state,
        mood="neutral",
        insult_type="none",
    )

    assert state.relationship.familiarity == pytest.approx(0.01)


def test_sweet_interaction_increases_warmth_and_affection() -> None:
    """Тёплое сообщение должно улучшать эмоциональный фон и отношения."""
    state = UserState()

    apply_interaction_effects(
        state,
        mood="sweet",
        insult_type="none",
    )

    assert state.emotions.warmth == pytest.approx(0.10)
    assert state.emotions.irritation == 0.0

    assert state.relationship.familiarity == pytest.approx(0.01)
    assert state.relationship.trust == pytest.approx(0.01)
    assert state.relationship.affection == pytest.approx(0.03)
    assert state.relationship.resentment == 0.0


def test_playful_interaction_increases_playfulness() -> None:
    """Игривое сообщение должно усиливать игривость и немного тепло."""
    state = UserState()

    apply_interaction_effects(
        state,
        mood="playful",
        insult_type="none",
    )

    assert state.emotions.playfulness == pytest.approx(0.10)
    assert state.emotions.warmth == pytest.approx(0.03)


def test_horny_interaction_increases_arousal() -> None:
    """Интимный контекст должен повышать возбуждение без изменения отношений."""
    state = UserState()

    apply_interaction_effects(
        state,
        mood="horny",
        insult_type="none",
    )

    assert state.emotions.arousal == pytest.approx(0.10)
    assert state.emotions.playfulness == pytest.approx(0.02)

    assert state.relationship.trust == 0.0
    assert state.relationship.affection == 0.0
    assert state.relationship.resentment == 0.0


def test_direct_insult_damages_relationship() -> None:
    """Прямое оскорбление должно накапливать раздражение и обиду."""
    state = UserState()

    state.emotions.adjust(
        warmth=0.5,
    )
    state.relationship.adjust(
        trust=0.5,
        affection=0.5,
    )

    apply_interaction_effects(
        state,
        mood="angry",
        insult_type="direct",
    )

    assert state.emotions.irritation == pytest.approx(0.18)
    assert state.emotions.warmth == pytest.approx(0.44)
    assert state.relationship.trust == pytest.approx(0.45)
    assert state.relationship.affection == pytest.approx(0.47)
    assert state.relationship.resentment == pytest.approx(0.08)


def test_question_insult_has_smaller_relationship_effect() -> None:
    """Оскорбительный вопрос должен вредить отношениям слабее прямого."""
    direct = UserState()
    question = UserState()

    direct.relationship.adjust(
        trust=0.5,
        affection=0.5,
    )
    question.relationship.adjust(
        trust=0.5,
        affection=0.5,
    )

    apply_interaction_effects(
        direct,
        mood="angry",
        insult_type="direct",
    )

    apply_interaction_effects(
        question,
        mood="angry",
        insult_type="question",
    )

    assert direct.relationship.resentment > question.relationship.resentment
    assert direct.relationship.trust < question.relationship.trust
    assert direct.relationship.affection < question.relationship.affection


def test_general_insult_does_not_damage_relationship() -> None:
    """Агрессия к третьему лицу не должна портить отношения с Дельтой."""
    state = UserState()

    state.relationship.adjust(
        trust=0.5,
        affection=0.5,
    )

    apply_interaction_effects(
        state,
        mood="angry",
        insult_type="general",
    )

    assert state.relationship.trust == pytest.approx(0.5)
    assert state.relationship.affection == pytest.approx(0.5)
    assert state.relationship.resentment == 0.0


def test_unknown_mood_does_not_repeat_previous_emotion() -> None:
    """Неудачная классификация не должна повторять предыдущую реакцию."""
    state = UserState()

    apply_interaction_effects(
        state,
        mood="sweet",
        insult_type="none",
    )

    affection = state.relationship.affection

    apply_interaction_effects(
        state,
        mood=None,
        insult_type="none",
    )

    assert state.emotions.warmth == pytest.approx(0.09)
    assert state.relationship.affection == pytest.approx(affection)
    assert state.relationship.familiarity == pytest.approx(0.02)


def test_neutral_interactions_decay_transient_emotions() -> None:
    """Нейтральные сообщения должны постепенно успокаивать эмоции."""
    state = UserState()

    state.emotions.adjust(
        warmth=0.20,
        irritation=0.20,
        playfulness=0.20,
        arousal=0.20,
    )

    apply_interaction_effects(
        state,
        mood="neutral",
        insult_type="none",
    )

    assert state.emotions.warmth == pytest.approx(0.19)
    assert state.emotions.irritation == pytest.approx(0.18)
    assert state.emotions.playfulness == pytest.approx(0.18)
    assert state.emotions.arousal == pytest.approx(0.18)


def test_emotional_decay_does_not_go_below_zero() -> None:
    """Затухание не должно уводить эмоциональные показатели ниже нуля."""
    state = UserState()

    state.emotions.adjust(
        warmth=0.005,
        irritation=0.005,
        playfulness=0.005,
        arousal=0.005,
    )

    apply_interaction_effects(
        state,
        mood="neutral",
        insult_type="none",
    )

    assert state.emotions.warmth == 0.0
    assert state.emotions.irritation == 0.0
    assert state.emotions.playfulness == 0.0
    assert state.emotions.arousal == 0.0


def test_interaction_effects_accumulate() -> None:
    """Повторные взаимодействия должны постепенно накапливать состояние."""
    state = UserState()

    for _ in range(3):
        apply_interaction_effects(
            state,
            mood="sweet",
            insult_type="none",
        )

    assert state.emotions.warmth == pytest.approx(0.28)
    assert state.relationship.familiarity == pytest.approx(0.03)
    assert state.relationship.affection == pytest.approx(0.09)
