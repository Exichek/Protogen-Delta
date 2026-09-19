"""Тесты преобразования накопленного состояния в контекст модели."""

from protogen_delta.core.user_state import UserState
from protogen_delta.services.state_context import build_state_context


def test_fresh_state_has_no_context() -> None:
    """Новому пользователю не нужен искусственный контекст отношений."""
    state = UserState()

    assert build_state_context(state) == []


def test_state_context_describes_familiarity() -> None:
    """Накопленное знакомство должно отражаться в контексте."""
    state = UserState()

    state.relationship.adjust(
        familiarity=0.30,
    )

    context = build_state_context(state)

    assert any(
        "не воспринимает его как совершенно незнакомого" in line for line in context
    )


def test_state_context_describes_affection_and_trust() -> None:
    """Сформированные доверие и симпатия должны влиять на контекст."""
    state = UserState()

    state.relationship.adjust(
        trust=0.40,
        affection=0.30,
    )

    context = build_state_context(state)

    assert any("склонен доверять пользователю" in line for line in context)
    assert any("сформировалась заметная симпатия" in line for line in context)


def test_state_context_preserves_conflict_residue() -> None:
    """Накопленная обида и раздражение должны сохранять след конфликта."""
    state = UserState()

    state.relationship.adjust(
        resentment=0.25,
    )
    state.emotions.adjust(
        irritation=0.30,
    )

    context = build_state_context(state)

    assert any("остался некоторый осадок" in line for line in context)
    assert any("сохраняется лёгкое раздражение" in line for line in context)


def test_state_context_can_mix_affection_and_resentment() -> None:
    """Тёплое отношение и обида могут существовать одновременно."""
    state = UserState()

    state.relationship.adjust(
        affection=0.40,
        resentment=0.25,
    )

    context = build_state_context(state)

    assert any("сформировалась заметная симпатия" in line for line in context)
    assert any("остался некоторый осадок" in line for line in context)


def test_state_context_describes_warmth_and_playfulness() -> None:
    """Накопленные краткосрочные эмоции должны отражаться в контексте."""
    state = UserState()

    state.emotions.adjust(
        warmth=0.30,
        playfulness=0.30,
    )

    context = build_state_context(state)

    assert any("сохраняется некоторое тепло" in line for line in context)
    assert any("остаётся немного игривости" in line for line in context)


def test_state_context_hides_arousal_outside_intimate_context() -> None:
    """Возбуждение не должно протекать в обычный разговор."""
    state = UserState()

    state.emotions.adjust(
        arousal=0.60,
    )

    context = build_state_context(
        state,
        include_intimate=False,
    )

    assert all("возбуждение" not in line for line in context)


def test_state_context_includes_arousal_in_intimate_context() -> None:
    """В интимном контексте накопленное возбуждение можно учитывать."""
    state = UserState()

    state.emotions.adjust(
        arousal=0.60,
    )

    context = build_state_context(
        state,
        include_intimate=True,
    )

    assert any("выраженное возбуждение" in line for line in context)
