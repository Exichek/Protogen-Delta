"""Тесты классификаторов на основе DeepSeek."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock

import pytest

from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier


def _create_deepseek_mock() -> tuple[DeepSeekService, AsyncMock]:
    """Создать мок DeepSeek без реальных API-запросов."""
    mock = AsyncMock(spec=DeepSeekService)

    return cast(DeepSeekService, mock), mock


@pytest.mark.parametrize(
    ("model_result", "expected"),
    [
        ("general", "general"),
        ("direct", "direct"),
        ("question", "question"),
        ("none", "none"),
    ],
)
def test_insult_classifier_returns_known_types(
    model_result: str,
    expected: str,
) -> None:
    """Известные ответы модели должны преобразовываться в тип оскорбления."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.return_value = model_result

    classifier = InsultClassifier(
        deepseek=deepseek,
        prompt="insult prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == expected
    mock.classify.assert_awaited_once_with(
        "insult prompt",
        "Сообщение",
    )


def test_insult_classifier_converts_direct_question() -> None:
    """Direct со знаком вопроса должен считаться question."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.return_value = "direct"

    classifier = InsultClassifier(
        deepseek=deepseek,
        prompt="prompt",
    )

    result = asyncio.run(
        classifier.classify("Ты тупой?"),
    )

    assert result == "question"


def test_insult_classifier_returns_none_for_unknown_result() -> None:
    """Неизвестный ответ модели должен превращаться в none."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.return_value = "something"

    classifier = InsultClassifier(
        deepseek=deepseek,
        prompt="prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == "none"


def test_insult_classifier_handles_deepseek_error() -> None:
    """Ошибка DeepSeek не должна ломать классификацию оскорбления."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.side_effect = DeepSeekError("API error")

    classifier = InsultClassifier(
        deepseek=deepseek,
        prompt="prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == "none"


def test_insult_classifier_rejects_empty_prompt() -> None:
    """Пустой промпт классификатора оскорблений запрещён."""
    deepseek, _ = _create_deepseek_mock()

    with pytest.raises(
        ValueError,
        match="Промпт классификации оскорблений",
    ):
        InsultClassifier(
            deepseek=deepseek,
            prompt="   ",
        )


@pytest.mark.parametrize(
    ("model_result", "expected"),
    [
        ("sweet", "sweet"),
        ("horny", "horny"),
        ("angry", "angry"),
        ("playful", "playful"),
    ],
)
def test_mood_classifier_returns_known_moods(
    model_result: str,
    expected: str,
) -> None:
    """Известное настроение должно возвращаться без изменений."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.return_value = model_result

    classifier = MoodClassifier(
        deepseek=deepseek,
        prompt="mood prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == expected
    mock.classify.assert_awaited_once_with(
        "mood prompt",
        "Сообщение",
    )


def test_mood_classifier_uses_playful_for_unknown_result() -> None:
    """Неизвестное настроение должно заменяться на playful."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.return_value = "unknown"

    classifier = MoodClassifier(
        deepseek=deepseek,
        prompt="prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == "playful"


def test_mood_classifier_handles_deepseek_error() -> None:
    """Ошибка DeepSeek должна возвращать None."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.side_effect = DeepSeekError("API error")

    classifier = MoodClassifier(
        deepseek=deepseek,
        prompt="prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result is None


def test_mood_classifier_rejects_empty_prompt() -> None:
    """Пустой промпт классификации настроения запрещён."""
    deepseek, _ = _create_deepseek_mock()

    with pytest.raises(
        ValueError,
        match="Промпт классификации настроения",
    ):
        MoodClassifier(
            deepseek=deepseek,
            prompt="",
        )


@pytest.mark.parametrize(
    ("model_result", "expected"),
    [
        ("active", "active"),
        ("passive", "passive"),
    ],
)
def test_fetish_role_classifier_returns_known_roles(
    model_result: str,
    expected: str,
) -> None:
    """Известная RP-роль должна возвращаться без изменений."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.return_value = model_result

    classifier = FetishRoleClassifier(
        deepseek=deepseek,
        prompt="role prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == expected
    mock.classify.assert_awaited_once_with(
        "role prompt",
        "Сообщение",
    )


def test_fetish_role_classifier_returns_unknown_for_invalid_result() -> None:
    """Неизвестный ответ модели должен превращаться в unknown."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.return_value = "something"

    classifier = FetishRoleClassifier(
        deepseek=deepseek,
        prompt="prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == "unknown"


def test_fetish_role_classifier_handles_deepseek_error() -> None:
    """Ошибка DeepSeek должна возвращать unknown."""
    deepseek, mock = _create_deepseek_mock()
    mock.classify.side_effect = DeepSeekError("API error")

    classifier = FetishRoleClassifier(
        deepseek=deepseek,
        prompt="prompt",
    )

    result = asyncio.run(
        classifier.classify("Сообщение"),
    )

    assert result == "unknown"


def test_fetish_role_classifier_rejects_empty_prompt() -> None:
    """Пустой промпт классификации роли запрещён."""
    deepseek, _ = _create_deepseek_mock()

    with pytest.raises(
        ValueError,
        match="Промпт классификации роли фетиша",
    ):
        FetishRoleClassifier(
            deepseek=deepseek,
            prompt=" ",
        )
