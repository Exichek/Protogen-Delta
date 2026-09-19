"""Тесты простых сервисов обработки текста."""

from protogen_delta.services.fetishes import detect_fetishes


def test_detect_fetishes_finds_multiple_matches() -> None:
    """В одном сообщении должны находиться все подходящие категории."""
    triggers = {
        "bondage": ["верёвка", "связал"],
        "oral": ["минет"],
        "public": ["публично"],
    }

    result = detect_fetishes(
        "Связал тебя верёвкой и сделал минет",
        triggers,
    )

    assert result == ["bondage", "oral"]


def test_detect_fetishes_is_case_insensitive() -> None:
    """Поиск ключевых слов не должен зависеть от регистра."""
    triggers = {
        "bondage": ["связал"],
    }

    result = detect_fetishes(
        "Я ТЕБЯ СВЯЗАЛ",
        triggers,
    )

    assert result == ["bondage"]


def test_detect_fetishes_does_not_match_inside_other_words() -> None:
    """Триггер не должен срабатывать как часть другого слова."""
    triggers = {
        "submission": ["раб"],
        "leather": ["кожан*"],
    }

    result = detect_fetishes(
        "Работа закончена, а кожура лежит на столе",
        triggers,
    )

    assert result == []


def test_detect_fetishes_supports_explicit_prefix_triggers() -> None:
    """Триггер со звёздочкой должен совпадать с началом слова."""
    triggers = {
        "bondage": ["свяж*"],
        "crossdressing": ["юбк*"],
    }

    result = detect_fetishes(
        "Свяжешь меня и наденешь юбку",
        triggers,
    )

    assert result == ["bondage", "crossdressing"]
