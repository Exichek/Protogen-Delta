"""Тесты простых сервисов обработки текста."""

from protogen_delta.config.json_loader import load_json
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


def test_real_fetish_triggers_recognize_combined_scene() -> None:
    """Рабочий словарь должен находить несколько явно введённых мотивов."""
    triggers = load_json("fetishes_triggers.json")

    result = detect_fetishes(
        "Надень латексный костюм, свяжи меня и кончи на спинку",
        triggers,
    )

    assert result == ["bondage", "latex", "bukkake"]


def test_real_fetish_triggers_handle_logged_rimming_phrase() -> None:
    """Фраза из живого теста должна определяться как римминг без кремпая."""
    triggers = load_json("fetishes_triggers.json")

    result = detect_fetishes(
        "Да, соси мою жопу, прямо туда внутрь язычком",
        triggers,
    )

    assert result == ["anal", "rimming"]


def test_real_fetish_triggers_do_not_use_generic_clothing_or_motion() -> None:
    """Обычные слова про кожу, лапы и движение внутрь не должны давать фетиши."""
    triggers = load_json("fetishes_triggers.json")

    result = detect_fetishes(
        "Кожа нагрелась, он сильно шагнул внутрь и коснулся лапами двери",
        triggers,
    )

    assert result == []
