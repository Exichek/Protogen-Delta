"""Проверки явных границ между разговором и сценой."""

import pytest

from protogen_delta.core.roleplay import (
    scene_character,
    scene_configuration,
    split_roleplay_stop,
)


@pytest.mark.parametrize("text", ["Стоп RP!", "  хватит   рп  ", "закончим RP"])
def test_stop_without_question(text: str) -> None:
    assert split_roleplay_stop(text) == ""


def test_stop_keeps_follow_up_question() -> None:
    assert split_roleplay_stop("Стоп RP, теперь просто поговорим. Что такое TCP?") == (
        "теперь просто поговорим. Что такое TCP?"
    )


@pytest.mark.parametrize(
    "text",
    ["не останавливай RP", "Он сказал: «стоп RP»", "стоп rpg", "Что значит стоп RP?"],
)
def test_quoted_or_unrelated_stop_is_not_a_command(text: str) -> None:
    assert split_roleplay_stop(text) is None


def test_configuration_requires_explicit_assignment() -> None:
    assert (
        scene_configuration("Давай в этой сцене ты в женской конфигурации.") == "female"
    )
    assert scene_configuration("Ты в мужской конфигурации") == "male"
    assert scene_configuration("Я в женской конфигурации") is None
    assert scene_configuration("Не будь в женской конфигурации") is None
    assert scene_configuration("Она подошла ближе") is None


def test_character_is_only_taken_from_explicit_description() -> None:
    assert scene_character("Мой персонаж: человек в пальто") == "человек в пальто"
    assert scene_character("У тебя визор") is None
    assert scene_character("Мой персонаж: " + "x" * 501) is None
