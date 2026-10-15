"""Проверки предпочтений облика Дельты в RP."""

import asyncio
from unittest.mock import patch

import pytest
from test_response_engine import _create_engine

from protogen_delta.core.appearance_profile import AppearanceProfile
from protogen_delta.core.user_state import ContentMode


def test_fetishes_are_passed_to_adult_rp() -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)

    state.content_mode = "adult"
    state.roleplay_active = True
    state.delta_appearance = "Тестовый облик"
    state.delta_appearance_profile = AppearanceProfile(
        fetishes="связал; неизвестное предпочтение"
    ).encode()

    asyncio.run(engine.respond(42, "*осматриваюсь вокруг*"))

    assert model.chat.await_args is not None
    prompt = model.chat.await_args.kwargs["system_prompt"]

    assert "связал; неизвестное предпочтение" in prompt
    assert "бондаж" in prompt
    assert "самостоятельно" in prompt

    # Предпочтения облика не меняют состояние текущей сцены.
    assert state.roleplay_fetishes == ()


@pytest.mark.parametrize(
    ("roleplay", "mode", "personal"),
    [
        (False, "adult", True),
        (True, "soft", True),
        (True, "adult", False),
    ],
)
def test_fetishes_not_leaked_outside_allowed_rp(
    roleplay: bool,
    mode: ContentMode,
    personal: bool,
) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)

    state.content_mode = mode
    state.roleplay_active = roleplay
    state.delta_appearance = "Тестовый облик"
    state.delta_appearance_profile = AppearanceProfile(
        fetishes="связал; СЕКРЕТНЫЙ_МАРКЕР"
    ).encode()

    asyncio.run(
        engine.respond(
            42,
            "Привет",
            use_personal_facts=personal,
        )
    )

    assert model.chat.await_args is not None
    prompt = model.chat.await_args.kwargs["system_prompt"]

    assert "СЕКРЕТНЫЙ_МАРКЕР" not in prompt


@pytest.mark.parametrize(
    ("fetishes", "expected"),
    [
        ("связал", True),
        ("", False),
    ],
)
def test_appearance_fetishes_activate_adult_rp_prompt(
    fetishes: str,
    expected: bool,
) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)

    state.content_mode = "adult"
    state.roleplay_active = True
    state.delta_appearance = "Тестовый облик"
    state.delta_appearance_profile = AppearanceProfile(fetishes=fetishes).encode()

    with patch.object(
        engine._prompt_composer,
        "assemble",
        wraps=engine._prompt_composer.assemble,
    ) as composer:
        asyncio.run(engine.respond(42, "*осматриваюсь вокруг*"))

    assert model.chat.await_args is not None
    assert composer.call_args is not None
    assert composer.call_args.kwargs["adult_context"] is expected

    # Само поле не активирует фетиши текущего разговора.
    assert state.roleplay_fetishes == ()


@pytest.mark.parametrize(
    "restriction",
    [
        "age_restricted",
        "delta_reference_restricted",
        "roleplay_character_restricted",
    ],
)
def test_restrictions_hide_appearance_fetishes(
    restriction: str,
) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)

    state.content_mode = "adult"
    state.roleplay_active = True
    state.delta_appearance = "Тестовый облик"
    state.delta_appearance_profile = AppearanceProfile(
        fetishes="связал; СЕКРЕТНЫЙ_МАРКЕР"
    ).encode()

    setattr(state, restriction, True)

    asyncio.run(engine.respond(42, "*осматриваюсь*"))

    assert model.chat.await_args is not None
    prompt = model.chat.await_args.kwargs["system_prompt"]

    assert "СЕКРЕТНЫЙ_МАРКЕР" not in prompt
    assert "Распознанные темы из словаря" not in prompt


def test_appearance_fetishes_update_between_messages() -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)

    state.content_mode = "adult"
    state.roleplay_active = True
    state.delta_appearance = "Первый облик"
    state.delta_appearance_profile = AppearanceProfile(fetishes="связал").encode()

    asyncio.run(engine.respond(42, "*осматриваюсь*"))

    assert model.chat.await_args is not None
    first_prompt = model.chat.await_args.kwargs["system_prompt"]

    assert "связал" in first_prompt
    assert "бондаж" in first_prompt

    # Имитируем сохранение другого профиля облика.
    state.delta_appearance = "Второй облик"
    state.delta_appearance_profile = AppearanceProfile(
        fetishes="неизвестная новая тема"
    ).encode()

    asyncio.run(engine.respond(42, "*продолжаю прогулку*"))

    second_prompt = model.chat.await_args.kwargs["system_prompt"]

    assert "неизвестная новая тема" in second_prompt
    assert "связал" not in second_prompt
    assert "бондаж" not in second_prompt

    # Без выбранного облика предпочтения не передаются.
    state.delta_appearance = ""

    asyncio.run(engine.respond(42, "*останавливаюсь*"))

    third_prompt = model.chat.await_args.kwargs["system_prompt"]

    assert "неизвестная новая тема" not in third_prompt
    assert state.roleplay_fetishes == ()


def test_scene_and_appearance_fetishes_work_together() -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)

    state.content_mode = "adult"
    state.roleplay_active = True
    state.roleplay_fetishes = ("bondage",)

    state.delta_appearance = "Тестовый облик"
    state.delta_appearance_profile = AppearanceProfile(
        fetishes="Уникальное предпочтение Дельты"
    ).encode()

    asyncio.run(engine.respond(42, "*осматриваюсь вокруг*"))

    assert model.chat.await_args is not None
    prompt = model.chat.await_args.kwargs["system_prompt"]

    # Сохранённые предпочтения Дельты доступны модели.
    assert "Уникальное предпочтение Дельты" in prompt

    # Старые мотивы сцены и предпочтения облика совместимы.
    assert "Не вводи произвольные фетиши из справочника." in prompt
    assert "Исключение — предпочтения текущего облика Дельты" in prompt

    # Старая противоречащая инструкция отсутствует.
    assert "Не добавляй другие фетиши из справочника" not in prompt

    # Предпочтения облика не изменяют фетиши сцены.
    assert state.roleplay_fetishes == ("bondage",)
