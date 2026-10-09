"""Согласованность двух персонажей на нейтральных сценах без внешнего API."""

import asyncio
import json
from dataclasses import replace

import pytest
from appearance_fixtures import verified
from test_response_engine import _create_engine

from protogen_delta.core.roleplay import roleplay_actions, user_will_start_scene
from protogen_delta.core.user_state import ConversationTurn, UserState
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.response_engine import AppearanceAnalysisError
from protogen_delta.services.scene_continuity import scene_continuity_context


def test_action_sources_and_order_are_explicit_and_closed_scene_is_excluded() -> None:
    state = UserState(roleplay_active=True)
    state.history.extend(
        [
            ConversationTurn(
                "*у старого маяка*", "*старая сцена*", context_closed=True
            ),
            ConversationTurn("*показываю на твой нос*", "*поднимаю руку*"),
            ConversationTurn("*беру фонарь*", "*опускаю руку и беру карту*"),
        ]
    )
    context = scene_continuity_context(state, has_images=True)
    events = json.loads(context.split("Недавние действия по авторам: ", 1)[1])
    assert events == [
        {"автор": "пользователь", "действия": "показываю на твой нос"},
        {"автор": "Дельта", "действия": "поднимаю руку"},
        {"автор": "пользователь", "действия": "беру фонарь"},
        {"автор": "Дельта", "действия": "опускаю руку и беру карту"},
    ]
    assert "старого маяка" not in context
    state.stop_roleplay()
    assert scene_continuity_context(state, has_images=True) == ""


def test_action_context_is_bounded_and_excludes_code_and_math() -> None:
    assert roleplay_actions("`*код*` **важно** 2 * 3 * 4; *беру карту*") == (
        "беру карту",
    )
    state = UserState(roleplay_active=True)
    state.history.extend(
        ConversationTurn("*" + "а" * 8000 + "*", "*" + "б" * 16000 + "*")
        for _ in range(30)
    )
    context = scene_continuity_context(state, has_images=True)
    events = json.loads(context.split("Недавние действия по авторам: ", 1)[1])
    assert len(events) == 6
    assert all(len(event["действия"]) <= 240 for event in events)
    assert len(context) < 4100


@pytest.mark.parametrize("configuration", ["male", "female"])
def test_actor_profiles_survive_small_dynamic_budget_and_image_input(
    configuration: str,
) -> None:
    engine, _, model, *_ = _create_engine()
    engine._config = replace(engine._config, dynamic_state_chars=100)
    state = engine._user_states.get(42)
    state.roleplay_active = True
    state.roleplay_configuration = configuration
    state.delta_appearance = "Белые волосы Дельты. " + "Светлый покров. " * 90
    state.roleplay_character = "Пользователь: синт, гладкий корпус без волос."
    state.roleplay_boundaries = "Не решай мои действия за меня."
    state.history.extend(
        ConversationTurn("Обсуждаем карту" * 30, "Старый ответ" * 30) for _ in range(8)
    )
    asyncio.run(
        engine.respond(
            42,
            "Это референс моего персонажа",
            images=(ImageInput(b"image", "image/png"),),
        )
    )
    prompt = model.chat.await_args.kwargs["system_prompt"]
    assert (
        "женская; говори о себе в женском роде" in prompt
        if configuration == "female"
        else "мужская; говори о себе в мужском роде" in prompt
    )
    assert state.delta_appearance in prompt
    assert state.roleplay_character in prompt
    assert state.roleplay_boundaries in prompt
    assert prompt.count(state.delta_appearance) == 1
    assert prompt.count(state.roleplay_character) == 1
    assert "Картинка не сбрасывает текущую сцену" in prompt
    assert "Старый ответ бота не подтверждает новую анатомию" in prompt
    assert state.roleplay_active
    model.chat.assert_awaited_once()
    model.analyze_visual_features.assert_not_awaited()


@pytest.mark.parametrize(
    "text", ["Хочу рп, без сюжета, я начну", "Давай RP. Начну с первого хода."]
)
def test_user_takes_first_turn_without_setup_interrogation(text: str) -> None:
    engine, _, model, *_ = _create_engine()
    assert user_will_start_scene(text)
    assert (
        asyncio.run(engine.respond(42, text))
        == "Хорошо, начинай. Подхвачу твой первый ход."
    )
    assert engine._user_states.get(42).roleplay_active
    model.chat.assert_not_awaited()


def test_reference_pose_is_not_saved_in_appearance_or_scene() -> None:
    engine, _, model, *_ = _create_engine()
    payload = json.loads(verified("Белые волосы, фиолетовые глаза."))
    payload["reference_state"] = "Рот приоткрыт, руки подняты над головой."
    model.chat.return_value = json.dumps(payload, ensure_ascii=False)
    asyncio.run(
        engine.set_delta_appearance_from_image(42, ImageInput(b"image", "image/png"))
    )
    state = engine._user_states.get(42)
    assert "Белые волосы" in state.delta_appearance
    assert "приоткрыт" not in state.delta_appearance
    assert "подняты" not in state.delta_appearance
    assert not state.roleplay_active and not state.history


@pytest.mark.parametrize("reference_state", [False, "x" * 401])
def test_bad_reference_state_preserves_previous_appearance(
    reference_state: object,
) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.delta_appearance = "Прежнее описание"
    payload = json.loads(verified("Новое описание"))
    payload["reference_state"] = reference_state
    model.chat.return_value = json.dumps(payload)
    with pytest.raises(AppearanceAnalysisError):
        asyncio.run(
            engine.set_delta_appearance_from_image(
                42, ImageInput(b"image", "image/png")
            )
        )
    assert state.delta_appearance == "Прежнее описание"
