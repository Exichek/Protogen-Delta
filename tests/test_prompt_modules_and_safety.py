"""Условные модули и возрастные ограничения проверяются без внешних запросов."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer
from appearance_fixtures import verified
from test_adult_handler import _create_callback
from test_media_handler import _message
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.core.conversation_safety import declares_minor, reference_is_child
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.adult import AGE_RESTRICTED_TEXT, create_adult_router
from protogen_delta.handlers.safety import AgeSafetyMiddleware
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.prompt_composer import PromptComposer, PromptSections


@pytest.mark.parametrize("kind", ["text", "caption", "forwarded", "ordinary"])
def test_age_middleware_handles_own_statement_before_file_processing(kind: str) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        message, answer = _message()
        raw = cast(Mock, message)
        raw.from_user = SimpleNamespace(id=42, is_bot=False)
        message.text = "мне 16 лет" if kind in {"text", "forwarded"} else None
        message.caption = "мне 16 лет" if kind == "caption" else None
        raw.forward_origin = SimpleNamespace() if kind == "forwarded" else None
        handler = AsyncMock()
        await AgeSafetyMiddleware(engine)(handler, message, {})
        if kind in {"text", "caption"}:
            handler.assert_not_awaited()
            answer.assert_awaited_once()
            assert engine._user_states.get(42).age_restricted
        else:
            handler.assert_awaited_once()
            answer.assert_not_awaited()
        model.chat.assert_not_awaited()

    asyncio.run(scenario())


def test_minor_character_uses_neutral_modules_for_adult_account() -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.content_mode = "adult"
    state.roleplay_active = True
    state.roleplay_character = "Возраст: 15 лет. Персонаж в синем пальто."
    state.roleplay_fetishes = ("bondage",)
    engine._prompt_composer = PromptComposer(
        PromptSections(
            core="CORE", adult_body_male="ADULT_BODY", adult_roleplay="ADULT_RP"
        )
    )
    asyncio.run(engine.respond(42, "*передаю карту*"))
    prompt = model.chat.await_args.kwargs["system_prompt"]
    assert "ADULT_BODY" not in prompt and "ADULT_RP" not in prompt
    assert "интимные мотивы" not in prompt


def test_minor_closes_persisted_groups_and_keeps_other_users_unchanged(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        for user_id in (42, 7):
            for chat_id in (-100, -200):
                async with states.use_conversation(user_id, chat_id) as state:
                    state.roleplay_active = True
                    state.roleplay_character = "Сохранённый персонаж"
                    state.delta_appearance = "Сохранённый облик"
                    from protogen_delta.core.user_state import ConversationTurn

                    state.history.append(ConversationTurn("*беру карту*", "OLD_SCENE"))
        # Previously saved groups are not loaded into the new runtime cache.
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        await engine.respond(42, "Мне 16 лет")
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        for chat_id in (-100, -200):
            async with restored.use_conversation(42, chat_id) as state:
                assert (
                    state.age_restricted
                    and not state.roleplay_active
                    and not state.history
                )
                assert state.roleplay_character == "Сохранённый персонаж"
                assert state.delta_appearance == "Сохранённый облик"
            async with restored.use_conversation(7, chat_id) as other:
                assert (
                    other.roleplay_active and not other.age_restricted and other.history
                )
        model.chat.assert_not_awaited()

    asyncio.run(scenario())


def test_minor_callback_interrupts_pending_delivery() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.content_mode = "adult"
        started = asyncio.Event()

        async def chat(**kwargs: object) -> str:
            started.set()
            await asyncio.Event().wait()
            return "OLD_REPLY"

        async def deliver(reply: str) -> None:
            raise AssertionError("Old reply must not arrive")

        model.chat.side_effect = chat
        pending = asyncio.create_task(
            engine.respond_and_deliver(42, "Подскажи дорогу", deliver)
        )
        await started.wait()
        router = create_adult_router(
            engine._user_states, restrict_minor=engine.restrict_minor
        )
        callback, _, _ = _create_callback("adult:soft:42", user_id=42)
        await router.callback_query.handlers[0].callback(callback)
        assert (
            pending.cancelled()
            and state.age_restricted
            and state.content_mode == "soft"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("signal", ["стоп", "Мне 16 лет"])
def test_safety_signal_cancels_pending_reply_before_delivery(signal: str) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.roleplay_active = True
        state.content_mode = "adult"
        started = asyncio.Event()
        sent: list[str] = []

        async def chat(**kwargs: object) -> str:
            started.set()
            await asyncio.Event().wait()
            return "OLD_REPLY"

        async def deliver(reply: str) -> None:
            sent.append(reply)

        model.chat.side_effect = chat
        task = asyncio.create_task(engine.respond_and_deliver(42, "Продолжай", deliver))
        await started.wait()
        await engine.respond_and_deliver(42, signal, deliver)
        assert task.cancelled()
        assert len(sent) == 1 and "OLD_REPLY" not in sent
        assert not state.roleplay_active
        assert not engine._reply_tasks and not engine._delivering_users

    asyncio.run(scenario())


def test_stop_signal_does_not_cancel_other_chat() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        for chat_id in (None, -100):
            engine._user_states.get_conversation(42, chat_id).roleplay_active = True
        started = asyncio.Event()
        release = asyncio.Event()

        async def chat(**kwargs: object) -> str:
            started.set()
            await release.wait()
            return "GROUP_REPLY"

        async def deliver(reply: str) -> None:
            assert reply == "GROUP_REPLY"

        model.chat.side_effect = chat
        group = asyncio.create_task(
            engine.respond_and_deliver(42, "Продолжай", deliver, chat_id=-100)
        )
        await started.wait()
        await engine.disable_roleplay(42)
        assert not group.cancelled()
        release.set()
        await group

    asyncio.run(scenario())


def test_modules_follow_actual_topic_configuration_and_appearance() -> None:
    composer = PromptComposer(
        PromptSections(
            core="CORE",
            body="BODY",
            roleplay="RP",
            female_body="FEMALE",
            adult_body_male="ADULT_MALE",
            adult_body_female="ADULT_FEMALE",
            adult_roleplay="ADULT_RP",
            technical="TECH",
            visual="VISUAL",
            voice="VOICE",
            scene_voice="SCENE_VOICE",
        )
    )
    assert composer.compose("Docker", is_roleplay=False) == "CORE\n\nTECH"
    assert (
        composer.compose("Привет", is_roleplay=False, has_images=True)
        == "CORE\n\nVISUAL"
    )
    soft = composer.compose("*беру карту*", is_roleplay=True, female_configuration=True)
    assert "FEMALE" in soft and "ADULT" not in soft
    adult = composer.compose("*беру карту*", is_roleplay=True, adult_context=True)
    assert "ADULT_MALE" in adult and "ADULT_FEMALE" not in adult
    custom = composer.compose(
        "*беру карту*", is_roleplay=True, adult_context=True, has_custom_appearance=True
    )
    assert "BODY" not in custom and "ADULT_MALE" not in custom
    assert composer.examples(is_roleplay=False) == "VOICE"
    assert composer.examples(is_roleplay=True) == "VOICE\n\nSCENE_VOICE"


@pytest.mark.parametrize(
    "text",
    [
        "мне 16",
        "Мне сейчас 17 лет",
        "Мой возраст: 15 лет",
        "я несовершеннолетний",
        "мне нет 18",
    ],
)
def test_direct_minor_declaration_is_recognized(text: str) -> None:
    assert declares_minor(text)


@pytest.mark.parametrize(
    "text",
    [
        "мне 18 лет",
        "Ему 16 лет",
        "моему персонажу 15 лет",
        "Он сказал: «мне 16 лет»",
        "```мне 15 лет```",
        "мне 16 задач задали",
    ],
)
def test_quotes_other_people_and_unrelated_numbers_are_not_self_age(text: str) -> None:
    assert not declares_minor(text)


@pytest.mark.parametrize("chat_id", [None, -100])
def test_declared_minor_disables_adult_and_survives_restart(
    tmp_path: Path, chat_id: int | None
) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        async with states.use(42) as state:
            state.content_mode = "adult"
        async with states.use_conversation(42, chat_id) as state:
            state.roleplay_active = True
        reply = await engine.respond(42, "Мне 16 лет", chat_id=chat_id)
        assert "выключен" in reply
        model.chat.assert_not_awaited()
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with restored.use(42) as state:
            assert state.age_restricted and state.content_mode == "soft"
        async with restored.use_conversation(42, chat_id) as state:
            assert not state.roleplay_active
        server = MiniAppServer(TOKEN, restored, response_engine=engine)
        async with TestClient(TestServer(server.application())) as client:
            response = await client.patch(
                "/api/profile",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
                json={"content_mode": "adult"},
            )
            assert response.status == 403
        callback, edit, answer = _create_callback("adult:adult:42", user_id=42)
        router = create_adult_router(restored)
        await router.callback_query.handlers[0].callback(callback)
        answer.assert_awaited_once_with(AGE_RESTRICTED_TEXT, show_alert=True)
        edit.assert_not_awaited()

    asyncio.run(scenario())


def test_custom_stopword_is_saved_and_stops_before_model(tmp_path: Path) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        server = MiniAppServer(TOKEN, states, response_engine=engine)
        async with TestClient(TestServer(server.application())) as client:
            response = await client.patch(
                "/api/profile",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
                json={"roleplay_stopword": "Пауза"},
            )
            assert response.status == 200
            assert (await response.json())["roleplay_stopword"] == "Пауза"
            for bad in ("", "x" * 41, "<script>"):
                assert (
                    await client.patch(
                        "/api/profile",
                        headers={"X-Telegram-Init-Data": _signed_init_data()},
                        json={"roleplay_stopword": bad},
                    )
                ).status == 400
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        async with states.use(42) as state:
            state.roleplay_active = True
        assert "Сцена завершена" in await engine.respond(42, "ПАУЗА!")
        model.chat.assert_not_awaited()
        assert not states.get(42).roleplay_active

    asyncio.run(scenario())


def test_technical_question_skips_scene_and_adult_modules_without_erasing_scene() -> (
    None
):
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.content_mode = "adult"
    state.roleplay_active = True
    state.roleplay_fetishes = ("bondage",)
    engine._prompt_composer = PromptComposer(
        PromptSections(
            core="CORE",
            body="BODY",
            roleplay="RP_MODULE",
            adult_body_male="ADULT_BODY",
            technical="TECH",
        )
    )
    asyncio.run(engine.respond(42, "Как установить Docker?"))
    prompt = model.chat.await_args.kwargs["system_prompt"]
    assert "TECH" in prompt and "RP_MODULE" not in prompt and "ADULT_BODY" not in prompt
    assert "интимные мотивы" not in prompt
    assert state.roleplay_active


def test_child_reference_restriction_survives_edit_and_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        payload = json.loads(verified("Персонаж в синем пальто."))
        payload["minor_reference"] = True
        model.chat.return_value = json.dumps(payload)
        async with states.use(42) as state:
            state.roleplay_active = True
            state.content_mode = "adult"
        await engine.set_delta_appearance_from_image(
            42, ImageInput(b"image", "image/png")
        )
        old = states.get(42).delta_appearance
        assert not states.get(42).roleplay_active
        await engine.set_delta_appearance_from_text(
            42, "Персонаж в зелёном пальто.", expected_appearance=old
        )
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with restored.use(42) as state:
            assert state.delta_reference_restricted

    asyncio.run(scenario())
    assert reference_is_child("Персонажу 15 лет")
