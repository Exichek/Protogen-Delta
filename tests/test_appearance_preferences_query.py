"""Чтение текущего поля профиля в чате без генерации и утечки в сцену."""

import asyncio
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from test_response_engine import _create_engine
from test_simple_handlers import _call_first_handler, _create_message_mock

from protogen_delta.core.appearance_profile import AppearanceProfile
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.appearance_preferences_query import (
    is_appearance_preferences_query,
)
from protogen_delta.services.stickers import ContextualStickerService

QUESTION = "Какие у меня сейчас фетиши?"


@pytest.mark.parametrize(
    "message",
    [
        QUESTION,
        "Какие фетиши у меня сейчас стоят?",
        "Дельта, какие у тебя фетиши сохранены?",
        "Какие у меня ща фетиши стоят",
        "Покажи сохранённые фетиши",
        "Напомни мне предпочтения текущего облика",
        "Что у меня в поле фетиши?",
        "Что сейчас записано в поле «Фетиши и предпочтения»?",
        "Что записано в поле „Фетиши и предпочтения“?",
    ],
)
def test_explicit_queries_are_recognized(message: str) -> None:
    assert is_appearance_preferences_query(message)


@pytest.mark.parametrize(
    "message",
    [
        "Что такое фетиши?",
        "Какие фетиши бывают?",
        "Какие у него сейчас фетиши?",
        "Какие у меня предпочтения в музыке?",
        "Объясни код, который отвечает на вопрос: какие у меня фетиши?",
        "> Какие у меня сейчас фетиши?",
        '"Какие у меня сейчас фетиши?"',
        "`Какие у меня сейчас фетиши?`",
        "*спрашиваю* Какие у меня сейчас фетиши?",
        "Какие у меня сейчас фетиши? И продолжи сцену",
        "Какие у меня сейчас фетиши?" + " " * 241,
    ],
)
def test_other_messages_keep_normal_processing(message: str) -> None:
    assert not is_appearance_preferences_query(message)


@pytest.mark.parametrize("active", [False, True])
def test_query_reads_saved_text_without_model_or_scene_changes(active: bool) -> None:
    engine, _, model, insult, mood, role = _create_engine()
    engine._config = replace(
        engine._config,
        fetish_triggers={"topic_a": ["комета"]},
        fetish_names={"topic_a": "Астрономия"},
    )
    state = engine._user_states.get(42)
    state.delta_appearance = "Выбранный облик"
    state.delta_appearance_profile = AppearanceProfile(
        fetishes="комета; мой неизвестный пункт"
    ).encode()
    state.roleplay_active = active
    state.roleplay_fetishes = ("chat_topic",)
    state.history.append(ConversationTurn("Прежний ход", "Прежний ответ"))
    before_history = list(state.history)
    before_emotions = asdict(state.emotions)
    before_version = state.history_updated_at

    reply = asyncio.run(engine.respond(42, QUESTION))

    assert "комета; мой неизвестный пункт" in reply
    assert "Распознанные темы: Астрономия." in reply
    assert state.roleplay_active is active
    assert state.roleplay_fetishes == ("chat_topic",)
    assert list(state.history) == before_history
    assert state.history_updated_at == before_version
    assert asdict(state.emotions) == before_emotions
    for service in (
        model.chat,
        model.classify,
        insult.classify,
        mood.classify,
        role.classify,
    ):
        service.assert_not_awaited()


@pytest.mark.parametrize(
    ("appearance", "profile", "expected"),
    [
        ("", AppearanceProfile(fetishes="orphan").encode(), "ничего не записано"),
        ("Облик", "", "ничего не записано"),
        ("Облик", "[]", "ничего не записано"),
        (
            "Облик",
            AppearanceProfile(fetishes="Неизвестное значение").encode(),
            "совпадений со словарём нет",
        ),
    ],
)
def test_absent_empty_invalid_and_unknown_values_are_reported(
    appearance: str, profile: str, expected: str
) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.delta_appearance = appearance
    state.delta_appearance_profile = profile
    reply = asyncio.run(engine.respond(42, QUESTION))
    assert expected in reply
    assert "orphan" not in reply
    model.chat.assert_not_awaited()


@pytest.mark.parametrize(
    ("chat_id", "personal"), [(None, False), (-100, False), (-100, True)]
)
def test_nonpersonal_context_cannot_disclose_saved_fields(
    chat_id: int | None, personal: bool
) -> None:
    engine, _, model, *_ = _create_engine()
    private = engine._user_states.get(42)
    private.delta_appearance = "Личный облик"
    private.delta_appearance_profile = AppearanceProfile(
        fetishes="PRIVATE_VALUE"
    ).encode()
    scoped = engine._user_states.get_conversation(42, chat_id)
    if chat_id is not None:
        scoped.delta_appearance = "Групповой облик"
        scoped.delta_appearance_profile = AppearanceProfile(
            fetishes="GROUP_VALUE"
        ).encode()
        scoped.roleplay_active = True
    reply = asyncio.run(
        engine.respond(42, QUESTION, chat_id=chat_id, use_personal_facts=personal)
    )
    assert "PRIVATE_VALUE" not in reply and "GROUP_VALUE" not in reply
    assert "в личном чате" in reply
    if chat_id is not None:
        assert scoped.roleplay_active
    model.chat.assert_not_awaited()


def test_queries_follow_profile_edits_and_persisted_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        original = AppearanceProfile(fetishes="OLD_VALUE")
        async with states.use(42) as state:
            state.delta_appearance = "Облик"
            state.delta_appearance_profile = original.encode()
        assert "OLD_VALUE" in await engine.respond(42, QUESTION)
        updated = AppearanceProfile(fetishes="NEW_VALUE")
        await engine.set_delta_appearance_profile(
            42, updated, expected_appearance="Облик", expected_profile=original
        )
        reply = await engine.respond(42, QUESTION)
        assert "NEW_VALUE" in reply and "OLD_VALUE" not in reply
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        assert "NEW_VALUE" in await engine.respond(42, QUESTION)
        assert "NEW_VALUE" not in await engine.respond(7, QUESTION)
        model.chat.assert_not_awaited()

    asyncio.run(scenario())


def test_stop_command_takes_priority_and_remaining_query_can_be_answered() -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.delta_appearance = "Облик"
    state.delta_appearance_profile = AppearanceProfile(fetishes="SAVED_VALUE").encode()
    state.roleplay_active = True
    state.roleplay_stopword = QUESTION.rstrip("?")
    reply = asyncio.run(engine.respond(42, QUESTION))
    assert "Сцена завершена" in reply and "SAVED_VALUE" not in reply
    state.roleplay_active = True
    state.roleplay_stopword = "стоп"
    reply = asyncio.run(engine.respond(42, "Стоп RP, " + QUESTION))
    assert "SAVED_VALUE" in reply
    assert not state.roleplay_active
    model.chat.assert_not_awaited()


def test_failed_delivery_can_retry_without_history_or_model_calls() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(42)
        state.delta_appearance = "Облик"
        state.delta_appearance_profile = AppearanceProfile(
            fetishes="SAVED_VALUE"
        ).encode()
        failed = AsyncMock(side_effect=RuntimeError("delivery failed"))
        with pytest.raises(RuntimeError, match="delivery failed"):
            await engine.respond_and_deliver(42, QUESTION, failed)
        assert not state.history and state.reply_count == 0
        delivered = AsyncMock()
        await engine.respond_and_deliver(42, QUESTION, delivered)
        delivered.assert_awaited_once()
        assert delivered.await_args is not None
        assert "SAVED_VALUE" in delivered.await_args.args[0]
        assert not state.history and state.reply_count == 1
        model.chat.assert_not_awaited()

    asyncio.run(scenario())


def test_metadata_reply_does_not_trigger_sticker_reaction() -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.delta_appearance = "Облик"
    state.delta_appearance_profile = AppearanceProfile(fetishes="SAVED_VALUE").encode()
    stickers = Mock(spec=ContextualStickerService)
    stickers.is_request.return_value = False
    stickers.capabilities_context.return_value = "Synthetic capabilities"
    stickers.maybe_send = AsyncMock()
    message, answer, _ = _create_message_mock(QUESTION, user_id=42)
    router = create_text_router(engine, sticker_service=stickers)
    asyncio.run(_call_first_handler(router, message))
    answer.assert_awaited_once()
    assert answer.await_args is not None
    assert "SAVED_VALUE" in answer.await_args.args[0]
    stickers.maybe_send.assert_not_awaited()
    model.chat.assert_not_awaited()


@pytest.mark.parametrize("attachment", ["document", "override"])
def test_attachments_keep_normal_model_handling(attachment: str) -> None:
    engine, _, model, *_ = _create_engine()
    state = engine._user_states.get(42)
    state.delta_appearance = "Облик"
    state.delta_appearance_profile = AppearanceProfile(fetishes="SAVED_VALUE").encode()
    reply = asyncio.run(
        engine.respond(
            42,
            QUESTION,
            attachment_text=(
                "Содержимое документа" if attachment == "document" else None
            ),
            model_message_override=(
                "Содержимое вложения" if attachment == "override" else None
            ),
        )
    )
    assert reply == "Ответ"
    model.chat.assert_awaited_once()
    assert "SAVED_VALUE" not in model.chat.await_args.kwargs["system_prompt"]
