"""Тесты движка формирования ответов."""

import asyncio
import logging
from typing import cast
from unittest.mock import ANY, AsyncMock

import pytest

from protogen_delta.core.log_context import LogContextFilter
from protogen_delta.core.message_utils import split_message
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import (
    ConversationTurn,
    UserStateStore,
)
from protogen_delta.services.deepseek import (
    DeepSeekAPIError,
    DeepSeekAuthError,
    DeepSeekConnectionError,
    DeepSeekError,
    DeepSeekRateLimitError,
    DeepSeekService,
    DeepSeekTimeoutError,
)
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.response_engine import (
    ResponseBusyError,
    ResponseEngine,
    ResponseEngineConfig,
)

TEST_USER_ID = 123456


def _create_engine() -> tuple[
    ResponseEngine,
    BotState,
    AsyncMock,
    AsyncMock,
    AsyncMock,
    AsyncMock,
]:
    """Создать движок с моками всех внешних сервисов."""
    deepseek_mock = AsyncMock(spec=DeepSeekService)
    insult_mock = AsyncMock(spec=InsultClassifier)
    mood_mock = AsyncMock(spec=MoodClassifier)
    role_mock = AsyncMock(spec=FetishRoleClassifier)

    deepseek_mock.chat.return_value = "Ответ"
    insult_mock.classify.return_value = "none"
    mood_mock.classify.return_value = "neutral"
    role_mock.classify.return_value = "unknown"

    state = BotState()
    # Проверки реакций не должны зависеть от скорости выполнения и decay.
    user_states = UserStateStore(wall_clock=lambda: 1000.0)

    config = ResponseEngineConfig(
        fetish_triggers={
            "bondage": ["связал"],
        },
        fetish_names={
            "bondage": "бондаж",
        },
        system_prompt="SYSTEM PROMPT",
        rp_prompt="RP PROMPT",
    )

    engine = ResponseEngine(
        deepseek=cast(DeepSeekService, deepseek_mock),
        insult_classifier=cast(InsultClassifier, insult_mock),
        mood_classifier=cast(MoodClassifier, mood_mock),
        fetish_role_classifier=cast(
            FetishRoleClassifier,
            role_mock,
        ),
        bot_state=state,
        user_states=user_states,
        config=config,
    )

    return (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        role_mock,
    )


def test_response_engine_routes_greeting_through_chat() -> None:
    """Одиночное приветствие должно обрабатываться основной моделью."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    deepseek_mock.chat.return_value = "Привет в ответ"

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Привет!"),
    )

    assert result == "Привет в ответ"
    assert state.reply_count == 1

    insult_mock.classify.assert_awaited_once_with(
        "Привет!",
    )
    mood_mock.classify.assert_awaited_once_with(
        "Привет!",
    )

    assert deepseek_mock.chat.await_args is not None
    assert deepseek_mock.chat.await_args.kwargs["system_prompt"].startswith(
        "SYSTEM PROMPT"
    )
    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt=ANY,
        user_message="Привет!",
        history=(),
    )

    assert list(engine._user_states.get(TEST_USER_ID).history) == [
        ConversationTurn(
            user_message="Привет!",
            assistant_message="Привет в ответ",
        )
    ]


def test_response_engine_passes_direct_insult_to_chat() -> None:
    """Прямое оскорбление должно передаваться модели как конфликтный контекст."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    insult_mock.classify.return_value = "direct"
    mood_mock.classify.return_value = "angry"
    deepseek_mock.chat.return_value = "Контекстный ответ"

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Ты идиот"),
    )

    assert result == "Контекстный ответ"
    assert state.reply_count == 1

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt.startswith("SYSTEM PROMPT")
    assert "оскорбляет Дельту напрямую" in prompt
    assert "раздражение или злость" in prompt

    assert list(engine._user_states.get(TEST_USER_ID).history) == [
        ConversationTurn(
            user_message="Ты идиот",
            assistant_message="Контекстный ответ",
        )
    ]


def test_response_engine_updates_mood_and_calls_chat() -> None:
    """Обычное сообщение должно обновить настроение и вызвать DeepSeek."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        mood_mock,
        role_mock,
    ) = _create_engine()

    mood_mock.classify.return_value = "neutral"
    deepseek_mock.chat.return_value = "Обычный ответ"

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела?"),
    )

    assert result == "Обычный ответ"
    assert engine._user_states.get(TEST_USER_ID).mood == "neutral"
    assert state.reply_count == 1

    assert deepseek_mock.chat.await_args is not None
    assert deepseek_mock.chat.await_args.kwargs["system_prompt"].startswith(
        "SYSTEM PROMPT"
    )
    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt=ANY,
        user_message="Как дела?",
        history=(),
    )

    role_mock.classify.assert_not_awaited()


def test_response_engine_uses_rp_prompt() -> None:
    """RP-сообщение должно использовать готовый RP-промпт без legacy-правил."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        role_mock,
    ) = _create_engine()

    deepseek_mock.chat.return_value = "RP ответ"

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "*обнял тебя*"),
    )

    assert result == "RP ответ"
    assert state.reply_count == 1
    assert engine._user_states.get(TEST_USER_ID).roleplay_active is True

    role_mock.classify.assert_not_awaited()

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt.startswith("RP PROMPT")
    assert "Текущая конфигурация Дельты" in prompt
    assert "Протогены не носят штанов" not in prompt
    assert "Никогда не используй слово" not in prompt


def test_response_engine_adds_fetish_context_and_role() -> None:
    """Тема и направление действия должны добавляться в динамический RP-контекст."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        role_mock,
    ) = _create_engine()

    role_mock.classify.return_value = "active"
    deepseek_mock.chat.return_value = "RP ответ"

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "*связал тебя*"),
    )

    assert result == "RP ответ"
    assert state.reply_count == 1

    role_mock.classify.assert_awaited_once_with(
        "*связал тебя*",
    )

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt.startswith("RP PROMPT")
    assert "## Контекст текущего сообщения" in prompt
    assert "бондаж" in prompt
    assert "совершить действие над пользователем" in prompt


def test_response_engine_does_not_classify_role_without_fetish() -> None:
    """Без найденного фетиша отдельная классификация роли не нужна."""
    (
        engine,
        _,
        _,
        _,
        _,
        role_mock,
    ) = _create_engine()

    asyncio.run(
        engine.respond(TEST_USER_ID, "*погладил тебя*"),
    )

    role_mock.classify.assert_not_awaited()


def test_response_engine_returns_fallback_on_chat_error() -> None:
    """Ошибка обычного запроса DeepSeek должна дать аварийный ответ."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = RuntimeError("API error")

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела"),
    )

    assert result == "Бля, у тостера что-то сломалось... ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_timeout() -> None:
    """Timeout DeepSeek должен возвращать понятный ответ пользователю."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = DeepSeekTimeoutError("DeepSeek не ответил вовремя")

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела"),
    )

    assert result == "Я чёт завис и слишком долго думаю... попробуй ещё раз ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_rate_limit() -> None:
    """Лимит запросов DeepSeek должен возвращать отдельный ответ."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = DeepSeekRateLimitError(
        "Превышен лимит запросов DeepSeek"
    )

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела"),
    )

    assert result == (
        "Меня сейчас слишком сильно дёргают запросами... " "дай мне немного времени ≧◡≦"
    )
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_connection_error() -> None:
    """Ошибка соединения с DeepSeek должна возвращать отдельный ответ."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = DeepSeekConnectionError(
        "Не удалось подключиться к DeepSeek"
    )

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела"),
    )

    assert result == "У меня отвалилось соединение... попробуй чуть позже ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_auth_error() -> None:
    """Ошибка доступа DeepSeek не должна раскрывать технические детали."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = DeepSeekAuthError("Ошибка авторизации DeepSeek")

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела"),
    )

    assert result == "У меня какая-то внутренняя хуйня сломалась... попробуй позже ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_api_error() -> None:
    """Ошибка API DeepSeek должна возвращать безопасный ответ."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = DeepSeekAPIError(
        "DeepSeek вернул ошибку API",
        status_code=500,
    )

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела"),
    )

    assert result == "У меня мозги сейчас чудят... попробуй чуть позже ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_unknown_deepseek_error() -> None:
    """Неизвестная ошибка DeepSeek должна использовать общий fallback."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = DeepSeekError("Неизвестная ошибка DeepSeek")

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Как дела"),
    )

    assert result == "Бля, у тостера что-то сломалось... ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_does_not_classify_fetish_role_outside_rp() -> None:
    """Обычный текст с fetish-триггером не должен вызывать классификатор роли."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        role_mock,
    ) = _create_engine()

    deepseek_mock.chat.return_value = "Обычный ответ"

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Ты меня связал?"),
    )

    assert result == "Обычный ответ"

    role_mock.classify.assert_not_awaited()

    assert deepseek_mock.chat.await_args is not None
    assert deepseek_mock.chat.await_args.kwargs["system_prompt"].startswith(
        "SYSTEM PROMPT"
    )
    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt=ANY,
        user_message="Ты меня связал?",
        history=(),
    )


def test_response_engine_handles_empty_deepseek_reply() -> None:
    """Пустой ответ DeepSeek должен заменяться понятным сообщением."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.return_value = ""

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Обычное сообщение"),
    )

    assert result == "Пустой ответ от DeepSeek"
    assert state.reply_count == 1


def test_response_engine_handles_whitespace_deepseek_reply() -> None:
    """Ответ только из пробелов должен считаться молчанием модели."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.return_value = "   "

    result = asyncio.run(
        engine.respond(TEST_USER_ID, "Обычное сообщение"),
    )

    assert result == "DeepSeek промолчал..."
    assert state.reply_count == 1


def test_response_engine_rejects_empty_system_prompt() -> None:
    """Пустой системный промпт должен запрещать создание движка."""
    deepseek_mock = AsyncMock(spec=DeepSeekService)
    insult_mock = AsyncMock(spec=InsultClassifier)
    mood_mock = AsyncMock(spec=MoodClassifier)
    role_mock = AsyncMock(spec=FetishRoleClassifier)

    config = ResponseEngineConfig(
        fetish_triggers={},
        fetish_names={},
        system_prompt="   ",
        rp_prompt="RP",
    )

    with pytest.raises(
        ValueError,
        match="Системный промпт",
    ):
        ResponseEngine(
            deepseek=cast(DeepSeekService, deepseek_mock),
            insult_classifier=cast(
                InsultClassifier,
                insult_mock,
            ),
            mood_classifier=cast(
                MoodClassifier,
                mood_mock,
            ),
            fetish_role_classifier=cast(
                FetishRoleClassifier,
                role_mock,
            ),
            bot_state=BotState(),
            user_states=UserStateStore(),
            config=config,
        )


def test_response_engine_passes_question_insult_to_chat() -> None:
    """Вопросительное оскорбление должно передаваться модели с контекстом."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    insult_mock.classify.return_value = "question"
    mood_mock.classify.return_value = "angry"
    deepseek_mock.chat.return_value = "Контекстный ответ"

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Ты совсем тупой?",
        )
    )

    assert result == "Контекстный ответ"
    assert state.reply_count == 1

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt.startswith("SYSTEM PROMPT")
    assert "оскорбление Дельты как вопрос" in prompt
    assert "раздражение или злость" in prompt


def test_response_engine_passes_general_insult_to_chat() -> None:
    """Оскорбление третьего лица должно обрабатываться в основном диалоге."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    insult_mock.classify.return_value = "general"
    deepseek_mock.chat.return_value = "Контекстный ответ"

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Этот разработчик долбоёб",
        )
    )

    assert result == "Контекстный ответ"
    assert state.reply_count == 1

    mood_mock.classify.assert_awaited_once_with(
        "Этот разработчик долбоёб",
    )

    deepseek_mock.chat.assert_awaited_once()


def test_response_engine_keeps_mood_when_classifier_returns_none() -> None:
    """Неудачная классификация настроения не должна менять состояние."""
    (
        engine,
        state,
        _,
        _,
        mood_mock,
        _,
    ) = _create_engine()

    engine._user_states.get(TEST_USER_ID).mood = "sweet"
    mood_mock.classify.return_value = None

    asyncio.run(engine.respond(TEST_USER_ID, "Обычное сообщение"))

    assert engine._user_states.get(TEST_USER_ID).mood == "sweet"


def test_response_engine_adds_passive_role_to_rp_prompt() -> None:
    """Направление действия пользователя должно отражаться в RP-контексте."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        role_mock,
    ) = _create_engine()

    role_mock.classify.return_value = "passive"
    deepseek_mock.chat.return_value = "RP ответ"

    asyncio.run(engine.respond(TEST_USER_ID, "*связал тебя*"))

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert "бондаж" in prompt
    assert "которое совершает над Дельтой" in prompt


def test_response_engine_adds_emotional_context_to_prompt() -> None:
    """Выраженная реакция должна передаваться модели через промпт."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        mood_mock,
        _,
    ) = _create_engine()

    mood_mock.classify.return_value = "sweet"
    deepseek_mock.chat.return_value = "Ответ"

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Ты очень милый",
        )
    )

    assert result == "Ответ"

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt.startswith("SYSTEM PROMPT")
    assert "## Контекст текущего сообщения" in prompt
    assert "тёплую эмоциональную реакцию" in prompt


def test_response_engine_does_not_add_context_for_neutral_message() -> None:
    """Нейтральное сообщение не должно раздувать системный промпт."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        mood_mock,
        _,
    ) = _create_engine()

    mood_mock.classify.return_value = "neutral"
    deepseek_mock.chat.return_value = "Ответ"

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Как установить Docker?",
        )
    )

    assert result == "Ответ"

    assert deepseek_mock.chat.await_args is not None
    assert deepseek_mock.chat.await_args.kwargs["system_prompt"].startswith(
        "SYSTEM PROMPT"
    )
    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt=ANY,
        user_message="Как установить Docker?",
        history=(),
    )


def test_response_engine_rejects_empty_rp_prompt() -> None:
    """Пустой RP-промпт должен запрещать создание движка."""
    deepseek_mock = AsyncMock(spec=DeepSeekService)
    insult_mock = AsyncMock(spec=InsultClassifier)
    mood_mock = AsyncMock(spec=MoodClassifier)
    role_mock = AsyncMock(spec=FetishRoleClassifier)

    config = ResponseEngineConfig(
        fetish_triggers={},
        fetish_names={},
        system_prompt="SYSTEM",
        rp_prompt="   ",
    )

    with pytest.raises(
        ValueError,
        match="RP-промпт",
    ):
        ResponseEngine(
            deepseek=cast(
                DeepSeekService,
                deepseek_mock,
            ),
            insult_classifier=cast(
                InsultClassifier,
                insult_mock,
            ),
            mood_classifier=cast(
                MoodClassifier,
                mood_mock,
            ),
            fetish_role_classifier=cast(
                FetishRoleClassifier,
                role_mock,
            ),
            bot_state=BotState(),
            user_states=UserStateStore(),
            config=config,
        )


def test_response_engine_keeps_user_state_separate() -> None:
    """Состояния разных пользователей не должны влиять друг на друга."""
    (
        engine,
        state,
        deepseek_mock,
        _,
        mood_mock,
        _,
    ) = _create_engine()

    mood_mock.classify.side_effect = [
        "sweet",
        "angry",
    ]

    deepseek_mock.chat.return_value = "Ответ"

    first_result = asyncio.run(
        engine.respond(
            111,
            "Сообщение пользователя",
        )
    )

    second_result = asyncio.run(
        engine.respond(
            222,
            "Сообщение пользователя",
        )
    )

    assert first_result == "Ответ"
    assert second_result == "Ответ"

    assert engine._user_states.get(111).mood == "sweet"
    assert engine._user_states.get(222).mood == "angry"

    assert engine._user_states.get(111).reply_count == 1
    assert engine._user_states.get(222).reply_count == 1

    assert state.reply_count == 2


def test_response_engine_passes_user_history_to_deepseek() -> None:
    """Предыдущие ходы пользователя должны передаваться в DeepSeek."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    previous_turn = ConversationTurn(
        user_message="Меня зовут Экси",
        assistant_message="Запомнил",
    )

    user_state = engine._user_states.get(TEST_USER_ID)
    user_state.history.append(previous_turn)

    deepseek_mock.chat.return_value = "Конечно помню"

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Как меня зовут?",
        )
    )

    assert result == "Конечно помню"

    assert deepseek_mock.chat.await_args is not None
    assert deepseek_mock.chat.await_args.kwargs["system_prompt"].startswith(
        "SYSTEM PROMPT"
    )
    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt=ANY,
        user_message="Как меня зовут?",
        history=(previous_turn,),
    )

    assert list(user_state.history) == [
        previous_turn,
        ConversationTurn(
            user_message="Как меня зовут?",
            assistant_message="Конечно помню",
        ),
    ]


def test_response_engine_keeps_histories_isolated() -> None:
    """История одного пользователя не должна передаваться другому."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = [
        "Ответ первому",
        "Ответ второму",
    ]

    asyncio.run(
        engine.respond(
            111,
            "Сообщение первого",
        )
    )

    asyncio.run(
        engine.respond(
            222,
            "Сообщение второго",
        )
    )

    calls = deepseek_mock.chat.await_args_list

    assert calls[0].kwargs["history"] == ()
    assert calls[1].kwargs["history"] == ()

    first_state = engine._user_states.get(111)
    second_state = engine._user_states.get(222)

    assert list(first_state.history) == [
        ConversationTurn(
            user_message="Сообщение первого",
            assistant_message="Ответ первому",
        )
    ]

    assert list(second_state.history) == [
        ConversationTurn(
            user_message="Сообщение второго",
            assistant_message="Ответ второму",
        )
    ]


def test_response_engine_serializes_requests_from_same_user() -> None:
    """Запросы одного пользователя должны обрабатываться последовательно."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    async def run_requests() -> tuple[list[str], int]:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        chat_calls = 0

        async def chat_side_effect(
            system_prompt: str,
            user_message: str,
            history: tuple[ConversationTurn, ...],
        ) -> str:
            nonlocal chat_calls

            chat_calls += 1

            if user_message == "Первое сообщение":
                first_started.set()
                await release_first.wait()
                return "Первый ответ"

            return "Второй ответ"

        deepseek_mock.chat.side_effect = chat_side_effect

        first_task = asyncio.create_task(
            engine.respond(
                TEST_USER_ID,
                "Первое сообщение",
            )
        )

        await first_started.wait()

        second_task = asyncio.create_task(
            engine.respond(
                TEST_USER_ID,
                "Второе сообщение",
            )
        )

        # Даём второй coroutine возможность дойти до блокировки.
        await asyncio.sleep(0)

        calls_before_release = deepseek_mock.chat.await_count

        release_first.set()

        results = list(
            await asyncio.gather(
                first_task,
                second_task,
            )
        )

        return results, calls_before_release

    results, calls_before_release = asyncio.run(
        run_requests(),
    )

    assert calls_before_release == 1

    assert results == [
        "Первый ответ",
        "Второй ответ",
    ]

    calls = deepseek_mock.chat.await_args_list

    assert calls[0].kwargs["history"] == ()

    assert calls[1].kwargs["history"] == (
        ConversationTurn(
            user_message="Первое сообщение",
            assistant_message="Первый ответ",
        ),
    )

    assert list(engine._user_states.get(TEST_USER_ID).history) == [
        ConversationTurn(
            user_message="Первое сообщение",
            assistant_message="Первый ответ",
        ),
        ConversationTurn(
            user_message="Второе сообщение",
            assistant_message="Второй ответ",
        ),
    ]


def test_response_engine_resets_only_requested_user() -> None:
    """Сброс должен очищать состояние только выбранного пользователя."""
    (
        engine,
        _,
        _,
        _,
        _,
        _,
    ) = _create_engine()

    first = engine._user_states.get(111)
    second = engine._user_states.get(222)

    first.mood = "sweet"
    first.reply_count = 3
    first.history.append(
        ConversationTurn(
            user_message="Первое",
            assistant_message="Ответ первому",
        )
    )

    second.mood = "angry"
    second.reply_count = 2
    second.history.append(
        ConversationTurn(
            user_message="Второе",
            assistant_message="Ответ второму",
        )
    )

    asyncio.run(
        engine.reset_user_context(111),
    )

    assert first.mood == "neutral"
    assert first.reply_count == 0
    assert list(first.history) == []

    assert second.mood == "angry"
    assert second.reply_count == 2
    assert list(second.history) == [
        ConversationTurn(
            user_message="Второе",
            assistant_message="Ответ второму",
        )
    ]


def test_response_engine_reset_waits_for_active_request() -> None:
    """Сброс должен дождаться активного запроса пользователя."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    async def run_test() -> tuple[str, bool]:
        request_started = asyncio.Event()
        release_request = asyncio.Event()

        async def chat_side_effect(
            system_prompt: str,
            user_message: str,
            history: tuple[ConversationTurn, ...],
        ) -> str:
            request_started.set()
            await release_request.wait()
            return "Ответ"

        deepseek_mock.chat.side_effect = chat_side_effect

        response_task = asyncio.create_task(
            engine.respond(
                TEST_USER_ID,
                "Сообщение",
            )
        )

        await request_started.wait()

        reset_task = asyncio.create_task(
            engine.reset_user_context(
                TEST_USER_ID,
            )
        )

        await asyncio.sleep(0)

        reset_finished_while_request_active = reset_task.done()

        release_request.set()

        response = await response_task
        await reset_task

        return (
            response,
            reset_finished_while_request_active,
        )

    response, reset_finished_while_request_active = asyncio.run(
        run_test(),
    )

    assert response == "Ответ"
    assert reset_finished_while_request_active is False

    user_state = engine._user_states.get(TEST_USER_ID)

    assert user_state.mood == "neutral"
    assert user_state.reply_count == 0
    assert list(user_state.history) == []


def test_response_engine_fully_resets_user() -> None:
    """Полный сброс должен очищать всё состояние выбранного пользователя."""
    (
        engine,
        _,
        _,
        _,
        _,
        _,
    ) = _create_engine()

    state = engine._user_states.get(TEST_USER_ID)

    state.mood = "angry"
    state.reply_count = 4
    state.roleplay_active = True

    state.history.append(
        ConversationTurn(
            user_message="Сообщение",
            assistant_message="Ответ",
        )
    )

    state.emotions.adjust(
        irritation=0.6,
        warmth=0.3,
    )

    state.relationship.adjust(
        familiarity=0.8,
        trust=0.7,
        affection=0.5,
        resentment=0.4,
    )

    asyncio.run(
        engine.reset_user(TEST_USER_ID),
    )

    assert state.mood == "neutral"
    assert state.reply_count == 0
    assert list(state.history) == []
    assert state.roleplay_active is False

    assert state.emotions.warmth == 0.0
    assert state.emotions.irritation == 0.0

    assert state.relationship.familiarity == 0.0
    assert state.relationship.trust == 0.0
    assert state.relationship.affection == 0.0
    assert state.relationship.resentment == 0.0


def test_response_engine_adds_horny_context_to_prompt() -> None:
    """Сексуальная реакция должна передаваться основной модели."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        mood_mock,
        _,
    ) = _create_engine()

    mood_mock.classify.return_value = "horny"
    deepseek_mock.chat.return_value = "Ответ"

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Явно сексуальное сообщение",
        )
    )

    assert result == "Ответ"

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt.startswith("SYSTEM PROMPT")
    assert "сексуальный или возбуждающий контекст" in prompt


def test_response_engine_accumulates_interaction_state() -> None:
    """Движок должен применять накопительные последствия каждого сообщения."""
    (
        engine,
        _,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    mood_mock.classify.side_effect = [
        "sweet",
        None,
    ]

    insult_mock.classify.side_effect = [
        "none",
        "none",
    ]

    deepseek_mock.chat.side_effect = [
        "Первый ответ",
        "Второй ответ",
    ]

    asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Ты милый",
        )
    )

    state = engine._user_states.get(TEST_USER_ID)

    assert state.emotions.warmth == pytest.approx(0.10)
    assert state.relationship.affection == pytest.approx(0.03)

    asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Не знаю",
        )
    )

    assert state.emotions.warmth == pytest.approx(0.10)
    assert state.relationship.affection == pytest.approx(0.03)
    assert state.relationship.familiarity == pytest.approx(0.02)


def test_response_engine_adds_persistent_state_context() -> None:
    """Накопленное отношение должно передаваться основной модели."""
    (
        engine,
        _,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    state = engine._user_states.get(TEST_USER_ID)

    state.relationship.adjust(
        affection=0.30,
        resentment=0.20,
    )
    state.emotions.adjust(
        irritation=0.30,
    )

    insult_mock.classify.return_value = "none"
    mood_mock.classify.return_value = "neutral"
    deepseek_mock.chat.return_value = "Ответ"

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Обычное сообщение",
        )
    )

    assert result == "Ответ"

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert "сформировалась заметная симпатия" in prompt
    assert "остался некоторый осадок" in prompt
    assert "сохраняется лёгкое раздражение" in prompt


def test_response_engine_runs_message_classifiers_in_parallel() -> None:
    """Классификаторы сообщения должны запускаться параллельно."""
    (
        engine,
        _,
        _,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    async def run_test() -> tuple[str, bool]:
        insult_started = asyncio.Event()
        mood_started = asyncio.Event()
        release_classifiers = asyncio.Event()

        async def classify_insult(
            user_message: str,
        ) -> str:
            assert user_message == "Обычное сообщение"

            insult_started.set()
            await release_classifiers.wait()

            return "none"

        async def classify_mood(
            user_message: str,
        ) -> str:
            assert user_message == "Обычное сообщение"

            mood_started.set()
            await release_classifiers.wait()

            return "neutral"

        insult_mock.classify.side_effect = classify_insult
        mood_mock.classify.side_effect = classify_mood

        response_task = asyncio.create_task(
            engine.respond(
                TEST_USER_ID,
                "Обычное сообщение",
            )
        )

        await insult_started.wait()
        await asyncio.sleep(0)

        classifiers_started_together = mood_started.is_set()

        release_classifiers.set()

        response = await response_task

        return (
            response,
            classifiers_started_together,
        )

    response, classifiers_started_together = asyncio.run(
        run_test(),
    )

    assert response == "Ответ"
    assert classifiers_started_together is True

    insult_mock.classify.assert_awaited_once_with(
        "Обычное сообщение",
    )
    mood_mock.classify.assert_awaited_once_with(
        "Обычное сообщение",
    )


def test_response_engine_keeps_roleplay_active_for_follow_up() -> None:
    """После RP-действия следующие сообщения должны продолжать RP-режим."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = [
        "RP ответ",
        "Продолжение RP",
    ]

    first_result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "*обнял тебя*",
        )
    )

    second_result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Что будешь делать дальше?",
        )
    )

    assert first_result == "RP ответ"
    assert second_result == "Продолжение RP"

    state = engine._user_states.get(TEST_USER_ID)

    assert state.roleplay_active is True

    assert deepseek_mock.chat.await_count == 2

    second_call = deepseek_mock.chat.await_args_list[1]

    assert second_call.kwargs["system_prompt"].startswith("RP PROMPT")

    assert second_call.kwargs["history"] == (
        ConversationTurn(
            user_message="*обнял тебя*",
            assistant_message="RP ответ",
        ),
    )


def test_response_engine_reset_disables_roleplay() -> None:
    """Сброс контекста должен завершать активный RP-режим."""
    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.side_effect = [
        "RP ответ",
        "Обычный ответ",
    ]

    asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "*обнял тебя*",
        )
    )

    state = engine._user_states.get(TEST_USER_ID)

    assert state.roleplay_active is True

    asyncio.run(
        engine.reset_user_context(TEST_USER_ID),
    )

    assert state.roleplay_active is False
    assert list(state.history) == []

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Привет",
        )
    )

    assert result == "Обычный ответ"

    last_call = deepseek_mock.chat.await_args

    assert last_call is not None
    assert last_call.kwargs["system_prompt"].startswith("SYSTEM PROMPT")
    assert "RP выключен" in last_call.kwargs["system_prompt"]
    assert last_call.kwargs["history"] == ()


def test_response_engine_disables_only_roleplay() -> None:
    """Выход из RP не должен сбрасывать остальное состояние пользователя."""
    (
        engine,
        _,
        _,
        _,
        _,
        _,
    ) = _create_engine()

    state = engine._user_states.get(TEST_USER_ID)

    state.mood = "sweet"
    state.reply_count = 3
    state.roleplay_active = True

    state.history.append(
        ConversationTurn(
            user_message="*подошёл ближе*",
            assistant_message="RP ответ",
        )
    )

    state.emotions.adjust(
        warmth=0.4,
        irritation=0.2,
    )

    state.relationship.adjust(
        familiarity=0.5,
        trust=0.4,
        affection=0.3,
        resentment=0.1,
    )

    was_active = asyncio.run(
        engine.disable_roleplay(TEST_USER_ID),
    )

    assert was_active is True
    assert state.roleplay_active is False

    assert state.mood == "sweet"
    assert state.reply_count == 3

    assert list(state.history) == [
        ConversationTurn(
            user_message="*подошёл ближе*",
            assistant_message="RP ответ",
        )
    ]

    assert state.emotions.warmth == pytest.approx(0.4)
    assert state.emotions.irritation == pytest.approx(0.2)

    assert state.relationship.familiarity == pytest.approx(0.5)
    assert state.relationship.trust == pytest.approx(0.4)
    assert state.relationship.affection == pytest.approx(0.3)
    assert state.relationship.resentment == pytest.approx(0.1)


def test_response_engine_reports_inactive_roleplay() -> None:
    """Повторное выключение RP должно сообщать, что режим уже неактивен."""
    (
        engine,
        _,
        _,
        _,
        _,
        _,
    ) = _create_engine()

    was_active = asyncio.run(
        engine.disable_roleplay(TEST_USER_ID),
    )

    assert was_active is False
    assert engine._user_states.get(TEST_USER_ID).roleplay_active is False


def test_response_engine_binds_same_log_context_for_request() -> None:
    """Параллельные части одного ответа должны иметь общий request_id."""
    (
        engine,
        _,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    observed_contexts: list[tuple[str, str]] = []

    def capture_context() -> None:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        LogContextFilter().filter(record)

        observed_contexts.append(
            (
                getattr(record, "user_id"),
                getattr(record, "request_id"),
            )
        )

    async def classify_insult(_: str) -> str:
        capture_context()
        return "none"

    async def classify_mood(_: str) -> str:
        capture_context()
        return "neutral"

    async def chat_with_context(**_: object) -> str:
        capture_context()
        return "Ответ"

    insult_mock.classify.side_effect = classify_insult
    mood_mock.classify.side_effect = classify_mood
    deepseek_mock.chat.side_effect = chat_with_context

    result = asyncio.run(
        engine.respond(
            TEST_USER_ID,
            "Привет",
        )
    )

    assert result == "Ответ"
    assert len(observed_contexts) == 3

    user_ids = {user_id for user_id, _ in observed_contexts}
    request_ids = {request_id for _, request_id in observed_contexts}

    assert user_ids == {str(TEST_USER_ID)}
    assert len(request_ids) == 1
    assert "-" not in request_ids


def test_mixed_stop_answers_question_without_restarting_scene() -> None:
    engine, _, deepseek, _, _, _ = _create_engine()
    state = engine._user_states.get(TEST_USER_ID)
    state.roleplay_active = True
    state.roleplay_configuration = "female"
    state.roleplay_character = "человек"
    state.relationship.trust = 0.7
    state.history.append(ConversationTurn("*подхожу*", "*подняла голову*"))
    asyncio.run(engine.respond(TEST_USER_ID, "Стоп RP, объясни *TCP*"))
    assert state.roleplay_active is False
    assert state.roleplay_configuration == "male"
    assert state.roleplay_character == ""
    assert state.relationship.trust == pytest.approx(0.7)
    call = deepseek.chat.await_args
    assert call is not None
    assert call.kwargs["user_message"] == "объясни *TCP*"
    assert "RP выключен" in call.kwargs["system_prompt"]
    assert "женская" not in call.kwargs["system_prompt"]
    assert len(call.kwargs["history"]) == 1


def test_scene_configuration_survives_history_window_and_is_isolated() -> None:
    engine, _, deepseek, _, _, _ = _create_engine()

    async def run() -> None:
        await engine.respond(TEST_USER_ID, "Ты в женской конфигурации.")
        await engine.respond(TEST_USER_ID, "Мой персонаж: человек в пальто")
        for _ in range(10):
            await engine.respond(TEST_USER_ID, "Продолжай")
        call = deepseek.chat.await_args
        assert call is not None
        prompt = call.kwargs["system_prompt"]
        assert "женская; говори о себе в женском роде" in prompt
        assert "человек в пальто" in prompt
        assert all(
            "конфигурации" not in turn.user_message for turn in call.kwargs["history"]
        )
        await engine.respond(TEST_USER_ID + 1, "*приветствую*")
        other = deepseek.chat.await_args
        assert other is not None
        assert "женская" not in other.kwargs["system_prompt"]
        assert "человек в пальто" not in other.kwargs["system_prompt"]
        await engine.disable_roleplay(TEST_USER_ID)
        state = engine._user_states.get(TEST_USER_ID)
        assert state.roleplay_configuration == "male"
        assert state.roleplay_character == ""

    asyncio.run(run())


def test_direct_stop_does_not_call_model() -> None:
    engine, _, deepseek, _, _, _ = _create_engine()
    engine._user_states.get(TEST_USER_ID).roleplay_active = True
    assert asyncio.run(engine.respond(TEST_USER_ID, "стоп RP")) == "RP-режим завершён."
    deepseek.chat.assert_not_awaited()


def test_delivery_commits_history_only_after_all_chunks() -> None:
    engine, state, deepseek, *_ = _create_engine()
    deepseek.chat.return_value = "a" * 5000
    sent: list[str] = []

    async def deliver(reply: str) -> None:
        for chunk in split_message(reply):
            assert not engine._user_states.get(TEST_USER_ID).history
            assert state.reply_count == 0
            sent.append(chunk)
            await asyncio.sleep(0)

    asyncio.run(engine.respond_and_deliver(TEST_USER_ID, "Привет", deliver))
    assert len(sent) == 2
    assert state.reply_count == 1
    assert list(engine._user_states.get(TEST_USER_ID).history) == [
        ConversationTurn("Привет", "a" * 5000)
    ]


@pytest.mark.parametrize("failed_chunk", [0, 1])
def test_failed_delivery_does_not_commit_and_allows_retry(failed_chunk: int) -> None:
    engine, state, deepseek, *_ = _create_engine()
    deepseek.chat.return_value = "a" * 5000
    sent: list[str] = []

    async def deliver(reply: str) -> None:
        for index, chunk in enumerate(split_message(reply)):
            if index == failed_chunk:
                raise RuntimeError("delivery failed")
            sent.append(chunk)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="delivery failed"):
            await engine.respond_and_deliver(TEST_USER_ID, "Привет", deliver)
        assert len(sent) == failed_chunk
        assert not engine._user_states.get(TEST_USER_ID).history
        assert engine._user_states.get(TEST_USER_ID).reply_count == 0
        assert state.reply_count == 0
        await engine.respond_and_deliver(TEST_USER_ID, "Ещё раз", AsyncMock())
        assert state.reply_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("blocked_stage", ["generation", "delivery"])
def test_busy_request_is_rejected_without_blocking_other_users(
    blocked_stage: str,
) -> None:
    engine, _, deepseek, *_ = _create_engine()

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs: object) -> str:
            if blocked_stage == "generation" and kwargs["user_message"] == "Первый":
                entered.set()
                await release.wait()
            return "Ответ"

        async def deliver(reply: str) -> None:
            if blocked_stage == "delivery":
                entered.set()
                await release.wait()

        deepseek.chat.side_effect = generate
        first = asyncio.create_task(
            engine.respond_and_deliver(TEST_USER_ID, "Первый", deliver)
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            with pytest.raises(ResponseBusyError):
                await engine.respond_and_deliver(TEST_USER_ID, "Повтор", AsyncMock())
            await asyncio.wait_for(
                engine.respond_and_deliver(999, "Другой пользователь", AsyncMock()),
                timeout=1,
            )
            assert deepseek.chat.await_count == 2
        finally:
            release.set()
            await first

    asyncio.run(scenario())


@pytest.mark.parametrize("reset_method", ["reset_user", "reset_user_context"])
def test_reset_waits_until_delivery_finishes(reset_method: str) -> None:
    engine, _, _, *_ = _create_engine()

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        events: list[str] = []

        async def deliver(reply: str) -> None:
            entered.set()
            await release.wait()
            events.append("sent")

        first = asyncio.create_task(
            engine.respond_and_deliver(TEST_USER_ID, "Привет", deliver)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)

        async def reset() -> None:
            await getattr(engine, reset_method)(TEST_USER_ID)
            events.append("reset")

        reset_task = asyncio.create_task(reset())
        await asyncio.sleep(0)
        assert not reset_task.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, reset_task), timeout=1)
        assert events == ["sent", "reset"]
        assert not engine._user_states.get(TEST_USER_ID).history

    asyncio.run(scenario())


@pytest.mark.parametrize("blocked_stage", ["generation", "delivery"])
def test_cancelled_request_releases_busy_guard_and_state(blocked_stage: str) -> None:
    engine, state, deepseek, *_ = _create_engine()

    async def scenario() -> None:
        entered = asyncio.Event()

        async def block(*args: object, **kwargs: object) -> None:
            entered.set()
            await asyncio.Event().wait()

        if blocked_stage == "generation":
            deepseek.chat.side_effect = block
        task = asyncio.create_task(
            engine.respond_and_deliver(TEST_USER_ID, "Привет", block)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        user_state = engine._user_states.get(TEST_USER_ID)
        assert not user_state.history
        assert not user_state.lock.locked()
        assert user_state.active_operations == 0
        assert state.reply_count == 0
        deepseek.chat.side_effect = None
        await engine.respond_and_deliver(TEST_USER_ID, "Повтор", AsyncMock())
        assert state.reply_count == 1

    asyncio.run(scenario())


def test_api_error_fallback_is_delivered_without_history_entry() -> None:
    engine, state, deepseek, *_ = _create_engine()
    deepseek.chat.side_effect = DeepSeekTimeoutError("timeout")
    deliver = AsyncMock()
    asyncio.run(engine.respond_and_deliver(TEST_USER_ID, "Привет", deliver))
    deliver.assert_awaited_once()
    assert deliver.await_args is not None
    assert "долго думаю" in deliver.await_args.args[0]
    assert not engine._user_states.get(TEST_USER_ID).history
    assert state.reply_count == 0


def test_delivery_keeps_generation_log_context() -> None:
    engine, _, deepseek, *_ = _create_engine()
    contexts: list[tuple[str, str]] = []

    def capture() -> None:
        record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)
        LogContextFilter().filter(record)
        contexts.append((getattr(record, "user_id"), getattr(record, "request_id")))

    async def generate(**kwargs: object) -> str:
        capture()
        return "Ответ"

    async def deliver(reply: str) -> None:
        capture()

    deepseek.chat.side_effect = generate
    asyncio.run(engine.respond_and_deliver(TEST_USER_ID, "Привет", deliver))
    assert contexts[0] == contexts[1]
    assert contexts[0][0] == str(TEST_USER_ID)
    assert contexts[0][1] != "-"
