"""Тесты движка формирования ответов."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock

import pytest

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
    user_states = UserStateStore()

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

    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
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

    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
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

    role_mock.classify.assert_not_awaited()

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt == "RP PROMPT"
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

    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
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

    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
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

    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
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
