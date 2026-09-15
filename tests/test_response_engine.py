"""Тесты движка формирования ответов."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock

import pytest

import protogen_delta.services.response_engine as response_engine_module
from protogen_delta.core.state import BotState
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
    mood_mock.classify.return_value = "playful"
    role_mock.classify.return_value = "unknown"

    state = BotState()

    config = ResponseEngineConfig(
        greetings=["Приветик"],
        insults=["Отвали"],
        question_insult_replies=["Сам такой вопрос задаёшь?"],
        horny_replies=["Horny reply"],
        moods={
            "playful": [],
            "sweet": [],
            "horny": [],
            "angry": [],
        },
        fetish_triggers={
            "bondage": ["связал"],
        },
        fetish_names={
            "bondage": "бондаж",
        },
        emote_categories={
            "NORMAL": ["UwU"],
            "BLUSH": [">///<"],
            "INSULT": [">:("],
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


def test_response_engine_returns_greeting_without_deepseek() -> None:
    """Одиночное приветствие не должно отправляться в DeepSeek."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        _,
        _,
    ) = _create_engine()

    result = asyncio.run(
        engine.respond("Привет!"),
    )

    assert result == "Приветик"
    assert state.reply_count == 1

    insult_mock.classify.assert_not_awaited()
    deepseek_mock.chat.assert_not_awaited()


def test_response_engine_returns_direct_insult_without_chat() -> None:
    """Прямое оскорбление должно получать быстрый локальный ответ."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    insult_mock.classify.return_value = "direct"

    result = asyncio.run(
        engine.respond("Ты идиот"),
    )

    assert result == "Отвали >:("
    assert state.reply_count == 1

    mood_mock.classify.assert_not_awaited()
    deepseek_mock.chat.assert_not_awaited()


def test_response_engine_updates_mood_and_calls_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обычное сообщение должно обновить настроение и вызвать DeepSeek."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

    (
        engine,
        state,
        deepseek_mock,
        _,
        mood_mock,
        role_mock,
    ) = _create_engine()

    mood_mock.classify.return_value = "sweet"
    deepseek_mock.chat.return_value = "Обычный ответ"

    result = asyncio.run(
        engine.respond("Как дела?"),
    )

    assert result == "Обычный ответ"
    assert state.mood == "sweet"
    assert state.reply_count == 1

    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
        user_message="Как дела?",
    )

    role_mock.classify.assert_not_awaited()


def test_response_engine_uses_rp_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RP-сообщение должно использовать RP-промпт."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("*обнял тебя*"),
    )

    assert result == "RP ответ"
    assert state.reply_count == 1

    role_mock.classify.assert_not_awaited()

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert prompt.startswith("RP PROMPT")
    assert "ВАЖНО" in prompt
    assert "без одежды" in prompt


def test_response_engine_adds_fetish_context_and_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Фетиш и роль должны добавляться в динамический RP-промпт."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("*связал тебя*"),
    )

    assert result == "RP ответ"
    assert state.reply_count == 1

    role_mock.classify.assert_awaited_once_with(
        "*связал тебя*",
    )

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert "бондаж (бот актив)" in prompt
    assert "Отыгрывай их в ролевке" in prompt
    assert "активного партнёра" in prompt


def test_response_engine_does_not_classify_role_without_fetish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без найденного фетиша отдельная классификация роли не нужна."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

    (
        engine,
        _,
        _,
        _,
        _,
        role_mock,
    ) = _create_engine()

    asyncio.run(
        engine.respond("*погладил тебя*"),
    )

    role_mock.classify.assert_not_awaited()


def test_response_engine_returns_fallback_on_chat_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка обычного запроса DeepSeek должна дать аварийный ответ."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Как дела"),
    )

    assert result == "Бля, у тостера что-то сломалось... ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout DeepSeek должен возвращать понятный ответ пользователю."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Как дела"),
    )

    assert result == "Я чёт завис и слишком долго думаю... попробуй ещё раз ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Лимит запросов DeepSeek должен возвращать отдельный ответ."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Как дела"),
    )

    assert result == (
        "Меня сейчас слишком сильно дёргают запросами... " "дай мне немного времени ≧◡≦"
    )
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка соединения с DeepSeek должна возвращать отдельный ответ."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Как дела"),
    )

    assert result == "У меня отвалилось соединение... попробуй чуть позже ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка доступа DeepSeek не должна раскрывать технические детали."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Как дела"),
    )

    assert result == "У меня какая-то внутренняя хуйня сломалась... попробуй позже ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_deepseek_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка API DeepSeek должна возвращать безопасный ответ."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Как дела"),
    )

    assert result == "У меня мозги сейчас чудят... попробуй чуть позже ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_handles_unknown_deepseek_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Неизвестная ошибка DeepSeek должна использовать общий fallback."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Как дела"),
    )

    assert result == "Бля, у тостера что-то сломалось... ≧◡≦"
    assert state.reply_count == 0


def test_response_engine_does_not_classify_fetish_role_outside_rp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обычный текст с fetish-триггером не должен вызывать классификатор роли."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Ты меня связал?"),
    )

    assert result == "Обычный ответ"

    role_mock.classify.assert_not_awaited()

    deepseek_mock.chat.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
        user_message="Ты меня связал?",
    )


def test_response_engine_handles_empty_deepseek_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой ответ DeepSeek должен заменяться понятным сообщением."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Обычное сообщение"),
    )

    assert result == "Пустой ответ от DeepSeek"
    assert state.reply_count == 1


def test_response_engine_handles_whitespace_deepseek_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ответ только из пробелов должен считаться молчанием модели."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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
        engine.respond("Обычное сообщение"),
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
        greetings=[],
        insults=[],
        question_insult_replies=[],
        horny_replies=[],
        moods={},
        fetish_triggers={},
        fetish_names={},
        emote_categories={},
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
            config=config,
        )


def test_response_engine_returns_question_insult_reply() -> None:
    """Вопросительное оскорбление должно получать отдельный ответ."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    insult_mock.classify.return_value = "question"

    result = asyncio.run(engine.respond("Ты совсем тупой?"))

    assert result == "Сам такой вопрос задаёшь? >///<"
    assert state.reply_count == 1

    mood_mock.classify.assert_not_awaited()
    deepseek_mock.chat.assert_not_awaited()


def test_response_engine_returns_general_insult_reply() -> None:
    """Общее оскорбление должно получать локальный ответ."""
    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    insult_mock.classify.return_value = "general"

    result = asyncio.run(engine.respond("Вот же идиотизм"))

    assert result == "Отвали"
    assert state.reply_count == 1

    mood_mock.classify.assert_not_awaited()
    deepseek_mock.chat.assert_not_awaited()


def test_response_engine_continues_when_greetings_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без локальных приветствий сообщение должно идти в DeepSeek."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

    (
        engine,
        state,
        deepseek_mock,
        insult_mock,
        mood_mock,
        _,
    ) = _create_engine()

    engine._config.greetings.clear()

    deepseek_mock.chat.return_value = "Ответ модели"

    result = asyncio.run(engine.respond("Привет!"))

    assert result == "Ответ модели"
    assert state.reply_count == 1

    insult_mock.classify.assert_awaited_once_with(
        "Привет!",
    )
    mood_mock.classify.assert_awaited_once_with(
        "Привет!",
    )


def test_response_engine_keeps_mood_when_classifier_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Неудачная классификация настроения не должна менять состояние."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

    (
        engine,
        state,
        _,
        _,
        mood_mock,
        _,
    ) = _create_engine()

    state.mood = "sweet"
    mood_mock.classify.return_value = None

    asyncio.run(engine.respond("Обычное сообщение"))

    assert state.mood == "sweet"


def test_response_engine_adds_passive_role_to_rp_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пассивная роль должна отражаться в динамическом RP-промпте."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 1.0,
    )

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

    asyncio.run(engine.respond("*связал тебя*"))

    call = deepseek_mock.chat.await_args

    assert call is not None

    prompt = call.kwargs["system_prompt"]

    assert "бондаж (бот пассив)" in prompt
    assert "пассивного партнёра" in prompt


def test_response_engine_adds_mood_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Реплика текущего настроения должна иногда добавляться к ответу."""
    random_values = iter(
        [
            0.0,
            1.0,
        ]
    )

    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: next(random_values),
    )
    monkeypatch.setattr(
        response_engine_module.random,
        "choice",
        lambda values: values[0],
    )

    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    engine._config.moods["playful"] = [
        "Mood reply",
    ]

    deepseek_mock.chat.return_value = "Ответ"

    result = asyncio.run(engine.respond("Обычное сообщение"))

    assert result == "Ответ\n\nMood reply"


def test_response_engine_adds_fetish_tease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RP с фетишем должен иногда получать дополнительную teasing-реплику."""
    random_values = iter(
        [
            0.0,
            1.0,
        ]
    )

    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: next(random_values),
    )
    monkeypatch.setattr(
        response_engine_module.random,
        "choice",
        lambda values: values[0],
    )

    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        role_mock,
    ) = _create_engine()

    role_mock.classify.return_value = "unknown"
    deepseek_mock.chat.return_value = "RP ответ"

    result = asyncio.run(engine.respond("*связал тебя*"))

    assert result == ("RP ответ\n\n" "Ммм, похоже ты любишь темы: бондаж… ^w^")


def test_response_engine_adds_rp_horny_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """После первого ответа RP может получить дополнительную horny-реплику."""
    random_values = iter(
        [
            0.0,
            1.0,
        ]
    )

    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: next(random_values),
    )
    monkeypatch.setattr(
        response_engine_module.random,
        "choice",
        lambda values: values[0],
    )

    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    state.reply_count = 1
    deepseek_mock.chat.return_value = "RP ответ"

    result = asyncio.run(engine.respond("*обнял тебя*"))

    assert result == "Horny reply\n\nRP ответ"
    assert state.reply_count == 2


def test_response_engine_adds_non_rp_horny_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обычный диалог после нескольких ответов может получить horny-реплику."""
    random_values = iter(
        [
            0.0,
            1.0,
        ]
    )

    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: next(random_values),
    )
    monkeypatch.setattr(
        response_engine_module.random,
        "choice",
        lambda values: values[0],
    )

    (
        engine,
        state,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    state.reply_count = 3
    deepseek_mock.chat.return_value = "Ответ"

    result = asyncio.run(engine.respond("Обычное сообщение"))

    assert result == "Ответ\n\nHorny reply"
    assert state.reply_count == 4


def test_response_engine_adds_normal_emote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обычный эмоут должен иногда добавляться в конец ответа."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 0.0,
    )
    monkeypatch.setattr(
        response_engine_module.random,
        "choice",
        lambda values: values[0],
    )

    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.return_value = "Ответ"

    result = asyncio.run(engine.respond("Обычное сообщение"))

    assert result == "Ответ UwU"


def test_response_engine_does_not_duplicate_emote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Уже существующий эмоут не должен добавляться повторно."""
    monkeypatch.setattr(
        response_engine_module.random,
        "random",
        lambda: 0.0,
    )

    (
        engine,
        _,
        deepseek_mock,
        _,
        _,
        _,
    ) = _create_engine()

    deepseek_mock.chat.return_value = "Ответ UwU"

    result = asyncio.run(engine.respond("Обычное сообщение"))

    assert result == "Ответ UwU"


def test_response_engine_rejects_empty_rp_prompt() -> None:
    """Пустой RP-промпт должен запрещать создание движка."""
    deepseek_mock = AsyncMock(spec=DeepSeekService)
    insult_mock = AsyncMock(spec=InsultClassifier)
    mood_mock = AsyncMock(spec=MoodClassifier)
    role_mock = AsyncMock(spec=FetishRoleClassifier)

    config = ResponseEngineConfig(
        greetings=[],
        insults=[],
        question_insult_replies=[],
        horny_replies=[],
        moods={},
        fetish_triggers={},
        fetish_names={},
        emote_categories={},
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
            config=config,
        )
