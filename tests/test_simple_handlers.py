"""Тесты простых Telegram-обработчиков."""

import asyncio
import logging
from typing import cast
from unittest.mock import ANY, AsyncMock, Mock

import pytest
from aiogram import Bot, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

import protogen_delta.handlers.unknown_command as unknown_command_module
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.help import HELP_TEXT, create_help_router
from protogen_delta.handlers.rp import (
    RP_ALREADY_DISABLED_REPLY,
    RP_DISABLED_REPLY,
)
from protogen_delta.handlers.text import (
    BUSY_REPLY,
    RATE_LIMIT_REPLY,
    create_text_router,
)
from protogen_delta.handlers.unknown_command import (
    UNKNOWN_COMMAND_REPLIES,
    create_unknown_command_router,
)
from protogen_delta.services.response_engine import (
    ReplyDelivery,
    ResponseBusyError,
    ResponseEngine,
)

TEST_USER_ID = 123456


def _create_engine_mock() -> AsyncMock:
    engine = AsyncMock(spec=ResponseEngine)

    async def respond_and_deliver(
        user_id: int, text: str, deliver: ReplyDelivery
    ) -> None:
        await deliver(engine.respond_and_deliver.return_value)

    engine.respond_and_deliver.side_effect = respond_and_deliver
    return engine


def _create_message_mock(
    text: str | None = None,
    user_id: int | None = TEST_USER_ID,
) -> tuple[Message, AsyncMock, AsyncMock]:
    """Создать Message с асинхронными методами answer и reply."""
    message_mock = Mock(spec=Message)

    message_mock.text = text
    message_mock.chat = Mock(id=321, type="private")
    message_mock.forward_origin = None

    if user_id is None:
        message_mock.from_user = None
    else:
        user_mock = Mock()
        user_mock.id = user_id
        message_mock.from_user = user_mock

    answer_mock = AsyncMock()
    reply_mock = AsyncMock()

    message_mock.answer = answer_mock
    message_mock.reply = reply_mock

    return (
        cast(Message, message_mock),
        answer_mock,
        reply_mock,
    )


async def _call_first_handler(
    router: Router,
    message: Message,
) -> None:
    """Вызвать первый зарегистрированный message-handler роутера."""
    handler = router.message.handlers[0]

    await handler.callback(message)


def test_help_handler_sends_help_text() -> None:
    """Команда /help должна отправлять текст справки."""
    router = create_help_router()

    message, answer_mock, _ = _create_message_mock()

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        HELP_TEXT.replace("/randomart – 🎨 Случайный арт\n", ""),
    )


def test_unknown_command_handler_sends_known_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Неизвестная команда должна получать одну из стандартных реплик."""
    router = create_unknown_command_router()

    monkeypatch.setattr(
        unknown_command_module.random,
        "choice",
        lambda values: values[0],
    )

    message, _, reply_mock = _create_message_mock()

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    reply_mock.assert_awaited_once_with(
        UNKNOWN_COMMAND_REPLIES[0],
    )


def test_text_handler_calls_response_engine() -> None:
    """Обычный текст должен передаваться движку ответов."""
    engine_mock = _create_engine_mock()
    engine_mock.respond_and_deliver.return_value = "Ответ бота"

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock, _ = _create_message_mock(
        "Привет, как дела?",
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.respond_and_deliver.assert_awaited_once_with(
        TEST_USER_ID,
        "Привет, как дела?",
        ANY,
    )
    answer_mock.assert_awaited_once_with(
        "Ответ бота",
    )


def test_text_handler_splits_long_response() -> None:
    """Длинный ответ должен отправляться несколькими сообщениями."""
    engine_mock = _create_engine_mock()
    engine_mock.respond_and_deliver.return_value = "a" * 5000

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock, _ = _create_message_mock(
        "Сообщение",
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.respond_and_deliver.assert_awaited_once_with(
        TEST_USER_ID,
        "Сообщение",
        ANY,
    )

    assert answer_mock.await_count == 2


def test_text_handler_sends_paragraphs_with_typing_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Многоабзацный ответ должен выглядеть как последовательность живых реплик."""
    engine = _create_engine_mock()
    first = "Первый содержательный абзац. " * 10
    second = "Второй абзац немного подлиннее. " * 10
    engine.respond_and_deliver.return_value = f"{first}\n\n{second}"
    bot = AsyncMock(spec=Bot)
    sleep = AsyncMock()
    monkeypatch.setattr("protogen_delta.handlers.delivery.sleep", sleep)
    router = create_text_router(cast(ResponseEngine, engine), bot=cast(Bot, bot))
    message, answer, _ = _create_message_mock("Расскажи подробнее")
    message.chat = Mock(id=321, type="private")

    asyncio.run(_call_first_handler(router, message))

    assert [call.args[0] for call in answer.await_args_list] == [
        first.strip(),
        second.strip(),
    ]
    bot.send_chat_action.assert_awaited_once_with(
        chat_id=321,
        action=ChatAction.TYPING,
    )
    sleep.assert_awaited_once()
    sleep_call = sleep.await_args
    assert sleep_call is not None
    assert 1.5 <= sleep_call.args[0] <= 2.5


def test_text_handler_ignores_commands() -> None:
    """Команды не должны попадать в обычный текстовый движок."""
    engine_mock = _create_engine_mock()

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock, _ = _create_message_mock(
        "/something",
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.respond_and_deliver.assert_not_awaited()
    answer_mock.assert_not_awaited()


def test_text_handler_ignores_missing_text() -> None:
    """Сообщение без текста не должно обрабатываться."""
    engine_mock = _create_engine_mock()

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock, _ = _create_message_mock(None)

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.respond_and_deliver.assert_not_awaited()
    answer_mock.assert_not_awaited()


def test_text_handler_ignores_message_without_user() -> None:
    """Текст без Telegram-пользователя не должен попадать в движок."""
    engine_mock = _create_engine_mock()

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock, _ = _create_message_mock(
        "Сообщение",
        user_id=None,
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.respond_and_deliver.assert_not_awaited()
    answer_mock.assert_not_awaited()


def test_text_handler_rate_limits_repeated_messages() -> None:
    """Повторное сообщение во время cooldown не должно вызывать движок."""
    times = iter(
        [
            100.0,
            100.5,
        ]
    )

    rate_limiter = UserRateLimiter(
        cooldown_seconds=2.0,
        clock=lambda: next(times),
    )

    engine_mock = _create_engine_mock()
    engine_mock.respond_and_deliver.return_value = "Ответ бота"

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
        rate_limiter=rate_limiter,
    )

    message, answer_mock, _ = _create_message_mock(
        "Сообщение",
        user_id=TEST_USER_ID,
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.respond_and_deliver.assert_awaited_once_with(
        TEST_USER_ID,
        "Сообщение",
        ANY,
    )

    assert answer_mock.await_count == 2
    assert answer_mock.await_args_list[1].args == (RATE_LIMIT_REPLY,)


def test_text_handler_rate_limit_is_per_user() -> None:
    """Cooldown одного пользователя не должен блокировать другого."""
    times = iter(
        [
            100.0,
            100.5,
        ]
    )

    rate_limiter = UserRateLimiter(
        cooldown_seconds=2.0,
        clock=lambda: next(times),
    )

    engine_mock = _create_engine_mock()
    engine_mock.respond_and_deliver.return_value = "Ответ бота"

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
        rate_limiter=rate_limiter,
    )

    first_message, first_answer_mock, _ = _create_message_mock(
        "Сообщение первого",
        user_id=111,
    )

    second_message, second_answer_mock, _ = _create_message_mock(
        "Сообщение второго",
        user_id=222,
    )

    asyncio.run(
        _call_first_handler(
            router,
            first_message,
        )
    )

    asyncio.run(
        _call_first_handler(
            router,
            second_message,
        )
    )

    assert engine_mock.respond_and_deliver.await_count == 2

    engine_mock.respond_and_deliver.assert_any_await(
        111,
        "Сообщение первого",
        ANY,
    )
    engine_mock.respond_and_deliver.assert_any_await(
        222,
        "Сообщение второго",
        ANY,
    )

    first_answer_mock.assert_awaited_once_with(
        "Ответ бота",
    )
    second_answer_mock.assert_awaited_once_with(
        "Ответ бота",
    )


def test_text_handler_stops_active_roleplay_naturally() -> None:
    """Явная фраза выхода должна завершать RP без обращения к модели."""
    engine_mock = _create_engine_mock()
    engine_mock.disable_roleplay.return_value = True

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock, _ = _create_message_mock(
        "стоп рп",
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.disable_roleplay.assert_awaited_once_with(
        TEST_USER_ID,
    )
    engine_mock.respond_and_deliver.assert_not_awaited()
    answer_mock.assert_awaited_once_with(
        RP_DISABLED_REPLY,
    )


def test_text_handler_reports_inactive_roleplay_on_natural_stop() -> None:
    """Фраза выхода должна сообщать, если RP уже неактивен."""
    engine_mock = _create_engine_mock()
    engine_mock.disable_roleplay.return_value = False

    router = create_text_router(
        cast(ResponseEngine, engine_mock),
    )

    message, answer_mock, _ = _create_message_mock(
        "  Стоп   RP  ",
    )

    asyncio.run(
        _call_first_handler(
            router,
            message,
        )
    )

    engine_mock.disable_roleplay.assert_awaited_once_with(
        TEST_USER_ID,
    )
    engine_mock.respond_and_deliver.assert_not_awaited()
    answer_mock.assert_awaited_once_with(
        RP_ALREADY_DISABLED_REPLY,
    )


def test_text_handler_forwards_mixed_stop_with_question() -> None:
    engine = _create_engine_mock()
    engine.respond_and_deliver.return_value = "TCP — протокол."
    router = create_text_router(cast(ResponseEngine, engine))
    text = "Стоп RP, что такое TCP?"
    message, answer, _ = _create_message_mock(text)
    asyncio.run(router.message.handlers[0].callback(message))
    engine.respond_and_deliver.assert_awaited_once_with(TEST_USER_ID, text, ANY)
    engine.disable_roleplay.assert_not_awaited()
    answer.assert_awaited_once_with("TCP — протокол.")


def test_text_handler_reports_busy_request() -> None:
    engine = _create_engine_mock()
    engine.respond_and_deliver.side_effect = ResponseBusyError
    router = create_text_router(cast(ResponseEngine, engine))
    message, answer, _ = _create_message_mock("Привет")
    asyncio.run(_call_first_handler(router, message))
    answer.assert_awaited_once_with(BUSY_REPLY)


@pytest.mark.parametrize("cancelled", [False, True])
def test_text_handler_propagates_partial_delivery_failure(
    cancelled: bool, caplog: pytest.LogCaptureFixture
) -> None:
    engine = _create_engine_mock()
    engine.respond_and_deliver.return_value = "a" * 5000
    router = create_text_router(cast(ResponseEngine, engine))
    message, answer, _ = _create_message_mock("Привет")
    error = asyncio.CancelledError if cancelled else RuntimeError
    answer.side_effect = [None, error()]
    with caplog.at_level(logging.INFO), pytest.raises(error):
        asyncio.run(_call_first_handler(router, message))
    assert answer.await_count == 2
    outcome = "cancelled" if cancelled else "failed"
    assert f"outcome={outcome} chunks=1/2" in caplog.text
    assert "a" * 100 not in caplog.text
