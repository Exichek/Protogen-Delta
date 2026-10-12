"""Тесты пользовательской настройки и команд создателя."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

from aiogram import Bot, Router
from aiogram.types import CallbackQuery, Message

from protogen_delta.handlers.creator import create_creator_router
from protogen_delta.handlers.proactive import create_proactive_router
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.users import UsersRepository


async def _call(router: Router, name: str, message: Message) -> None:
    for handler in router.message.handlers:
        if handler.callback.__name__ == name:
            await handler.callback(message)
            return
    raise AssertionError(name)


def _message(user_id: int, text: str) -> tuple[Message, AsyncMock]:
    message = Mock(spec=Message)
    message.from_user = SimpleNamespace(id=user_id)
    message.text = text
    message.answer = AsyncMock()
    return cast(Message, message), message.answer


def test_creator_can_message_one_user_and_others_are_denied(tmp_path: Path) -> None:
    """Адресная отправка доступна только CREATOR_ID."""
    bot = AsyncMock(spec=Bot)
    router = create_creator_router(
        bot=cast(Bot, bot),
        users_repository=UsersRepository(tmp_path),
        creator_id=123,
    )

    async def scenario() -> None:
        allowed, answer = _message(123, "/message 456 Привет")
        await _call(router, "message_user", allowed)
        bot.send_message.assert_awaited_once_with(456, "Привет")
        answer_call = answer.await_args
        assert answer_call is not None
        assert "отправлено" in answer_call.args[0]

        denied, denied_answer = _message(999, "/message 456 nope")
        await _call(router, "message_user", denied)
        denied_call = denied_answer.await_args
        assert denied_call is not None
        assert "только создателю" in denied_call.args[0]

    asyncio.run(scenario())


def test_broadcast_requires_confirmation(tmp_path: Path) -> None:
    """Команда рассылки должна сначала показать превью с кнопками."""
    users = UsersRepository(tmp_path)
    users.add(1)
    users.add(2)
    router = create_creator_router(
        bot=cast(Bot, AsyncMock(spec=Bot)),
        users_repository=users,
        creator_id=123,
    )
    message, answer = _message(123, "/broadcast Новость")
    asyncio.run(_call(router, "prepare_broadcast", message))
    answer_call = answer.await_args
    assert answer_call is not None
    assert "2 пользователей" in answer_call.args[0]
    assert answer_call.kwargs["reply_markup"] is not None


def test_confirmed_broadcast_sends_to_registered_users(tmp_path: Path) -> None:
    """После подтверждения текст должен уйти всем зарегистрированным ID."""
    users = UsersRepository(tmp_path)
    users.add(1)
    users.add(2)
    bot = AsyncMock(spec=Bot)
    router = create_creator_router(
        bot=cast(Bot, bot), users_repository=users, creator_id=123
    )
    message, answer = _message(123, "/broadcast Новость")

    async def scenario() -> None:
        await _call(router, "prepare_broadcast", message)
        answer_call = answer.await_args
        assert answer_call is not None
        markup = answer_call.kwargs["reply_markup"]
        data = markup.inline_keyboard[0][0].callback_data
        callback = Mock(spec=CallbackQuery)
        callback.from_user = SimpleNamespace(id=123)
        callback.data = data
        callback.answer = AsyncMock()
        callback.message = Mock(spec=Message)
        callback.message.edit_text = AsyncMock()
        await router.callback_query.handlers[0].callback(callback)
        assert bot.send_message.await_count == 2
        bot.send_message.assert_any_await(1, "Новость")
        bot.send_message.assert_any_await(2, "Новость")
        edit_call = callback.message.edit_text.await_args
        assert edit_call is not None
        assert "Отправлено: 2" in edit_call.args[0]

    asyncio.run(scenario())


def test_user_can_disable_proactive_messages(tmp_path: Path) -> None:
    """Пользователь должен сам управлять фоновыми сообщениями."""
    repository = MemoriesRepository(tmp_path)
    router = create_proactive_router(repository)
    message, answer = _message(77, "/proactive off")
    asyncio.run(_call(router, "proactive", message))
    answer_call = answer.await_args
    assert answer_call is not None
    assert "выключены" in answer_call.args[0]
    assert asyncio.run(repository.proactive_enabled(77)) is False


def test_creator_commands_validate_arguments_and_cancel_broadcast(
    tmp_path: Path,
) -> None:
    """Ошибочные аргументы не отправляются, а превью можно отменить."""
    bot = AsyncMock(spec=Bot)
    router = create_creator_router(
        bot=cast(Bot, bot),
        users_repository=UsersRepository(tmp_path),
        creator_id=123,
    )

    async def scenario() -> None:
        invalid, invalid_answer = _message(123, "/message abc")
        await _call(router, "message_user", invalid)
        invalid_call = invalid_answer.await_args
        assert invalid_call is not None
        assert "Использование" in invalid_call.args[0]

        empty, empty_answer = _message(123, "/broadcast")
        await _call(router, "prepare_broadcast", empty)
        empty_call = empty_answer.await_args
        assert empty_call is not None
        assert "Использование" in empty_call.args[0]

        prepared, prepared_answer = _message(123, "/broadcast Не отправлять")
        await _call(router, "prepare_broadcast", prepared)
        prepared_call = prepared_answer.await_args
        assert prepared_call is not None
        markup = prepared_call.kwargs["reply_markup"]
        data = markup.inline_keyboard[0][1].callback_data
        callback = Mock(spec=CallbackQuery)
        callback.from_user = SimpleNamespace(id=123)
        callback.data = data
        callback.answer = AsyncMock()
        callback.message = Mock(spec=Message)
        callback.message.edit_text = AsyncMock()
        await router.callback_query.handlers[0].callback(callback)
        bot.send_message.assert_not_awaited()
        callback.message.edit_text.assert_awaited_once_with("Рассылка отменена.")

    asyncio.run(scenario())


def test_proactive_command_reports_status_and_rejects_unknown_action(
    tmp_path: Path,
) -> None:
    """Без аргумента команда показывает статус, а опечатку объясняет."""
    repository = MemoriesRepository(tmp_path)
    router = create_proactive_router(repository)

    async def scenario() -> None:
        status, status_answer = _message(77, "/proactive")
        await _call(router, "proactive", status)
        status_call = status_answer.await_args
        assert status_call is not None
        assert "включены" in status_call.args[0]

        invalid, invalid_answer = _message(77, "/proactive maybe")
        await _call(router, "proactive", invalid)
        invalid_call = invalid_answer.await_args
        assert invalid_call is not None
        assert "Использование" in invalid_call.args[0]

    asyncio.run(scenario())
