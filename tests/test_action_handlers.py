"""Команды выполняют только свою явно запрошенную операцию без модели."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.types import Message

from protogen_delta.handlers.actions import create_actions_router
from protogen_delta.repositories.reminders import RemindersRepository
from protogen_delta.services.reminders import ReminderService


def message(text: str, *, user: int = 10) -> Mock:
    return Mock(
        spec=Message,
        text=text,
        from_user=SimpleNamespace(id=user, is_bot=False),
        chat=SimpleNamespace(id=user, type="private"),
        message_id=1,
        forward_origin=None,
        message_thread_id=None,
        answer=AsyncMock(),
    )


def test_command_and_natural_create_list_cancel_and_duplicate(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        bot = AsyncMock(spec=Bot)
        service = ReminderService(cast(Bot, bot), repo, clock=lambda: 100)
        router = create_actions_router(service, clock=lambda: 100)
        create, natural = (
            router.message.handlers[1].callback,
            router.message.handlers[2].callback,
        )
        first = message("/remind через час проверить")
        await create(first)
        assert "сохранено" in first.answer.await_args.args[0]
        assert "МСК" in first.answer.await_args.args[0]
        assert first.answer.await_args.kwargs["parse_mode"] is None
        await create(first)
        assert len(await service.list_user(10)) == 1
        second = message("Дельта, пожалуйста напомни мне через 2 часа позвонить")
        second.message_id = 2
        await natural(second)
        assert len(await service.list_user(10)) == 2
        listed = message("Дельта, покажи мои напоминания")
        await natural(listed)
        assert "позвонить" in listed.answer.await_args.args[0]
        job = (await service.list_user(10))[0]
        cancelled = message(f"/reminders cancel {job.id}")
        await create(cancelled)
        assert "отменено" in cancelled.answer.await_args.args[0]
        await create(cancelled)
        assert "нет" in cancelled.answer.await_args.args[0]
        await create(first)
        assert "уже обработана" in first.answer.await_args.args[0]
        await natural(
            message(f"Отмени напоминание {(await service.list_user(10))[0].id}")
        )
        empty = message("/reminders")
        await create(empty)
        assert "нет" in empty.answer.await_args.args[0]
        bot.send_message.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "scenario", ["group", "forward", "foreign_chat", "bot", "anonymous"]
)
def test_reminders_reject_untrusted_or_nonprivate_source(
    tmp_path: Path, scenario: str
) -> None:
    async def check() -> None:
        repo = RemindersRepository(tmp_path)
        service = ReminderService(
            cast(Bot, AsyncMock(spec=Bot)), repo, clock=lambda: 100
        )
        router = create_actions_router(service, clock=lambda: 100)
        request = message("/remind через час проверить")
        if scenario == "group":
            request.chat.type = "supergroup"
        elif scenario == "forward":
            request.forward_origin = object()
        elif scenario == "foreign_chat":
            request.chat.id = 20
        elif scenario == "bot":
            request.from_user.is_bot = True
        else:
            request.from_user = None
        await router.message.handlers[1].callback(request)
        assert not await service.list_user(10)

    asyncio.run(check())


@pytest.mark.parametrize(
    "text",
    [
        "/remind",
        "/remind когда-нибудь проверить",
        "/reminders bad",
        "/reminders cancel чужое",
        "Отмени напоминание",
    ],
)
def test_invalid_action_gets_help_and_does_not_write(tmp_path: Path, text: str) -> None:
    async def check() -> None:
        repo = RemindersRepository(tmp_path)
        service = ReminderService(
            cast(Bot, AsyncMock(spec=Bot)), repo, clock=lambda: 100
        )
        router = create_actions_router(service, clock=lambda: 100)
        request = message(text)
        await router.message.handlers[1].callback(request)
        assert request.answer.await_count > 0
        assert not await service.list_user(10)

    asyncio.run(check())


def test_failed_storage_does_not_claim_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def check() -> None:
        repo = RemindersRepository(tmp_path)
        service = ReminderService(
            cast(Bot, AsyncMock(spec=Bot)), repo, clock=lambda: 100
        )
        monkeypatch.setattr(
            service, "create", AsyncMock(side_effect=OSError("secret private path"))
        )
        request = message("/remind через час проверить")
        await create_actions_router(service, clock=lambda: 100).message.handlers[
            1
        ].callback(request)
        reply = request.answer.await_args.args[0]
        assert "Не могу подтвердить" in reply and "secret" not in reply
        assert not await repo.list_user(10)

    asyncio.run(check())


def test_long_list_is_split_and_failure_is_visible(tmp_path: Path) -> None:
    async def check() -> None:
        repo = RemindersRepository(tmp_path)
        service = ReminderService(
            cast(Bot, AsyncMock(spec=Bot)), repo, clock=lambda: 100
        )
        for index in range(20):
            await repo.create(10, 10, index + 1, None, "x" * 400, 101, 100)
        failed = (await repo.claim_due(101))[0]
        await repo.finish(failed, 101, failed=True)
        request = message("/reminders")
        await create_actions_router(service, clock=lambda: 100).message.handlers[
            1
        ].callback(request)
        replies = [call.args[0] for call in request.answer.await_args_list]
        assert len(replies) >= 2 and all(len(reply) <= 4096 for reply in replies)
        assert "не удалось доставить" in "".join(replies)

    asyncio.run(check())


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("/calc 0,1+0,2", "= 0.3"),
        ("/calc@bot (25+17)*3", "= 126"),
        ("/calc", "Калькулятор"),
        ("/calc 1/0", "деление на ноль"),
    ],
)
def test_calculator_command(tmp_path: Path, expression: str, expected: str) -> None:
    service = ReminderService(
        cast(Bot, AsyncMock(spec=Bot)), RemindersRepository(tmp_path)
    )
    router = create_actions_router(service)
    request = message(expression)
    asyncio.run(router.message.handlers[0].callback(request))
    assert expected in request.answer.await_args.args[0]


def test_forwarded_calc_requires_own_command(tmp_path: Path) -> None:
    service = ReminderService(
        cast(Bot, AsyncMock(spec=Bot)), RemindersRepository(tmp_path)
    )
    request = message("/calc 2+2")
    request.forward_origin = object()
    asyncio.run(create_actions_router(service).message.handlers[0].callback(request))
    assert "своей командой" in request.answer.await_args.args[0]
