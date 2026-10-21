"""Поздний ответ в личном диалоге снова разрешает будущую инициативу."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message, PhotoSize, User
from test_lively_initiative import build, saved
from test_proactive_schedule import deadline

from protogen_delta.core.input_operations import InputOperations
from protogen_delta.handlers.proactive_activity import ProactiveIncomingMiddleware
from protogen_delta.repositories.initiative import MSK


def at(hour: int) -> float:
    return datetime(2026, 10, 17, hour, tzinfo=MSK).timestamp()


@pytest.mark.parametrize("photo", [False, True])
@pytest.mark.parametrize("restart", [False, True])
def test_reply_after_ten_hours_rearms_text_and_photo_across_restart(
    tmp_path: Path, photo: bool, restart: bool
) -> None:
    async def run() -> None:
        now = [at(9) + 600]
        memories, _, _, states, bot, model, service = build(tmp_path, now)
        service._local_datetime = lambda: datetime.fromtimestamp(now[0], MSK)
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 1
        assert deadline(tmp_path)[1] == 1

        now[0] += 10 * 3600
        assert await service.run_once() == 0
        assert model.chat.await_count == 1
        event = Message(
            message_id=2,
            date=datetime.fromtimestamp(now[0], MSK),
            chat=Chat(id=7, type="private"),
            from_user=User(id=7, is_bot=False, first_name="Тест"),
            text=None if photo else "Отвечаю вечером",
            photo=(
                [PhotoSize(file_id="test", file_unique_id="test", width=10, height=10)]
                if photo
                else None
            ),
        )
        middleware = ProactiveIncomingMiddleware(
            memories, states, InputOperations(), clock=lambda: now[0]
        )
        await middleware(AsyncMock(return_value=None), event, {})
        assert deadline(tmp_path) == (None, 0)
        now[0] += 30
        await memories.note_dialogue_activity(7, now[0])

        if restart:
            memories, _, _, states, bot, model, service = build(tmp_path, now)
            service._local_datetime = lambda: datetime.fromtimestamp(now[0], MSK)
        model.chat.return_value = "Предлагаю выбрать самое подозрительное печенье :3"
        assert await service.run_once() == 0
        assert deadline(tmp_path) == (now[0] + 5400, 0)
        now[0] += 3 * 3600
        assert await service.run_once() == 1
        assert deadline(tmp_path)[1] == 1
        assert bot.send_message.await_count == (1 if restart else 2)
        # Без нового ответа следующего сообщения снова не будет.
        now[0] += 30 * 3600
        assert await service.run_once() == 0

    asyncio.run(run())


def test_evening_reply_survives_quiet_hours_and_rearms_next_morning(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [at(19)]
        memories, _, _, _, bot, model, service = build(tmp_path, now)
        service._local_datetime = lambda: datetime.fromtimestamp(now[0], MSK)
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 1
        now[0] = at(22)
        await memories.note_user_activity(7, now[0])
        await memories.note_dialogue_activity(7, now[0] + 10)
        model.chat.return_value = (
            "Есть мысль: придумать облаку очень серьёзную должность :3"
        )
        now[0] = at(23) + 3600
        assert await service.run_once() == 0
        now[0] = at(9) + 86400
        assert await service.run_once() == 0
        now[0] += 600
        assert await service.run_once() == 1
        assert bot.send_message.await_count == 2
        assert (
            saved(tmp_path)[4] == 1
        )  # Новый день, не сброс суточного предела ответом.

    asyncio.run(run())


def test_answer_does_not_cancel_opt_out_or_reset_daily_cap(tmp_path: Path) -> None:
    async def run() -> None:
        now = [at(9) + 600]
        memories, _, _, _, bot, model, service = build(tmp_path, now)
        service._local_datetime = lambda: datetime.fromtimestamp(now[0], MSK)
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 1
        await memories.note_user_activity(7, now[0] + 10)
        model.chat.return_value = "Выбирай: дракон-библиотекарь или кот-капитан?"
        now[0] += 4 * 3600 + 30
        assert await service.run_once() == 1
        await memories.note_user_activity(7, now[0] + 10)
        now[0] += 4 * 3600 + 30
        assert await service.run_once() == 0
        assert deadline(tmp_path)[1] == 0
        assert saved(tmp_path)[4] == 2
        assert bot.send_message.await_count == 2
        await memories.set_proactive(7, False, now[0])
        await memories.note_user_activity(7, now[0] + 10)
        now[0] = at(12) + 86400
        assert await service.run_once() == 0
        assert not await memories.proactive_enabled(7)

    asyncio.run(run())
