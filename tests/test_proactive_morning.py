"""Ночная очередь получает сохраняемый разброс, не обходит новые ответы и лимиты."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_lively_initiative import build
from test_proactive_schedule import deadline

from protogen_delta.repositories.initiative import MSK
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.services.proactive import ProactiveConfig


def morning() -> float:
    return datetime(2026, 10, 21, 9, tzinfo=MSK).timestamp()


@pytest.mark.parametrize("stickers", [False, True])
def test_night_queue_is_spread_before_batch_and_survives_restart(
    tmp_path: Path, stickers: bool
) -> None:
    async def run() -> None:
        wake = morning()
        random_delay = Mock(side_effect=[600, 7200])
        repo = MemoriesRepository(tmp_path, random_delay=random_delay)
        for owner in (7, 8):
            await repo.note_user_activity(owner, wake - 12 * 3600)

        async def due(at: float) -> list[int]:
            return [
                item.user_id
                for item in await repo.due_candidates(
                    now=at,
                    idle_seconds=5400,
                    cooldown_seconds=72000,
                    limit=1,
                    sticker_delay=(1800, 1800) if stickers else None,
                    quiet_end_at=wake,
                )
            ]

        # Подготовить ночные сроки без вызова источника случайности.
        repo._random_delay = lambda lo, hi: lo
        await due(wake - 1)
        repo._random_delay = random_delay
        assert await due(wake) == []
        assert deadline(tmp_path) == (wake + 600, 0)
        assert random_delay.call_count == 2
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            rows = db.execute(
                "SELECT next_proactive_at,next_sticker_at FROM engagement ORDER BY user_id"
            ).fetchall()
        assert rows == [
            (wake + delay, wake + delay if stickers else None) for delay in (600, 7200)
        ]
        # Новый процесс не перевыбирает уже сохранённые сроки.
        repo = MemoriesRepository(
            tmp_path, random_delay=Mock(side_effect=AssertionError)
        )
        assert await due(wake + 599) == []
        assert await due(wake + 600) == [7]
        await repo.note_proactive_sent(7, wake + 600)
        assert await due(wake + 7199) == []
        assert await due(wake + 7200) == [8]

    asyncio.run(run())


@pytest.mark.parametrize("value", [599, 7201, float("nan"), float("inf")])
def test_invalid_morning_randomness_rolls_back(tmp_path: Path, value: float) -> None:
    async def run() -> None:
        wake = morning()
        repo = MemoriesRepository(tmp_path, random_delay=lambda *_: value)
        await repo.note_user_activity(7, wake - 12 * 3600)
        with pytest.raises(ValueError, match="тихих часов"):
            await repo.due_candidates(
                now=wake,
                idle_seconds=5400,
                cooldown_seconds=72000,
                limit=10,
                quiet_end_at=wake,
            )
        assert deadline(tmp_path) == (None, 0)

    asyncio.run(run())


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("new_input", [False, True])
def test_messenger_waits_without_model_and_new_reply_rearms(
    tmp_path: Path, restart: bool, new_input: bool
) -> None:
    async def run() -> None:
        wake = morning()
        now = [wake - 1]
        memories, _, _, _, bot, model, service = build(tmp_path, now)
        service._local_datetime = lambda: datetime.fromtimestamp(now[0], MSK)
        await memories.note_user_activity(7, wake - 12 * 3600)
        assert await service.run_once() == 0
        now[0] = wake
        assert await service.run_once() == 0
        assert deadline(tmp_path) == (wake + 600, 0)
        model.chat.assert_not_awaited()
        bot.send_message.assert_not_awaited()
        if restart:
            memories, _, _, _, bot, model, service = build(tmp_path, now)
            service._local_datetime = lambda: datetime.fromtimestamp(now[0], MSK)
        now[0] = wake + 599
        assert await service.run_once() == 0
        if new_input:
            await memories.note_user_activity(7, now[0])
            await memories.note_dialogue_activity(7, now[0] + 1)
        now[0] = wake + 600
        assert await service.run_once() == (0 if new_input else 1)
        if new_input:
            model.chat.assert_not_awaited()
            assert deadline(tmp_path) == (wake + 600 + 5400, 0)
            now[0] += 3 * 3600
            assert await service.run_once() == 1
        bot.send_message.assert_awaited_once()
        # Без ответа не возникает следующей инициативы на следующее утро.
        now[0] = wake + 86400
        assert await service.run_once() == 0
        model.chat.assert_awaited_once()

    asyncio.run(run())


def test_daytime_deadlines_and_disabled_quiet_hours_are_preserved(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        wake = morning()
        now = [wake + 5400]
        memories, _, _, _, bot, _, service = build(tmp_path, now)
        service._local_datetime = lambda: datetime.fromtimestamp(now[0], MSK)
        await memories.note_user_activity(7, wake + 1)
        assert await service.run_once() == 0
        assert deadline(tmp_path) == (wake + 5401, 0)
        now[0] += 1800
        assert await service.run_once() == 1
        bot.send_message.assert_awaited_once()
        service._config = ProactiveConfig(quiet_start_hour=0, quiet_end_hour=0)
        assert service._quiet_end_at() is None
        service._config = ProactiveConfig(quiet_start_hour=9, quiet_end_hour=12)
        # Для поддерживаемой дневной паузы берётся последний закончившийся интервал.
        now[0] = wake - 3600
        assert (
            service._quiet_end_at()
            == wake + 3 * 3600 - timedelta(days=1).total_seconds()
        )

    asyncio.run(run())
