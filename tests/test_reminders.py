"""Сохранность очереди, изоляция, доставка и гонки отмены/сброса."""

import asyncio
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)

from protogen_delta.core.reminder_request import ReminderRequest
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.reminders import Reminder, RemindersRepository
from protogen_delta.services.reminders import ReminderService


async def add(
    repo: RemindersRepository, user: int = 1, message: int = 1, due: float = 10
) -> Reminder:
    return await repo.create(user, user, message, None, "проверить проект", due, 1)


def test_durable_queue_duplicate_and_owner_isolation(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        first = await add(repo)
        assert (await add(repo)).id == first.id
        await add(repo, 2)
        reloaded = RemindersRepository(tmp_path)
        assert len(await reloaded.list_user(1)) == 1
        assert not await reloaded.cancel(2, first.id, 2)
        assert await reloaded.cancel(1, first.id, 2)
        assert not await reloaded.cancel(1, first.id, 3)
        assert (await add(repo)).status == "cancelled"
        assert await reloaded.list_user(1) == []
        assert len(await reloaded.list_user(2)) == 1
        await reloaded.delete_user(2)
        assert await reloaded.list_user(2) == []
        # Обе схемы сосуществуют и входят в прежние SQLite-бэкапы.
        episodes = MemoriesRepository(tmp_path)
        await episodes.remember(1, "topic", "Драконы", 1)
        assert len(await episodes.search(1, "дракон")) == 1
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    asyncio.run(scenario())


def test_limits_are_atomic_across_repository_instances(tmp_path: Path) -> None:
    async def scenario() -> None:
        first, second = RemindersRepository(
            tmp_path, per_user_limit=2, total_limit=3
        ), RemindersRepository(tmp_path, per_user_limit=2, total_limit=3)
        results = await asyncio.gather(
            add(first, message=1),
            add(second, message=2),
            add(first, message=3),
            return_exceptions=True,
        )
        assert sum(isinstance(result, Reminder) for result in results) == 2
        assert sum(isinstance(result, ValueError) for result in results) == 1
        await add(second, user=2)
        with pytest.raises(ValueError, match="Очередь"):
            await add(second, user=3)
        # Победителя гонки выбирает порядок транзакций, а не asyncio.gather.
        saved = next(result for result in results if isinstance(result, Reminder))
        assert (await add(first, message=saved.source_message_id)).id == saved.id

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("user", "chat", "message", "thread", "due", "text"),
    [
        (0, 0, 1, None, 10, "x"),
        (1, 2, 1, None, 10, "x"),
        (1, 1, 0, None, 10, "x"),
        (True, 1, 1, None, 10, "x"),
        (1, 1, 1, -1, 10, "x"),
        (1, 1, 1, None, 0, "x"),
        (1, 1, 1, None, float("nan"), "x"),
        (1, 1, 1, None, float("inf"), "x"),
        (1, 1, 1, None, 366 * 86400, "x"),
        (1, 1, 1, None, 10, ""),
        (1, 1, 1, None, 10, "x" * 501),
    ],
)
def test_invalid_data(
    tmp_path: Path,
    user: int,
    chat: int,
    message: int,
    thread: int | None,
    due: float,
    text: str,
) -> None:
    repo = RemindersRepository(tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(repo.create(user, chat, message, thread, text, due, 1))


def test_invalid_repository_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RemindersRepository(tmp_path, per_user_limit=0)
    with pytest.raises(ValueError):
        RemindersRepository(tmp_path, total_limit=0)


def test_claim_is_exclusive_recovers_expired_lease_and_rejects_stale_finish(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first, second = RemindersRepository(tmp_path), RemindersRepository(tmp_path)
        await add(first)
        assert await first.claim_due(9) == []
        claims = await asyncio.gather(first.claim_due(10), second.claim_due(10))
        job = next(batch[0] for batch in claims if batch)
        assert sum(map(len, claims)) == 1
        assert await second.claim_due(129) == []
        recovered = (await second.claim_due(130))[0]
        assert (
            recovered.id == job.id
            and recovered.attempts == 2
            and recovered.lease != job.lease
        )
        assert not await first.is_claimed(job)
        await first.finish(job, 130)
        assert await second.is_claimed(recovered)
        await second.finish(recovered, 130)
        assert not await second.claim_due(1000)
        assert not await second.list_user(1)

    asyncio.run(scenario())


def test_claim_respects_batch_and_one_per_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        for user in range(1, 13):
            await add(repo, user)
            await add(repo, user, 2)
        batch = await repo.claim_due(10)
        assert len(batch) == 10 and len({job.user_id for job in batch}) == 10
        for job in batch:
            await repo.finish(job, 10)
        next_batch = await repo.claim_due(11)
        assert len(next_batch) == 10

    asyncio.run(scenario())


def test_lease_renewal_and_recovery_have_only_one_winner(tmp_path: Path) -> None:
    async def scenario() -> None:
        first, second = RemindersRepository(tmp_path), RemindersRepository(tmp_path)
        await add(first)
        job = (await first.claim_due(10))[0]
        renewed, recovered = await asyncio.gather(
            first.renew_claim(job, 150), second.claim_due(150)
        )
        assert int(renewed) + len(recovered) == 1
        if renewed:
            assert await second.claim_due(269) == []
        else:
            assert not await first.renew_claim(job, 150)

    asyncio.run(scenario())


def test_retry_limit_and_terminal_retention(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        await add(repo)
        for attempt in range(5):
            now = 10 + attempt
            job = (await repo.claim_due(now))[0]
            await repo.finish(job, now, retry_at=now + 1)
            assert await repo.claim_due(now + 0.5) == []
        assert (await repo.list_user(1))[0].status == "failed"
        assert await repo.claim_due(100) == []
        await repo.claim_due(100 * 86400)
        assert await repo.list_user(1) == []
        # Пятый оборванный запуск также становится failed после истечения аренды.
        await repo.create(2, 2, 1, None, "тест", 100 * 86400 + 1, 100 * 86400)
        for attempt in range(5):
            assert len(await repo.claim_due(100 * 86400 + 1 + attempt * 120)) == 1
        assert not await repo.claim_due(100 * 86400 + 601)
        assert (await repo.list_user(2))[0].status == "failed"

    asyncio.run(scenario())


def test_worker_delivers_plain_text_once_and_preserves_topic(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        job = await repo.create(1, 1, 1, 12, "<b>проверить</b>", 10, 1)
        bot = AsyncMock(spec=Bot)
        service = ReminderService(cast(Bot, bot), repo, clock=lambda: 10)
        await service.deliver_due()
        bot.send_message.assert_awaited_once()
        args = bot.send_message.await_args.kwargs
        assert args["chat_id"] == 1 and args["message_thread_id"] == 12
        assert args["parse_mode"] is None and "<b>проверить</b>" in args["text"]
        assert args["link_preview_options"].is_disabled is True
        await ReminderService(
            cast(Bot, bot), RemindersRepository(tmp_path), clock=lambda: 20
        ).deliver_due()
        assert bot.send_message.await_count == 1
        assert not await service.cancel(1, job.id)
        assert await service.list_user(1) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("error", ["rate", "forbidden", "bad_request", "network"])
def test_delivery_failures_have_bounded_retry(tmp_path: Path, error: str) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        await add(repo)
        bot = AsyncMock(spec=Bot)
        errors = {
            "rate": TelegramRetryAfter(
                method=Mock(), message="synthetic", retry_after=40
            ),
            "forbidden": TelegramForbiddenError(method=Mock(), message="synthetic"),
            "bad_request": TelegramBadRequest(method=Mock(), message="synthetic"),
            "network": OSError("private text must not be logged"),
        }
        bot.send_message.side_effect = errors[error]
        service = ReminderService(cast(Bot, bot), repo, clock=lambda: 10)
        await service.deliver_due()
        jobs = await service.list_user(1)
        assert jobs[0].status == (
            "failed" if error in {"forbidden", "bad_request"} else "pending"
        )
        assert await repo.claim_due(39) == []
        if jobs[0].status == "pending":
            bot.send_message.side_effect = None
            await ReminderService(cast(Bot, bot), repo, clock=lambda: 50).deliver_due()
            assert not await service.list_user(1)

    asyncio.run(scenario())


def test_cancel_or_reset_prevents_already_claimed_job_from_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        bot = AsyncMock(spec=Bot)
        service = ReminderService(cast(Bot, bot), repo, clock=lambda: 10)
        cancelled = await add(repo)
        reset = await add(repo, user=2)
        jobs = await repo.claim_due(10)
        assert len(jobs) == 2
        assert await service.cancel(1, cancelled.id)
        await service.delete_user(2)
        assert not await repo.is_claimed(reset)
        monkeypatch.setattr(repo, "claim_due", AsyncMock(return_value=jobs))
        await service.deliver_due()
        bot.send_message.assert_not_awaited()

    asyncio.run(scenario())


def test_reset_waits_for_inflight_send_and_leaves_no_late_jobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        await add(repo)
        bot = AsyncMock(spec=Bot)
        entered, release = asyncio.Event(), asyncio.Event()

        async def send(**kwargs: object) -> None:
            entered.set()
            await release.wait()

        bot.send_message.side_effect = send
        service = ReminderService(cast(Bot, bot), repo, clock=lambda: 10)
        delivery = asyncio.create_task(service.deliver_due())
        await entered.wait()
        assert service.snapshot()["active"] == 1
        reset = asyncio.create_task(service.delete_user(1))
        await asyncio.sleep(0)
        assert not reset.done()
        release.set()
        await asyncio.gather(delivery, reset)
        assert await service.list_user(1) == []
        await service.deliver_due()
        assert bot.send_message.await_count == 1

    asyncio.run(scenario())


def test_cancelled_create_finishes_sqlite_before_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        service = ReminderService(cast(Bot, AsyncMock(spec=Bot)), repo, clock=lambda: 1)
        entered, release = threading.Event(), threading.Event()
        original = repo._create

        def slow(*args: object) -> Reminder:
            entered.set()
            assert release.wait(5)
            return original(*args)  # type: ignore[arg-type]

        monkeypatch.setattr(repo, "_create", slow)
        create = asyncio.create_task(
            service.create(1, 1, 1, None, ReminderRequest("тест", 10))
        )
        assert await asyncio.to_thread(entered.wait, 5)
        create.cancel()
        reset = asyncio.create_task(service.delete_user(1))
        await asyncio.sleep(0)
        assert not reset.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await create
        await reset
        assert await service.list_user(1) == []

    asyncio.run(scenario())


def test_stop_releases_claims_and_queue_survives_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        await add(repo)
        bot = AsyncMock(spec=Bot)
        entered = asyncio.Event()

        async def wait(**kwargs: object) -> None:
            entered.set()
            await asyncio.Event().wait()

        bot.send_message.side_effect = wait
        service = ReminderService(cast(Bot, bot), repo, clock=lambda: 10, interval=0.01)
        await service.start()
        await service.start()
        await entered.wait()
        await service.close()
        await service.close()
        assert (await repo.list_user(1))[0].status == "pending"
        assert service.snapshot()["active"] == 0
        bot.send_message.side_effect = None
        await ReminderService(
            cast(Bot, bot), RemindersRepository(tmp_path), clock=lambda: 41
        ).deliver_due()
        assert not await repo.list_user(1)

    asyncio.run(scenario())


def test_worker_survives_repository_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        repo = RemindersRepository(tmp_path)
        service = ReminderService(cast(Bot, AsyncMock(spec=Bot)), repo, interval=0.01)
        failed = asyncio.Event()

        async def unavailable(now: float) -> list[Reminder]:
            failed.set()
            raise OSError("private-data")

        monkeypatch.setattr(repo, "claim_due", unavailable)
        await service.start()
        await failed.wait()
        await asyncio.sleep(0.02)
        assert service._task is not None and not service._task.done()
        await service.close()
        assert int(cast(int, service.snapshot()["failed"])) >= 1

    asyncio.run(scenario())


def test_invalid_worker_interval(tmp_path: Path) -> None:
    for interval in (0, 61, float("nan")):
        with pytest.raises(ValueError):
            ReminderService(
                cast(Bot, AsyncMock(spec=Bot)),
                RemindersRepository(tmp_path),
                interval=interval,
            )
