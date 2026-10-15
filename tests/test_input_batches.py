"""Короткие пачки, вложения с пояснением, очередь во время ответа и отмена сбросом."""

import asyncio
import io
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from contextvars import copy_context
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from zipfile import ZipFile

import pytest
from aiogram import Bot, Dispatcher, Router
from aiogram.types import Message, Update
from test_media_handler import PNG_DATA
from test_response_engine import _create_engine

from protogen_delta.core.input_operations import (
    InputOperations,
    InputTicket,
    MemoryInputsMiddleware,
    ReceiveInputsMiddleware,
)
from protogen_delta.core.rate_limiter import MessageTokenBucket
from protogen_delta.core.reply_capture import current_reply_capture
from protogen_delta.handlers.delivery_ledger import DeliveryLedgerMiddleware
from protogen_delta.handlers.documents import create_document_router
from protogen_delta.handlers.input_batches import InputBatchMiddleware, _Input, _Part
from protogen_delta.handlers.media import create_media_router
from protogen_delta.handlers.rp import create_rp_router
from protogen_delta.handlers.sender import SenderMiddleware
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.delivery_ledger import DeliveryLedgerRepository
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.delivery_ledger import current_delivery


def message(
    number: int,
    text: str | None = None,
    *,
    user: int = 123456,
    chat: int | None = None,
    **extra: Any,
) -> Message:
    """Создать настоящее Telegram-сообщение с синтетическими идентификаторами."""
    return Message.model_validate(
        {
            "message_id": number,
            "date": datetime.now(timezone.utc),
            "chat": {
                "id": chat or user,
                "type": "private" if chat is None else "supergroup",
            },
            "from": {"id": user, "is_bot": False, "first_name": "Тест"},
            "text": text,
            **extra,
        }
    )


class Harness:
    """Пропустить сообщения через настоящие middleware и роутеры без сети и платной модели."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        quiet: float = 0.02,
        maximum: float = 0.08,
        bucket: MessageTokenBucket | None = None,
    ) -> None:
        self.engine, _, self.model, _, _, _ = _create_engine()
        self.now = [100.0]
        monkeypatch.setattr(
            "protogen_delta.handlers.input_batches.monotonic", lambda: self.now[0]
        )
        self.inputs = InputOperations()
        self.ledger = DeliveryLedgerRepository(tmp_path)
        self.batch = InputBatchMiddleware(
            self.engine,
            self.inputs,
            self.ledger,
            bucket=bucket or MessageTokenBucket(clock=lambda: self.now[0]),
            quiet_seconds=quiet,
            max_wait_seconds=maximum,
        )
        self.engine.cancel_pending_inputs = self.batch.cancel
        self.dispatcher = Dispatcher()
        self.bot = Bot("987654:synthetic")
        self.answer = AsyncMock()
        monkeypatch.setattr(Message, "answer", self.answer)
        monkeypatch.setattr(Bot, "send_chat_action", AsyncMock())
        self.dispatcher.message.outer_middleware(DeliveryLedgerMiddleware(self.ledger))
        self.dispatcher.message.outer_middleware(ReceiveInputsMiddleware(self.inputs))
        self.dispatcher.message.outer_middleware(SenderMiddleware())

    def routers(self, *routers: Router) -> None:
        for router in routers:
            router.message.middleware(MemoryInputsMiddleware(self.inputs))
            router.message.middleware(self.batch)
        self.dispatcher.include_routers(*routers)

    async def feed(self, event: Message) -> None:
        await self.dispatcher.feed_update(
            self.bot, Update(update_id=event.message_id, message=event)
        )

    async def drain(self) -> None:
        self.release_window()
        await asyncio.wait_for(asyncio.gather(*tuple(self.batch._workers)), 5)

    def release_window(self) -> None:
        """Продвинуть часы независимо от скорости SQLite на машине проверки."""
        self.now[0] += 1
        for queue in self.batch._queues.values():
            queue.changed.set()

    async def close(self) -> None:
        await self.batch.close()
        await self.bot.session.close()


def test_five_forwarded_messages_are_one_ordered_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        try:
            forwarded = {
                "type": "hidden_user",
                "date": datetime.now(timezone.utc),
                "sender_user_name": "Цитата",
            }
            await asyncio.gather(
                *(
                    h.feed(message(index, f"MARKER_{index}", forward_origin=forwarded))
                    for index in [3, 1, 5, 4, 2]
                )
            )
            await h.drain()
            assert h.model.chat.await_count == 1 and h.answer.await_count == 1
            call = str(h.model.chat.await_args)
            positions = [call.index(f"MARKER_{index}") for index in range(1, 6)]
            assert positions == sorted(positions) and "пересланная цитата" in call
            assert not h.inputs._users and h.batch.snapshot()["queued_messages"] == 0
            with closing(sqlite3.connect(tmp_path / "delivery_ledger.db")) as db:
                assert (
                    db.execute(
                        "SELECT COUNT(*) FROM delivery_ledger WHERE state='answered'"
                    ).fetchone()[0]
                    == 5
                )
        finally:
            await h.close()

    asyncio.run(run())


def test_zip_and_later_text_are_one_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        output = io.BytesIO()
        with ZipFile(output, "w") as archive:
            archive.writestr("main.py", "ZIP_CODE_MARKER = 1")
        download = AsyncMock()

        async def fetch(file_id: str, *, destination: io.BytesIO) -> None:
            await asyncio.sleep(0.01)
            destination.write(output.getvalue())

        download.side_effect = fetch
        monkeypatch.setattr(h.bot, "download", download)
        h.routers(
            create_document_router(h.engine, h.bot, native_work=BlockingWorkPool(2)),
            create_text_router(h.engine),
        )
        try:
            await h.feed(
                message(
                    1,
                    document={
                        "file_id": "file",
                        "file_unique_id": "unique",
                        "file_name": "project.zip",
                        "file_size": len(output.getvalue()),
                    },
                )
            )
            await h.feed(message(2, "FOLLOWUP_MARKER: оцени полезность проекта"))
            await h.drain()
            assert h.model.chat.await_count == 1 and h.answer.await_count == 1
            call = str(h.model.chat.await_args)
            assert "ZIP_CODE_MARKER" in call and "FOLLOWUP_MARKER" in call
            assert "project.zip" in call and "Сообщение 2" in call
        finally:
            await h.close()

    asyncio.run(run())


def test_incoming_while_generating_waits_for_next_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        started, release = asyncio.Event(), asyncio.Event()

        async def reply(**kwargs: Any) -> str:
            if h.model.chat.await_count == 1:
                started.set()
                await release.wait()
            return "Готово"

        h.model.chat.side_effect = reply
        try:
            await h.feed(message(1, "FIRST_MARKER"))
            h.release_window()
            await asyncio.wait_for(started.wait(), 3)
            await asyncio.wait_for(h.feed(message(2, "SECOND_MARKER")), 2)
            await h.feed(message(3, "THIRD_MARKER"))
            assert h.model.chat.await_count == 1 and h.answer.await_count == 0
            release.set()
            await h.drain()
            assert h.model.chat.await_count == 2 and h.answer.await_count == 2
            assert "SECOND_MARKER" in str(
                h.model.chat.await_args
            ) and "THIRD_MARKER" in str(h.model.chat.await_args)
            assert "Слишком быстро" not in str(h.answer.await_args_list)
        finally:
            await h.close()

    asyncio.run(run())


def test_flood_is_silently_dropped_before_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch, quiet=0.1, maximum=0.2)
        h.routers(create_text_router(h.engine))
        try:
            await asyncio.gather(
                *(
                    h.feed(message(index, f"FLOOD_{index:02d}"))
                    for index in range(1, 31)
                )
            )
            await h.drain()
            assert h.model.chat.await_count == 1 and h.answer.await_count == 1
            with closing(sqlite3.connect(tmp_path / "delivery_ledger.db")) as db:
                counts = dict(
                    db.execute(
                        "SELECT state,COUNT(*) FROM delivery_ledger GROUP BY state"
                    )
                )
            assert counts == {"answered": 8, "handled": 22}
        finally:
            await h.close()

    asyncio.run(run())


@pytest.mark.parametrize("active", [False, True])
def test_reset_cancels_pending_and_running_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, active: bool
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch, quiet=0.02 if active else 0.5, maximum=0.6)
        h.routers(create_text_router(h.engine))
        started = asyncio.Event()

        async def blocked(**kwargs: Any) -> str:
            started.set()
            await asyncio.Event().wait()
            return "Не должно отправиться"

        h.model.chat.side_effect = blocked
        try:
            await h.feed(message(1, "OLD_MARKER"))
            if active:
                h.release_window()
                await asyncio.wait_for(started.wait(), 3)
                await h.feed(message(2, "QUEUED_MARKER"))
            async with h.inputs.resetting(123456):
                assert not h.batch._queues and not h.engine._delivering_users
            h.answer.assert_not_awaited()
            h.model.chat.side_effect = None
            h.model.chat.return_value = "Новый ответ"
            await h.feed(message(3, "NEW_MARKER"))
            await h.drain()
            assert h.answer.await_count == 1
            assert "OLD_MARKER" not in str(h.model.chat.await_args)
            assert h.batch.snapshot()["queued_messages"] == 0
        finally:
            await h.close()

    asyncio.run(run())


def test_duplicate_update_does_not_enter_batch_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        try:
            event = message(1, "UNIQUE_MARKER")
            await asyncio.gather(h.feed(event), h.feed(event))
            await h.drain()
            assert h.model.chat.await_count == 1 and h.answer.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_different_users_in_group_are_separate_serial_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        active, maximum = 0, 0

        async def reply(**kwargs: Any) -> str:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "Ответ"

        h.model.chat.side_effect = reply
        try:
            await asyncio.gather(
                h.feed(message(1, "PERSON_A", user=1, chat=-100)),
                h.feed(message(2, "PERSON_B", user=2, chat=-100)),
            )
            await h.drain()
            assert h.model.chat.await_count == 2 and maximum == 1
            assert not any(
                "PERSON_A" in str(call) and "PERSON_B" in str(call)
                for call in h.model.chat.await_args_list
            )
        finally:
            await h.close()

    asyncio.run(run())


def test_token_bucket_refill_and_creator_burst() -> None:
    now = [100.0]
    bucket = MessageTokenBucket(clock=lambda: now[0], privileged=frozenset({2}))
    assert all(bucket.allow(1) for _ in range(8))
    assert not bucket.allow(1)
    assert all(bucket.allow(2) for _ in range(16)) and not bucket.allow(2)
    now[0] += 1.25
    assert not bucket.allow(1)
    now[0] += 1.25
    assert bucket.allow(1) and not bucket.allow(1)
    now[0] += 301
    assert all(bucket.allow(1) for _ in range(8))
    assert MessageTokenBucket(refill_seconds=0).allow(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": 0},
        {"capacity": True},
        {"refill_seconds": float("nan")},
        {"refill_seconds": -1},
        {"privileged_capacity": 1},
    ],
)
def test_invalid_bucket_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MessageTokenBucket(**kwargs)


def test_maximum_window_flushes_before_stream_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch, bucket=MessageTokenBucket(refill_seconds=0))
        now = h.now
        h.routers(create_text_router(h.engine))
        sent, first_at = 0, []

        async def reply(**kwargs: Any) -> str:
            first_at.append(sent)
            return "Ответ"

        h.model.chat.side_effect = reply
        try:
            for index in range(1, 13):
                sent = index
                now[0] += 0.019
                await h.feed(message(index, f"STREAM_{index}"))
            now[0] += 1
            for queue in h.batch._queues.values():
                queue.changed.set()
            await h.drain()
            assert first_at and first_at[0] < 12
        finally:
            await h.close()

    asyncio.run(run())


def test_rp_off_bypasses_and_cancels_pending_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch, quiet=0.5, maximum=0.6)
        h.dispatcher.include_router(create_rp_router(h.engine))
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(message(1, "QUEUED_MARKER"))
            await h.feed(message(2, "/rp off"))
            assert not h.batch._queues
            h.model.chat.assert_not_awaited()
            assert h.answer.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_bad_part_does_not_discard_other_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        router = Router()

        @router.message()
        async def prepare(event: Message) -> None:
            if event.message_id == 1:
                raise ValueError("synthetic parser failure")
            await h.engine.respond_and_deliver(123456, event.text or "", h.answer)

        h.routers(router)
        try:
            await h.feed(message(1, "BAD_MARKER"))
            await h.feed(message(2, "GOOD_MARKER"))
            await h.drain()
            assert h.model.chat.await_count == 1 and h.answer.await_count == 1
            with closing(sqlite3.connect(tmp_path / "delivery_ledger.db")) as db:
                assert dict(
                    db.execute("SELECT source_key,state FROM delivery_ledger")
                ) == {
                    "telegram:987654:123456:1": "failed",
                    "telegram:987654:123456:2": "answered",
                }
        finally:
            await h.close()

    asyncio.run(run())


def test_combined_attachments_are_bounded_and_marked() -> None:
    async def unused(event: Any, data: Any) -> None:
        pass

    answer = AsyncMock()
    parts = []
    for index in (1, 2):
        item = _Input(
            message(index, "текст" * 6000), unused, {}, copy_context(), 0, None, None
        )
        part = _Part(
            "текст" * 6000,
            answer,
            {
                "attachment_text": "CONTENT" * 10000,
                "attachment_name": f"file{index}.txt",
                "images": (b"synthetic",) * 8,
                "model_message_override": "Модельный текст" * 3000,
            },
        )
        parts.append((item, part))
    text, options = InputBatchMiddleware._joined(parts)
    assert len(text) <= 16000 and len(options["model_message_override"]) <= 20000
    assert len(options["attachment_text"]) <= 60000
    assert (
        "file1.txt" in options["attachment_text"]
        and "file2.txt" in options["attachment_text"]
    )
    assert "частично" in options["attachment_text"]
    assert (
        len(options["images"]) == 12
        and "изображений не передана" in options["trusted_input_context"]
    )


def test_unknown_delivery_applies_to_each_part_and_is_not_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        router = Router()

        async def uncertain(reply: str) -> None:
            operation = current_delivery.get()
            assert operation is not None
            operation.uncertain = True
            raise TimeoutError("synthetic delivery")

        @router.message()
        async def prepare(event: Message) -> None:
            await h.engine.respond_and_deliver(123456, event.text or "", uncertain)

        h.routers(router)
        try:
            await h.feed(message(1, "FIRST"))
            await h.feed(message(2, "SECOND"))
            await h.drain()
            with closing(sqlite3.connect(tmp_path / "delivery_ledger.db")) as db:
                assert (
                    db.execute(
                        "SELECT COUNT(*) FROM delivery_ledger WHERE state='delivery_unknown'"
                    ).fetchone()[0]
                    == 2
                )
            await h.feed(message(1, "FIRST"))
            assert h.model.chat.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_last_settle_does_not_swallow_new_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        entered, release = asyncio.Event(), asyncio.Event()
        original = h.batch._settle
        blocked = False

        async def settle(inputs: Any, *, failed: bool = False) -> None:
            nonlocal blocked
            await original(inputs, failed=failed)
            if not inputs and not blocked:
                blocked = True
                entered.set()
                await release.wait()

        monkeypatch.setattr(h.batch, "_settle", settle)
        try:
            await h.feed(message(1, "FIRST"))
            h.release_window()
            await asyncio.wait_for(entered.wait(), 3)
            await h.feed(message(2, "SECOND"))
            release.set()
            await h.drain()
            assert (
                h.model.chat.await_count == 2
                and h.batch.snapshot()["queued_messages"] == 0
            )
        finally:
            release.set()
            await h.close()

    asyncio.run(run())


def test_ten_photo_album_uses_one_burst_token_and_one_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)

        async def download(file_id: str, *, destination: io.BytesIO) -> None:
            destination.write(PNG_DATA)

        monkeypatch.setattr(h.bot, "download", AsyncMock(side_effect=download))
        h.routers(
            create_media_router(h.engine, h.bot, album_delay_seconds=0.01),
            create_text_router(h.engine),
        )
        original = h.engine.respond_and_deliver
        seen: list[int] = []

        async def observe(*args: Any, **kwargs: Any) -> None:
            if current_reply_capture.get() is None:
                seen.append(len(kwargs.get("images", ())))
            await original(*args, **kwargs)

        monkeypatch.setattr(h.engine, "respond_and_deliver", observe)
        try:
            await asyncio.gather(
                *(
                    h.feed(
                        message(
                            index,
                            media_group_id="album",
                            photo=[
                                {
                                    "file_id": f"photo{index}",
                                    "file_unique_id": f"unique{index}",
                                    "width": 1,
                                    "height": 1,
                                    "file_size": len(PNG_DATA),
                                }
                            ],
                        )
                    )
                    for index in range(1, 11)
                )
            )
            await h.feed(message(11, "Расскажи об этих картинках"))
            await h.drain()
            assert seen == [10] and h.model.chat.await_count == 1
            assert h.answer.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_reset_during_worker_registration_leaves_no_old_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        original = h.inputs.receive
        reached = asyncio.Event()

        @asynccontextmanager
        async def receive(owner: int) -> AsyncIterator[InputTicket]:
            if any(
                queue.task is asyncio.current_task()
                for queue in h.batch._queues.values()
            ):
                reached.set()
                await asyncio.Event().wait()
            async with original(owner) as ticket:
                yield ticket

        monkeypatch.setattr(h.inputs, "receive", receive)
        task = asyncio.create_task(h.feed(message(1, "OLD")))
        try:
            await asyncio.wait_for(reached.wait(), 3)
            async with h.inputs.resetting(123456):
                assert (
                    not h.batch._queues and h.batch.snapshot()["queued_messages"] == 0
                )
            await asyncio.gather(task, return_exceptions=True)
            h.model.chat.assert_not_awaited()
            monkeypatch.setattr(h.inputs, "receive", original)
            await h.feed(message(2, "NEW"))
            await h.drain()
            assert h.model.chat.await_count == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await h.close()

    asyncio.run(run())
