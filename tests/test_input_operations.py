"""Сброс во время скачивания, распознавания, SQL и ожидания диалога."""

import asyncio
import io
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Chat, Message, TelegramObject, Update, User
from test_document_handler import TEST_USER_ID, _message
from test_reset_handler import _create_callback_mock
from test_response_engine import _create_engine
from test_voice_handler import _message as voice_message

from protogen_delta.core.input_operations import (
    InputOperations,
    MemoryInputsMiddleware,
    ReceiveInputsMiddleware,
)
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.core.user_statistics import UserStatisticsMiddleware
from protogen_delta.handlers.documents import create_document_router
from protogen_delta.handlers.reset import create_reset_router
from protogen_delta.handlers.voice import create_voice_router
from protogen_delta.repositories.user_statistics import UserStatisticsRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.speech import SpeechTranscriber


def _event(
    user_id: int = TEST_USER_ID, text: str = "hello", number: int = 1
) -> Message:
    return Message(
        message_id=number,
        date=datetime.now(timezone.utc),
        chat=Chat(id=user_id, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="Synthetic"),
        text=text,
    )


async def _reset(engine: Any, inputs: InputOperations, tmp_path: Path) -> None:
    router = create_reset_router(
        engine, UsersRepository(tmp_path), input_operations=inputs
    )
    callback, _, _ = _create_callback_mock(
        f"reset:confirm:{TEST_USER_ID}:{int(datetime.now().timestamp())}"
    )
    await router.callback_query.handlers[0].callback(callback)


def test_reset_stops_document_download_before_it_can_repopulate_memory(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, model, _, _, _ = _create_engine()
        async with engine._user_states.use(TEST_USER_ID) as state:
            state.content_mode = "adult"
        inputs = InputOperations()
        receive, memory = ReceiveInputsMiddleware(inputs), MemoryInputsMiddleware(
            inputs
        )
        started, release = asyncio.Event(), asyncio.Event()
        bot = AsyncMock(spec=Bot)

        async def download(file_id: str, *, destination: io.BytesIO) -> None:
            started.set()
            await release.wait()
            destination.write(b"OLD_PRIVATE_DOCUMENT")

        bot.download.side_effect = download
        document = create_document_router(engine, bot, native_work=BlockingWorkPool(2))
        message, answer = _message(
            SimpleNamespace(
                file_id="synthetic",
                file_size=20,
                file_name="old.txt",
                mime_type="text/plain",
            ),
            "Разбери старый документ",
        )
        assert message.from_user is not None
        message.from_user.is_bot = False

        async def deliver(event: Any, data: Any) -> None:
            await document.message.handlers[0].callback(event)

        async def select(event: Any, data: Any) -> None:
            await memory(deliver, event, data)

        task = asyncio.create_task(receive(select, message, {}))
        await started.wait()
        await _reset(engine, inputs, tmp_path)
        assert task.cancelled()
        async with engine._user_states.use(TEST_USER_ID) as state:
            state.content_mode = "adult"
        release.set()
        answer.assert_not_awaited()
        model.chat.assert_not_awaited()
        assert not engine._user_states.get(TEST_USER_ID).history
        assert not inputs._users

    asyncio.run(scenario())


def test_reset_cancels_voice_pipeline_and_waits_for_decoder_cleanup(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, model, _, _, _ = _create_engine()
        inputs = InputOperations()
        transcriber = AsyncMock(spec=SpeechTranscriber)
        started, cleanup, release_cleanup = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def transcribe(data: bytes) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup.set()
                await release_cleanup.wait()

        transcriber.transcribe.side_effect = transcribe
        bot = AsyncMock(spec=Bot)

        async def download(file_id: str, *, destination: io.BytesIO) -> None:
            destination.write(b"synthetic")

        bot.download.side_effect = download
        router = create_voice_router(
            engine, bot, transcriber, native_work=BlockingWorkPool(2)
        )
        message, answer = voice_message(
            voice=SimpleNamespace(file_id="voice", file_size=10, duration=1)
        )
        assert message.from_user is not None
        message.from_user.is_bot = False
        receive, memory = ReceiveInputsMiddleware(inputs), MemoryInputsMiddleware(
            inputs
        )

        async def deliver(event: Any, data: Any) -> None:
            await router.message.handlers[0].callback(event)

        async def select(event: Any, data: Any) -> None:
            await memory(deliver, event, data)

        task = asyncio.create_task(receive(select, message, {}))
        await started.wait()
        reset = asyncio.create_task(_reset(engine, inputs, tmp_path))
        await cleanup.wait()
        assert not reset.done()
        release_cleanup.set()
        await reset
        assert task.cancelled()
        answer.assert_not_awaited()
        model.chat.assert_not_awaited()
        assert not inputs._users

    asyncio.run(scenario())


def test_selected_memory_router_is_cancelled_but_command_and_other_user_continue() -> (
    None
):
    async def scenario() -> None:
        inputs = InputOperations()
        dispatcher = Dispatcher()
        dispatcher.message.outer_middleware(ReceiveInputsMiddleware(inputs))
        commands, conversation = Router(), Router()
        command_started, old_started, other_started = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        release = asyncio.Event()
        finished: list[str] = []

        @commands.message(Command("e6"))
        async def command(message: Message) -> None:
            command_started.set()
            await release.wait()
            finished.append("command")

        @conversation.message()
        async def text(message: Message) -> None:
            assert message.from_user is not None
            (old_started if message.from_user.id == 1 else other_started).set()
            await release.wait()
            finished.append(str(message.from_user.id))

        conversation.message.middleware(MemoryInputsMiddleware(inputs))
        dispatcher.include_routers(commands, conversation)
        bot = Bot("123456:synthetic")
        try:
            tasks = [
                asyncio.create_task(
                    dispatcher.feed_update(
                        bot, Update(update_id=n, message=_event(user, text, n))
                    )
                )
                for n, user, text in (
                    (1, 1, "/e6 dragon"),
                    (2, 1, "old"),
                    (3, 2, "other"),
                )
            ]
            await asyncio.gather(
                command_started.wait(), old_started.wait(), other_started.wait()
            )
            async with inputs.resetting(1):
                assert tasks[1].cancelled()
                assert not tasks[0].done() and not tasks[2].done()
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert sorted(finished) == ["2", "command"]
            assert not inputs._users
        finally:
            await bot.session.close()

    asyncio.run(scenario())


def test_stale_receipt_waiting_for_router_never_enters_memory_handler() -> None:
    async def scenario() -> None:
        inputs = InputOperations()
        receive, memory = ReceiveInputsMiddleware(inputs), MemoryInputsMiddleware(
            inputs
        )
        started, release = asyncio.Event(), asyncio.Event()
        handler = AsyncMock()

        async def delayed_router(event: Any, data: Any) -> Any:
            started.set()
            await release.wait()
            return await memory(handler, event, data)

        task = asyncio.create_task(receive(delayed_router, _event(), {}))
        await started.wait()
        async with inputs.resetting(TEST_USER_ID):
            assert not task.done()
        release.set()
        assert await task is None
        handler.assert_not_awaited()
        assert not inputs._users

    asyncio.run(scenario())


def test_reset_waits_for_delivery_then_cancels_prepared_job_behind_writer(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        states = engine._user_states
        async with states.use(TEST_USER_ID) as state:
            state.content_mode = "adult"
        inputs = InputOperations()
        delivery_started, release_delivery, delivery_finished, queued = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )

        async def old_delivery(reply: str) -> None:
            delivery_started.set()
            await release_delivery.wait()
            delivery_finished.set()

        async def reply() -> None:
            async with inputs.receive(TEST_USER_ID) as ticket:
                assert inputs.start_memory(ticket)
                await engine.respond_and_deliver(TEST_USER_ID, "active", old_delivery)

        active = asyncio.create_task(reply())
        await delivery_started.wait()
        reset = asyncio.create_task(_reset(engine, inputs, tmp_path))
        # Writer registration is deterministic: wait until the gate reports it.
        async with asyncio.timeout(2):
            while not states.get(TEST_USER_ID).conversation_gate._waiting_writers:
                await asyncio.sleep(0)
        stale_deliver = AsyncMock()

        async def prepared() -> None:
            async with inputs.receive(TEST_USER_ID) as ticket:
                assert inputs.start_memory(ticket)
                queued.set()
                await engine.respond_and_deliver(
                    TEST_USER_ID, "old queued", stale_deliver, chat_id=-2
                )

        pending = asyncio.create_task(prepared())
        await queued.wait()
        assert not reset.done()
        release_delivery.set()
        results = await asyncio.gather(active, reset, return_exceptions=True)
        assert results[1] is None and delivery_finished.is_set()
        # The writer may acquire the gate before the finished reply exits its
        # receipt context; cancellation at that point cannot interrupt delivery.
        assert results[0] is None or isinstance(results[0], asyncio.CancelledError)
        assert pending.cancelled()
        stale_deliver.assert_not_awaited()
        assert not states.get(TEST_USER_ID).history
        async with states.use(TEST_USER_ID) as state:
            state.content_mode = "adult"
        fresh = AsyncMock()
        await engine.respond_and_deliver(TEST_USER_ID, "fresh", fresh)
        fresh.assert_awaited_once()
        assert states.get(TEST_USER_ID).history[-1].user_message == "fresh"
        assert not inputs._users

    asyncio.run(scenario())


def test_statistics_write_finishes_before_reset_and_stale_receipt_skips_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        inputs, states = InputOperations(), UserStateStore()
        repo = UserStatisticsRepository(tmp_path)
        original = repo.record
        entered, release = asyncio.Event(), asyncio.Event()

        async def record(*args: Any) -> None:
            entered.set()
            await release.wait()
            await original(*args)

        repo.record = record  # type: ignore[method-assign, assignment]
        statistics = UserStatisticsMiddleware(repo, states, inputs)
        receive, memory = ReceiveInputsMiddleware(inputs), MemoryInputsMiddleware(
            inputs
        )
        handler = AsyncMock()

        async def selected(event: Any, data: Any) -> Any:
            return await memory(handler, event, data)

        async def counted(event: Any, data: Any) -> Any:
            return await statistics(selected, event, data)

        first = asyncio.create_task(receive(counted, _event(), {}))
        await entered.wait()
        first.cancel()
        reset_entered = asyncio.Event()

        async def clear() -> None:
            async with inputs.resetting(TEST_USER_ID):
                reset_entered.set()
                await repo.delete_user(TEST_USER_ID)

        reset = asyncio.create_task(states.reset_user(TEST_USER_ID, cleanup=clear))
        async with asyncio.timeout(2):
            while not states.get(TEST_USER_ID).conversation_gate._waiting_writers:
                await asyncio.sleep(0)
        stale = asyncio.create_task(receive(counted, _event(number=2), {}))
        await asyncio.sleep(0)
        assert not reset_entered.is_set() and not first.done()
        release.set()
        await reset
        assert first.cancelled()
        assert await stale is None
        assert await repo.get(TEST_USER_ID, datetime.now().timestamp()) is None
        handler.assert_not_awaited()
        assert not inputs._users
        await receive(counted, _event(number=3), {})
        stats = await repo.get(TEST_USER_ID, datetime.now().timestamp())
        assert stats and stats.total == 1
        handler.assert_awaited_once()

    asyncio.run(scenario())


def test_new_arrival_waits_for_cleanup_and_cancelled_waiter_leaves_no_entry() -> None:
    async def scenario() -> None:
        inputs = InputOperations()
        handler = AsyncMock(return_value="fresh")
        receive = ReceiveInputsMiddleware(inputs)
        async with inputs.resetting(TEST_USER_ID):
            task = asyncio.create_task(receive(handler, _event(), {}))
            cancelled = asyncio.create_task(receive(handler, _event(), {}))
            await asyncio.sleep(0)
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)
            handler.assert_not_awaited()
        assert await task == "fresh"
        assert not inputs._users

    asyncio.run(scenario())


def test_nested_receipts_and_failed_handler_release_tracking() -> None:
    async def scenario() -> None:
        inputs = InputOperations()
        async with inputs.receive(1) as outer:
            assert inputs.start_memory(outer)
            async with inputs.receive(1):
                pass
            assert outer.task in outer.entry.memory_tasks
            async with inputs.resetting(1, exempt=outer.task):
                assert not inputs.is_current(outer)
        assert not inputs._users
        with pytest.raises(ValueError):
            await ReceiveInputsMiddleware(inputs)(
                AsyncMock(side_effect=ValueError()), _event(), {}
            )
        assert not inputs._users

    asyncio.run(scenario())


def test_repeated_callback_cancellation_keeps_reset_barrier_until_worker_cleanup(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        inputs = InputOperations()
        started, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def old() -> None:
            async with inputs.receive(TEST_USER_ID) as ticket:
                assert inputs.start_memory(ticket)
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup.set()
                    await release.wait()

        pending = asyncio.create_task(old())
        await started.wait()
        reset = asyncio.create_task(_reset(engine, inputs, tmp_path))
        await cleanup.wait()
        reset.cancel()
        await asyncio.sleep(0)
        reset.cancel()
        handler = AsyncMock()
        fresh = asyncio.create_task(
            ReceiveInputsMiddleware(inputs)(handler, _event(), {})
        )
        await asyncio.sleep(0)
        assert not reset.done() and not fresh.done()
        handler.assert_not_awaited()
        release.set()
        results = await asyncio.gather(reset, pending, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        await fresh
        handler.assert_awaited_once()
        assert not inputs._users

    asyncio.run(scenario())


def test_non_user_events_and_standalone_memory_handler_are_unchanged() -> None:
    async def scenario() -> None:
        inputs = InputOperations()
        handler = AsyncMock(return_value="done")
        receive = ReceiveInputsMiddleware(inputs)
        for event in (
            TelegramObject(),
            _event().model_copy(update={"from_user": None}),
            _event().model_copy(
                update={"from_user": User(id=1, is_bot=True, first_name="Bot")}
            ),
        ):
            assert await receive(handler, event, {}) == "done"
        assert await MemoryInputsMiddleware(inputs)(handler, _event(), {}) == "done"
        assert not inputs._users

    asyncio.run(scenario())
