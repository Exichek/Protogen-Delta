"""Разные чаты идут параллельно, общие изменения дожидаются доставки и SQLite."""

import asyncio
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from test_reset_handler import _create_callback_mock
from test_response_engine import _create_engine

from protogen_delta.core.conversation_gate import ConversationGate
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.handlers.reset import RESET_SUCCESS_TEXT, create_reset_router
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_facts import FactChange, UserFactsRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.repositories.user_statistics import UserStatisticsRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.response_engine import ResponseBusyError
from protogen_delta.services.user_facts import UserFactsService


def test_private_and_two_groups_deliver_in_parallel_and_restore_separate_histories(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with engine._user_states.use(1) as state:
            state.content_mode = "adult"
        release = asyncio.Event()
        entered = {chat: asyncio.Event() for chat in (None, -10, -20)}

        async def deliver(chat: int | None, reply: str) -> None:
            assert reply == "Ответ"
            entered[chat].set()
            await release.wait()

        tasks = []
        for chat in entered:
            tasks.append(
                asyncio.create_task(
                    engine.respond_and_deliver(
                        1,
                        f"Привет из {chat}",
                        partial(deliver, chat),
                        chat_id=chat,
                    )
                )
            )
            # A blocked private delivery must not stop the next group.
            await asyncio.wait_for(entered[chat].wait(), 5)
        assert len(engine._delivering_users) == 3
        release.set()
        await asyncio.gather(*tasks)
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        for chat in entered:
            async with restored.use_conversation(1, chat) as state:
                assert state.content_mode == "adult"
                assert tuple(state.history) == (
                    ConversationTurn(f"Привет из {chat}", "Ответ"),
                )
        assert not engine._delivering_users

    asyncio.run(scenario())


@pytest.mark.parametrize("chat", [None, -10])
def test_busy_rejects_only_same_chat_and_cancellation_releases_it(
    chat: int | None,
) -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        entered, release = asyncio.Event(), asyncio.Event()

        async def deliver(reply: str) -> None:
            entered.set()
            await release.wait()

        active = asyncio.create_task(
            engine.respond_and_deliver(1, "Привет", deliver, chat_id=chat)
        )
        await entered.wait()
        with pytest.raises(ResponseBusyError):
            await engine.respond_and_deliver(1, "Ещё", AsyncMock(), chat_id=chat)
        await engine.respond_and_deliver(1, "Другая группа", AsyncMock(), chat_id=-20)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert not engine._user_states.get_conversation(1, chat).history
        await engine.respond_and_deliver(1, "Повтор", AsyncMock(), chat_id=chat)
        assert not engine._delivering_users

    asyncio.run(scenario())


def test_private_appearance_can_change_while_group_is_delivering() -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        entered, release = asyncio.Event(), asyncio.Event()

        async def deliver(reply: str) -> None:
            entered.set()
            await release.wait()

        task = asyncio.create_task(
            engine.respond_and_deliver(1, "Привет", deliver, chat_id=-10)
        )
        await entered.wait()
        await asyncio.wait_for(
            engine.set_delta_appearance_from_text(1, "Синий сергал"), 2
        )
        assert engine._user_states.get(1).delta_appearance == "Синий сергал"
        assert engine._user_states.get_conversation(1, -10).delta_appearance == ""
        release.set()
        await task

    asyncio.run(scenario())


def test_age_change_waits_all_active_chats_and_precedes_new_replies() -> None:
    async def scenario() -> None:
        states = UserStateStore()
        async with states.use(1) as state:
            state.content_mode = "adult"
        private_entered, group_entered = asyncio.Event(), asyncio.Event()
        private_release, group_release = asyncio.Event(), asyncio.Event()
        change_requested, change_entered = asyncio.Event(), asyncio.Event()
        change_release, new_entered = asyncio.Event(), asyncio.Event()

        async def conversation(
            chat: int | None, entered: asyncio.Event, release: asyncio.Event
        ) -> None:
            async with states.use_conversation(1, chat) as state:
                assert state.content_mode == "adult"
                entered.set()
                await release.wait()
                assert state.content_mode == "adult"

        async def change() -> None:
            change_requested.set()
            async with states.use(1) as state:
                state.content_mode = "soft"
                change_entered.set()
                await change_release.wait()

        async def new_conversation() -> None:
            async with states.use_conversation(1, -20) as state:
                assert state.content_mode == "soft"
                new_entered.set()

        private = asyncio.create_task(
            conversation(None, private_entered, private_release)
        )
        group = asyncio.create_task(conversation(-10, group_entered, group_release))
        await asyncio.wait_for(
            asyncio.gather(private_entered.wait(), group_entered.wait()), 2
        )
        writer = asyncio.create_task(change())
        await change_requested.wait()
        later = asyncio.create_task(new_conversation())
        await asyncio.sleep(0)
        assert not new_entered.is_set()
        private_release.set()
        await private
        assert not change_entered.is_set()
        group_release.set()
        await group
        await asyncio.wait_for(change_entered.wait(), 2)
        assert not new_entered.is_set()
        change_release.set()
        await asyncio.gather(writer, later)

    asyncio.run(scenario())


def test_full_reset_waits_both_deliveries_and_removes_every_scope(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, _, _, _, _ = _create_engine()
        repository = UserStateRepository(tmp_path)
        engine._user_states = UserStateStore(persistence=repository)
        for chat in (None, -10, -20):
            async with engine._user_states.use_conversation(1, chat) as state:
                state.roleplay_active = True
                state.delta_appearance = "old"
        async with engine._user_states.use_conversation(2, -10) as state:
            state.delta_appearance = "other"
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def deliver(index: int, reply: str) -> None:
            entered[index].set()
            await release[index].wait()

        responses = [
            asyncio.create_task(
                engine.respond_and_deliver(
                    1, "Привет", partial(deliver, i), chat_id=chat
                )
            )
            for i, chat in enumerate((None, -10))
        ]
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered)), 5)
        reset = asyncio.create_task(engine.reset_user(1))
        await asyncio.sleep(0)
        release[0].set()
        await responses[0]
        assert not reset.done()
        release[1].set()
        await asyncio.gather(responses[1], reset)
        for key in (1, (-10, 1), (-20, 1)):
            assert await repository.load(key) is None
        other = await repository.load((-10, 2))
        assert other and other.delta_appearance == "other"
        for chat in (None, -10, -20):
            state = engine._user_states.get_conversation(1, chat)
            assert not state.history and not state.roleplay_active
            assert state.delta_appearance == ""
            assert state.content_mode == "unselected"

    asyncio.run(scenario())


def test_cancelled_sql_save_keeps_global_reset_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        repository = UserStateRepository(tmp_path)
        states = UserStateStore(persistence=repository)
        original_save = repository.save
        entered, release = asyncio.Event(), asyncio.Event()

        async def save(*args: Any, **kwargs: Any) -> None:
            entered.set()
            await release.wait()
            await original_save(*args, **kwargs)

        monkeypatch.setattr(repository, "save", save)

        async def conversation() -> None:
            async with states.use_conversation(1, -10) as state:
                state.history.append(ConversationTurn("old", "reply"))

        task = asyncio.create_task(conversation())
        await entered.wait()
        task.cancel()
        reset = asyncio.create_task(states.reset_user(1))
        await asyncio.sleep(0)
        assert not reset.done()
        task.cancel()  # Repeated cancellation must not release the gate early.
        await asyncio.sleep(0)
        assert not reset.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await reset
        assert await repository.load((-10, 1)) is None
        assert states.get(1).coordinating_operations == 0

    asyncio.run(scenario())


def test_cancelled_waiting_writer_does_not_block_new_groups() -> None:
    async def scenario() -> None:
        states = UserStateStore()
        active, release, requested = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def first() -> None:
            async with states.use_conversation(1, -10):
                active.set()
                await release.wait()

        async def writer() -> None:
            requested.set()
            async with states.use(1):
                pytest.fail("Cancelled writer must not enter")

        task = asyncio.create_task(first())
        await active.wait()
        change = asyncio.create_task(writer())
        await requested.wait()
        change.cancel()
        with pytest.raises(asyncio.CancelledError):
            await change

        async def second() -> None:
            async with states.use_conversation(1, -20):
                pass

        await asyncio.wait_for(second(), 2)
        release.set()
        await task
        assert states.get(1).coordinating_operations == 0

    asyncio.run(scenario())


def test_gate_survives_cache_expiry_while_group_or_writer_is_waiting() -> None:
    async def scenario() -> None:
        now = [0.0]
        states = UserStateStore(retention_seconds=10, clock=lambda: now[0])
        private = states.get(1)
        async with states.use_conversation(1, -10):
            now[0] = 11
            states.get(2)
            assert states.get(1) is private
            assert not states.remove(1)
        assert private.coordinating_operations == 0
        now[0] = 22
        states.get(2)
        assert 1 not in states._states

    asyncio.run(scenario())


def test_cancelled_waiting_reader_and_failed_writer_release_coordination() -> None:
    async def scenario() -> None:
        states = UserStateStore()
        async with states.use(1):

            async def reader() -> None:
                async with states.use_conversation(1, -10):
                    pytest.fail("Cancelled reader must not enter")

            task = asyncio.create_task(reader())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        with pytest.raises(RuntimeError):
            async with states.use(1):
                raise RuntimeError("failed")
        async with states.use_conversation(1, -10):
            pass
        assert states.get(1).coordinating_operations == 0

    asyncio.run(scenario())


def test_cold_private_load_does_not_hold_group_until_private_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        repository = UserStateRepository(tmp_path)
        async with UserStateStore(persistence=repository).use(1) as state:
            state.content_mode = "adult"
        engine, _, _, _, _, _ = _create_engine()
        engine._user_states = UserStateStore(persistence=repository)
        original_load = repository.load
        loading, loaded = asyncio.Event(), asyncio.Event()
        private_entered, group_entered = asyncio.Event(), asyncio.Event()
        release = asyncio.Event()

        async def load(key: Any) -> Any:
            if key == 1:
                loading.set()
                await loaded.wait()
            return await original_load(key)

        monkeypatch.setattr(repository, "load", load)

        async def deliver(entered: asyncio.Event, reply: str) -> None:
            entered.set()
            await release.wait()

        private = asyncio.create_task(
            engine.respond_and_deliver(1, "Привет", partial(deliver, private_entered))
        )
        await loading.wait()
        group = asyncio.create_task(
            engine.respond_and_deliver(
                1, "Привет", partial(deliver, group_entered), chat_id=-10
            )
        )
        await asyncio.sleep(0)
        loaded.set()
        await asyncio.wait_for(
            asyncio.gather(private_entered.wait(), group_entered.wait()), 5
        )
        assert engine._user_states.get_conversation(1, -10).content_mode == "adult"
        release.set()
        await asyncio.gather(private, group)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_reset_router_protects_facts_statistics_and_registration_until_cleanup_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("protogen_delta.handlers.reset.time", lambda: 1000.0)
        engine, _, model, _, _, _ = _create_engine()
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine._user_states = states
        users = UsersRepository(tmp_path)
        facts = UserFactsRepository(tmp_path)
        episodes = MemoriesRepository(tmp_path)
        statistics = UserStatisticsRepository(tmp_path)
        memory = MemoryService(episodes, facts=UserFactsService(facts, model))
        for user in (1, 2):
            users.add(user)
            await facts.apply(
                user, [FactChange("name", "add", "Test", "name Test")], 1000
            )
            await episodes.remember(user, "topic", "OLD_TOPIC", 1000)
            await statistics.record(user, "Test", "test", user, 1, "text", 1000)
        async with states.use(1) as state:
            state.content_mode = "adult"
            state.history.append(ConversationTurn("old", "reply"))
        clearing_facts, clearing_statistics = asyncio.Event(), asyncio.Event()
        facts_release, statistics_release = asyncio.Event(), asyncio.Event()
        original_clear, original_statistics_delete = facts.clear, statistics.delete_user

        async def clear_facts(*args: Any, **kwargs: Any) -> Any:
            clearing_facts.set()
            await facts_release.wait()
            return await original_clear(*args, **kwargs)

        async def clear_statistics(user_id: int) -> None:
            clearing_statistics.set()
            await statistics_release.wait()
            await original_statistics_delete(user_id)

        monkeypatch.setattr(facts, "clear", clear_facts)
        monkeypatch.setattr(statistics, "delete_user", clear_statistics)

        async def update_menu(user_id: int, mode: str) -> None:
            # This callback may use the store only after the reset barrier ends.
            async with states.use(user_id) as state:
                assert state.content_mode == mode == "unselected"

        router = create_reset_router(engine, users, memory, update_menu, statistics)
        callback, edit, _ = _create_callback_mock("reset:confirm:1:1000", user_id=1)
        reset = asyncio.create_task(
            router.callback_query.handlers[0].callback(callback)
        )
        await clearing_facts.wait()
        new_entered = asyncio.Event()

        async def new_read() -> None:
            async with states.use_conversation(1, -10) as state:
                assert state.content_mode == "unselected"
                assert not state.history
                assert not await facts.all(1)
                assert not await episodes.recent(1)
                assert await statistics.get(1, 1000) is None
                assert 1 not in users.get_all()
                new_entered.set()

        later = asyncio.create_task(new_read())
        await asyncio.sleep(0)
        assert not new_entered.is_set()
        facts_release.set()
        await clearing_statistics.wait()
        assert not new_entered.is_set()
        if cancel:
            reset.cancel()
            await asyncio.sleep(0)
            reset.cancel()
            await asyncio.sleep(0)
            assert not reset.done()
        statistics_release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await reset
            edit.assert_not_awaited()
        else:
            await reset
            edit.assert_awaited_once_with(RESET_SUCCESS_TEXT, reply_markup=None)
        await asyncio.wait_for(later, 5)
        assert await facts.all(2)
        assert await episodes.recent(2)
        assert await statistics.get(2, 1000) is not None
        assert 2 in users.get_all()

    asyncio.run(scenario())


def test_gate_cleanup_survives_repeated_cancellation() -> None:
    async def scenario() -> None:
        gate = ConversationGate()
        entered, release = asyncio.Event(), asyncio.Event()

        async def reader() -> None:
            async with gate.shared():
                entered.set()
                await release.wait()

        task = asyncio.create_task(reader())
        await entered.wait()
        # Delay cleanup using the condition's own mutex to expose its await.
        async with gate._condition:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(2):
            async with gate.exclusive():
                pass

    asyncio.run(scenario())
