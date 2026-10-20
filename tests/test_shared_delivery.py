"""Проверки общей памяти, доставки, отмены и сохранения авторства."""

import asyncio
import json
import sqlite3
from contextlib import closing
from contextvars import Context
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage
from aiogram.types import TelegramObject
from test_dialogue_archive import message, telegram

from protogen_delta.config.settings import _parse_dialogue_groups
from protogen_delta.core.input_operations import InputOperations
from protogen_delta.core.shared_dialogue import SharedDialogueAccounts
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.handlers.delivery_ledger import DeliveryLedgerMiddleware
from protogen_delta.handlers.dialogue_archive import (
    ArchiveIncomingMiddleware,
    ArchiveOutgoingMiddleware,
)
from protogen_delta.repositories.delivery_ledger import DeliveryLedgerRepository
from protogen_delta.repositories.dialogue_archive import DialogueArchiveRepository
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_facts import FactChange, UserFactsRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.delivery_ledger import current_delivery, recovery_context
from protogen_delta.services.dialogue_archive import (
    DialogueArchiveService,
    dialogue_owner,
)
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.user_facts import UserFactsService


def state_of(path: Path, key: str) -> tuple[str, int]:
    with closing(sqlite3.connect(path / "delivery_ledger.db")) as db:
        row = db.execute(
            "SELECT state,confirmed_chunks FROM delivery_ledger WHERE source_key=?",
            (key,),
        ).fetchone()
        assert row is not None
        return cast(tuple[str, int], row)


def test_claim_is_atomic_across_parallel_handlers_and_restart(tmp_path: Path) -> None:
    async def run() -> None:
        repo = DeliveryLedgerRepository(tmp_path)
        claims = await asyncio.gather(
            *(repo.claim(1, "telegram:101:1:7") for _ in range(12))
        )
        assert sum(claims) == 1
        await repo.mark("telegram:101:1:7", "processing")
        restarted = DeliveryLedgerRepository(tmp_path)
        assert state_of(tmp_path, "telegram:101:1:7")[0] == "interrupted"
        assert not await restarted.claim(1, "telegram:101:1:7")
        assert await restarted.claim(1, "telegram:202:1:7")
        with closing(sqlite3.connect(tmp_path / "delivery_ledger.db")) as db:
            assert "text" not in {
                row[1] for row in db.execute("PRAGMA table_info(delivery_ledger)")
            }

    asyncio.run(run())


def test_pending_recovery_uses_received_question_not_notice_or_action(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        archive = DialogueArchiveRepository(tmp_path)
        repo = DeliveryLedgerRepository(tmp_path)
        bot = Bot("101:fake-token")
        incoming = ArchiveIncomingMiddleware(
            DialogueArchiveService(archive, frozenset({1}), tmp_path), InputOperations()
        )
        ledger = DeliveryLedgerMiddleware(repo)
        calls = 0

        async def busy(event: TelegramObject, data: dict[str, Any]) -> None:
            nonlocal calls
            calls += 1
            assert current_delivery.get() is not None

        async def archived(event: TelegramObject, data: dict[str, Any]) -> Any:
            return await incoming(busy, event, data)

        try:
            await ledger(
                archived,
                telegram(text="Сколько будет 28 умножить на 42?"),
                {"bot": bot},
            )
            await ledger(
                archived,
                telegram(text="Сколько будет 28 умножить на 42?"),
                {"bot": bot},
            )
            assert calls == 1
            assert state_of(tmp_path, "telegram:101:1:1")[0] == "deferred"
            lines = await recovery_context(
                repo, archive, 1, "Я выше писал, ты не ответил"
            )
            assert "28 умножить на 42" in " ".join(lines)
            assert not await recovery_context(repo, archive, 1, "Привет")
            assert "не найден" in " ".join(
                await recovery_context(repo, archive, 2, "ты не ответил")
            )
            await archive.save(1, message("telegram:101:1:2", "/download PRIVATE_URL"))
            await repo.claim(1, "telegram:101:1:2")
            await repo.mark("telegram:101:1:2", "failed")
            assert "PRIVATE_URL" not in " ".join(
                await recovery_context(repo, archive, 1, "ты не ответил")
            )
            await repo.forget(frozenset({1}))
            assert not await repo.claim(1, "telegram:101:1:1")
            await repo.mark("telegram:101:1:1", "answered")
            assert state_of(tmp_path, "telegram:101:1:1")[0] == "forgotten"
        finally:
            await bot.session.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure,expected,confirmed",
    [
        ("none", "answered", 2),
        ("network", "delivery_unknown", 1),
        ("bad", "partial", 1),
    ],
)
def test_confirmed_chunks_and_ack_loss(
    tmp_path: Path, failure: str, expected: str, confirmed: int
) -> None:
    async def run() -> None:
        repo = DeliveryLedgerRepository(tmp_path)
        archive = DialogueArchiveService(
            DialogueArchiveRepository(tmp_path), frozenset({1}), tmp_path
        )
        sender = ArchiveOutgoingMiddleware(archive, repo)
        bot = Bot("101:fake-token")

        async def handler(event: TelegramObject, data: dict[str, Any]) -> None:
            operation = current_delivery.get()
            assert operation is not None
            operation.answering = operation.reply_started = True
            for index in range(2):
                method = SendMessage(chat_id=1, text="Ответ")
                if index and failure == "network":
                    callback = AsyncMock(side_effect=TimeoutError())
                elif index and failure == "bad":
                    callback = AsyncMock(
                        side_effect=TelegramBadRequest(method, "rejected")
                    )
                else:
                    callback = AsyncMock(
                        return_value=telegram(mid=10 + index, text="Ответ", bot=True)
                    )
                await sender(callback, bot, method)
            operation.completed = True

        try:
            ledger = DeliveryLedgerMiddleware(repo)
            if failure == "none":
                await ledger(handler, telegram(), {"bot": bot})
            else:
                with pytest.raises((TimeoutError, TelegramBadRequest)):
                    await ledger(handler, telegram(), {"bot": bot})
            assert state_of(tmp_path, "telegram:101:1:1") == (expected, confirmed)
            restarted = DeliveryLedgerRepository(tmp_path)
            assert not await restarted.claim(1, "telegram:101:1:1")
            assert state_of(tmp_path, "telegram:101:1:1") == (expected, confirmed)
        finally:
            await bot.session.close()

    asyncio.run(run())


def test_shared_search_preserves_authors_and_cannot_choose_foreign_owner(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = DialogueArchiveRepository(tmp_path)
        shared = SharedDialogueAccounts((frozenset({1, 2}),))
        service = DialogueArchiveService(repo, frozenset({1, 2, 3}), tmp_path, shared)
        await repo.save(1, message("one", "Сигнал ORBIT_ONE", at=1000))
        await repo.save(2, message("two", "Сигнал ORBIT_TWO", at=2000))
        await repo.save(3, message("three", "Сигнал ORBIT_PRIVATE", at=3000))
        for owner in (1, 2):
            with dialogue_owner(owner):
                result = await service.search_tool({"query": "Сигнал", "user_id": "3"})
            assert result["scope"] == "shared_private_dialogue"
            assert {row["account_id"] for row in result["messages"]} == {1, 2}
            context = " ".join(await service.context(owner, "Не совпадающее слово"))
            assert (
                "ORBIT_ONE" in context
                and "ORBIT_TWO" in context
                and "ORBIT_PRIVATE" not in context
            )
        destination = tmp_path / "export.jsonl"
        await repo.export(2, destination)
        assert json.loads(destination.read_text("utf-8").splitlines()[0]) == {
            "schema": 1,
            "user_id": 2,
        }
        assert "account_id" not in destination.read_text("utf-8")
        old_generation = service.generation(2)
        await service.delete_user(1)
        await service.record(2, message("late", "Сигнал ORBIT_LATE"), old_generation)
        assert not await repo.search(1, "Сигнал") and not await repo.search(2, "Сигнал")
        assert await repo.search(3, "Сигнал")

    asyncio.run(run())


@pytest.mark.parametrize(
    "value", ["1", "1,2;2,3", "1,0", "1,9223372036854775808", "1,1", "abc,2"]
)
def test_invalid_shared_config(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_dialogue_groups(value)


def test_empty_or_explicit_shared_config() -> None:
    assert _parse_dialogue_groups("") == ()
    assert _parse_dialogue_groups("1,2;3,4") == (frozenset({1, 2}), frozenset({3, 4}))


def test_peer_context_deleted_on_reset_without_settings_transfer(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        shared = SharedDialogueAccounts((frozenset({1, 2}),))
        persistence = UserStateRepository(tmp_path)
        states = UserStateStore(persistence=persistence, shared_dialogue=shared)
        repo = DialogueArchiveRepository(tmp_path)
        archive = DialogueArchiveService(repo, frozenset({1, 2}), tmp_path, shared)
        facts = UserFactsRepository(tmp_path)
        memory = MemoryService(
            MemoriesRepository(tmp_path),
            dialogue_archive=archive,
            facts=UserFactsService(facts, AsyncMock()),
            states=states,
        )
        for owner in (1, 2):
            async with states.use(owner) as state:
                state.content_mode = "soft"
                state.roleplay_allow_profanity = owner == 2
                state.history.append(ConversationTurn("PRIVATE_OLD", "PRIVATE_REPLY"))
                states.mark_history_updated(state)
            await repo.save(owner, message(f"{owner}", "PRIVATE_OLD"))
        async with states.use_conversation(2, -10) as state:
            state.history.append(ConversationTurn("PRIVATE_GROUP", "PRIVATE_REPLY"))
            states.mark_history_updated(state)
        await states.reset_user(1, cleanup=lambda: memory.delete_user(1))
        restored = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with restored.use(2) as state:
            assert not state.history
            assert state.content_mode == "soft" and state.roleplay_allow_profanity
        async with restored.use_conversation(2, -10) as state:
            assert not state.history
        assert not await repo.search(2, "PRIVATE")

    asyncio.run(run())


def test_forgetting_fact_removes_peer_archive_and_profile(tmp_path: Path) -> None:
    async def run() -> None:
        shared = SharedDialogueAccounts((frozenset({1, 2}),))
        repo = DialogueArchiveRepository(tmp_path)
        facts = UserFactsRepository(tmp_path)
        memory = MemoryService(
            MemoriesRepository(tmp_path),
            dialogue_archive=DialogueArchiveService(
                repo, frozenset({1, 2}), tmp_path, shared
            ),
            facts=UserFactsService(facts, AsyncMock()),
        )
        for owner in (1, 2):
            await facts.apply(
                owner, [FactChange("city", "add", "Тула", "Я живу в Туле")], 1000
            )
            await repo.save(owner, message(str(owner), "Я живу в Туле"))
        await memory.forget_facts(1, "city")
        assert not await facts.all(1) and not await facts.all(2)
        assert not await repo.search(1, "Тула") and not await repo.search(2, "Тула")

    asyncio.run(run())


def test_group_reset_cancels_peer_pipeline_and_serializes_state() -> None:
    async def run() -> None:
        shared = SharedDialogueAccounts((frozenset({1, 2}),))
        inputs = InputOperations(shared)
        started = asyncio.Event()

        async def old() -> None:
            async with inputs.receive(2) as ticket:
                assert inputs.start_memory(ticket)
                started.set()
                await asyncio.Event().wait()

        pending = asyncio.create_task(old())
        await started.wait()
        async with inputs.resetting(1):
            assert pending.cancelled()
        states = UserStateStore(shared_dialogue=shared)
        entered = asyncio.Event()

        async def peer() -> None:
            async with states.use(2):
                entered.set()

        async with states.use(1):
            task = asyncio.create_task(peer(), context=Context())
            await asyncio.sleep(0)
            assert not entered.is_set()
        await task
        assert entered.is_set() and not inputs._users

    asyncio.run(run())


def test_recovery_is_passed_to_model_only_in_eligible_private_chat(
    tmp_path: Path,
) -> None:
    from test_response_engine import TEST_USER_ID, _create_engine

    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        repo = DialogueArchiveRepository(tmp_path)
        ledger = DeliveryLedgerRepository(tmp_path)
        key = f"telegram:101:{TEST_USER_ID}:1"
        await repo.save(
            TEST_USER_ID, message(key, "RECOVERY_REAL_QUESTION: что такое FTS5?")
        )
        await ledger.claim(TEST_USER_ID, key)
        await ledger.mark(key, "failed")
        engine._memory = MemoryService(
            MemoriesRepository(tmp_path),
            dialogue_archive=DialogueArchiveService(
                repo, frozenset({TEST_USER_ID}), tmp_path
            ),
            delivery_ledger=ledger,
        )
        for scope in (None, -10):
            await engine.respond(
                TEST_USER_ID, "Я выше писал, ты не ответил", chat_id=scope
            )
            assert model.chat.await_args is not None
            prompt = model.chat.await_args.kwargs["system_prompt"]
            assert ("RECOVERY_REAL_QUESTION" in prompt) == (scope is None)
        await repo.save(TEST_USER_ID, message(key, "Мне 15 лет, SECRET_OLD_MINOR_BODY"))
        lines = await recovery_context(ledger, repo, TEST_USER_ID, "ты не ответил")
        assert "SECRET_OLD_MINOR_BODY" not in " ".join(lines)
        assert "несовершеннолетнего" in " ".join(lines)

    asyncio.run(run())


def test_ledger_errors_do_not_repeat_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        bot = Bot("101:fake-token")
        repo = DeliveryLedgerRepository(tmp_path)
        middleware = DeliveryLedgerMiddleware(repo)
        handler = AsyncMock()
        try:
            assert (
                await middleware(handler, TelegramObject(), {}) is handler.return_value
            )
            assert (
                await middleware(handler, telegram(bot=True), {"bot": bot})
                is handler.return_value
            )
            handler.reset_mock()
            failing = AsyncMock(side_effect=sqlite3.OperationalError("synthetic"))
            original = repo.claim
            monkeypatch.setattr(repo, "claim", failing)
            assert await middleware(handler, telegram(), {"bot": bot}) is None
            handler.assert_not_awaited()
            monkeypatch.setattr(repo, "claim", original)
            original_mark = repo.mark

            async def mark(key: str, state: str, confirmed: int = 0) -> None:
                if state == "handled":
                    raise sqlite3.OperationalError("synthetic")
                await original_mark(key, state, confirmed)

            monkeypatch.setattr(repo, "mark", mark)
            await middleware(handler, telegram(text="/status"), {"bot": bot})
            await middleware(handler, telegram(text="/status"), {"bot": bot})
            handler.assert_awaited_once()
            assert current_delivery.get() is None
            with pytest.raises(ValueError):
                await original(0, "telegram:101:1:2")
            with pytest.raises(ValueError):
                await original_mark("telegram:101:1:1", "invalid")
        finally:
            await bot.session.close()

    asyncio.run(run())


def test_linked_reset_router_stops_peer_before_state_barrier(tmp_path: Path) -> None:
    from time import time

    from test_reset_handler import _create_callback_mock
    from test_response_engine import _create_engine

    from protogen_delta.handlers.reset import create_reset_router
    from protogen_delta.repositories.users import UsersRepository

    async def run() -> None:
        shared = SharedDialogueAccounts((frozenset({42, 7}),))
        states = UserStateStore(shared_dialogue=shared)
        inputs = InputOperations(shared)
        engine, *_ = _create_engine()
        engine._user_states = states
        archive = DialogueArchiveService(
            DialogueArchiveRepository(tmp_path), frozenset({42, 7}), tmp_path, shared
        )
        memory = MemoryService(
            MemoriesRepository(tmp_path),
            dialogue_archive=archive,
            states=states,
            inputs=inputs,
        )
        router = create_reset_router(
            engine, UsersRepository(tmp_path), memory, input_operations=inputs
        )
        started = asyncio.Event()

        async def pending() -> None:
            async with inputs.receive(7) as ticket:
                assert inputs.start_memory(ticket)
                async with states.use_conversation(7):
                    started.set()
                    await asyncio.Event().wait()

        task = asyncio.create_task(pending())
        await started.wait()
        callback, _, _ = _create_callback_mock(
            f"reset:confirm:42:{int(time())}", user_id=42
        )
        async with asyncio.timeout(2):
            await router.callback_query.handlers[0].callback(callback)
        assert task.cancelled() and not inputs._users
        await archive.close()

    asyncio.run(run())
