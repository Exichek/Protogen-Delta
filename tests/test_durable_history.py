"""История переживает рестарт, сохраняет изоляцию и удаляется по TTL/сбросу."""

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from test_response_engine import _create_engine

from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.repositories.user_state import UserStateRepository


def _store(path: Path, clock: list[float], limit: int = 8) -> UserStateStore:
    return UserStateStore(
        persistence=UserStateRepository(path),
        wall_clock=lambda: clock[0],
        history_limit=limit,
        history_ttl_seconds=100,
    )


def test_successful_history_reopens_in_its_own_scope(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = [1000.0]
        engine, _, _, _, _, _ = _create_engine()
        engine._user_states = _store(tmp_path, clock)
        for user, chat, marker in (
            (1, None, "PRIVATE"),
            (1, -10, "A"),
            (1, -20, "B"),
            (2, -10, "OTHER"),
        ):
            await engine.respond_and_deliver(user, marker, AsyncMock(), chat_id=chat)
        restored = _store(tmp_path, clock)
        for user, chat, marker in (
            (1, None, "PRIVATE"),
            (1, -10, "A"),
            (1, -20, "B"),
            (2, -10, "OTHER"),
        ):
            async with restored.use_conversation(user, chat) as state:
                assert tuple(state.history) == (ConversationTurn(marker, "Ответ"),)
                assert state.history_updated_at == 1000
        engine._user_states = restored
        await engine.respond(1, "Продолжай", chat_id=-10)
        model = engine._deepseek
        assert isinstance(model, AsyncMock) and model.chat.await_args is not None
        assert model.chat.await_args.kwargs["history"] == (
            ConversationTurn("A", "Ответ"),
        )

    asyncio.run(scenario())


def test_failed_or_cancelled_delivery_is_not_persisted(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = [1000.0]
        engine, _, _, _, _, _ = _create_engine()
        engine._user_states = _store(tmp_path, clock)
        for failure in (RuntimeError(), asyncio.CancelledError()):
            with pytest.raises(type(failure)):
                await engine.respond_and_deliver(
                    1, "UNDELIVERED", AsyncMock(side_effect=failure), chat_id=-10
                )
        async with _store(tmp_path, clock).use_conversation(1, -10) as restored:
            assert not restored.history

    asyncio.run(scenario())


def test_retention_and_setting_reads_do_not_refresh_conversation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        clock = [1000.0]
        states = _store(tmp_path, clock)
        async with states.use_conversation(1, -10) as state:
            state.history.append(ConversationTurn("OLD", "reply"))
            state.roleplay_character = "Persistent character"
        for now in (1020, 1050, 1099):
            clock[0] = now
            async with states.use_conversation(1, -10) as state:
                assert state.history_updated_at == 1000
                state.roleplay_configuration = "female"
        clock[0] = 1100
        async with states.use_conversation(1, -10) as state:
            assert not state.history
            assert state.roleplay_character == "Persistent character"
            assert state.roleplay_configuration == "female"
        row = await UserStateRepository(tmp_path).load((-10, 1))
        assert row and row.history == () and row.history_updated_at == 0

    asyncio.run(scenario())


def test_expiry_cleans_unloaded_histories_with_an_index(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = [1000.0]
        states = _store(tmp_path, clock)
        for chat in (None, -10):
            async with states.use_conversation(1, chat) as state:
                state.history.append(ConversationTurn("OLD", "reply"))
        clock[0] = 1200
        async with _store(tmp_path, clock).use(2):
            pass
        for key in (1, (-10, 1)):
            row = await UserStateRepository(tmp_path).load(key)
            assert row and not row.history
        with closing(sqlite3.connect(tmp_path / "user_states.db")) as db:
            plan = db.execute(
                "EXPLAIN QUERY PLAN UPDATE conversation_states SET history='[]' "
                "WHERE history_updated_at > 0 AND history_updated_at <= ?",
                (1100,),
            ).fetchall()
            assert any("history_age" in str(line) for line in plan)

    asyncio.run(scenario())


def test_history_stays_bounded_and_forgetting_survives_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = [1000.0]
        states = _store(tmp_path, clock, limit=2)
        async with states.use(1) as state:
            for i in range(5):
                state.history.append(ConversationTurn(str(i), "reply"))
        async with _store(tmp_path, clock, limit=2).use(1) as state:
            assert [turn.user_message for turn in state.history] == ["3", "4"]
            state.history.clear()
        async with _store(tmp_path, clock).use(1) as state:
            assert not state.history
            assert state.history_updated_at == 0
        async with states.use_conversation(1, -10) as group:
            group.history.append(ConversationTurn("GROUP", "reply"))
        await states.reset_user(1)
        async with _store(tmp_path, clock).use_conversation(1, -10) as state:
            assert not state.history

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "value", ["broken", "{}", '[{"user":1,"assistant":"x"}]', '[{"user":"x"}]']
)
def test_corrupt_history_does_not_break_profile(tmp_path: Path, value: str) -> None:
    async def scenario() -> None:
        clock = [1000.0]
        async with _store(tmp_path, clock).use(1) as state:
            state.roleplay_character = "Character"
        with closing(sqlite3.connect(tmp_path / "user_states.db")) as db, db:
            db.execute("UPDATE user_states SET history=? WHERE user_id=1", (value,))
        async with _store(tmp_path, clock).use(1) as state:
            assert not state.history
            assert state.roleplay_character == "Character"

    asyncio.run(scenario())


def test_db_snapshot_limits_individual_messages(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _store(tmp_path, [1000.0]).use(1) as state:
            state.history.append(ConversationTurn("x" * 9000, "y" * 20000))
        with closing(sqlite3.connect(tmp_path / "user_states.db")) as db:
            history = json.loads(
                db.execute(
                    "SELECT history FROM user_states WHERE user_id=1"
                ).fetchone()[0]
            )
            assert len(history[0]["user"]) == 8000
            assert len(history[0]["assistant"]) == 16000

    asyncio.run(scenario())


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan")])
def test_history_ttl_must_be_finite_and_positive(ttl: float) -> None:
    with pytest.raises(ValueError):
        UserStateStore(history_ttl_seconds=ttl)


def test_repeated_identical_delivered_turn_refreshes_ttl(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = [1000.0]
        engine, _, _, _, _, _ = _create_engine()
        engine._user_states = _store(tmp_path, clock, limit=1)
        await engine.respond(1, "Привет")
        clock[0] = 1099
        await engine.respond(1, "Привет")
        assert engine._user_states.get(1).history_updated_at == 1099
        clock[0] = 1101
        async with _store(tmp_path, clock, limit=1).use(1) as state:
            assert len(state.history) == 1
            assert state.history_updated_at == 1099

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["save", "reset"])
def test_cancelled_sqlite_write_finishes_before_reset_or_unlock(
    tmp_path: Path, operation: str
) -> None:
    async def scenario() -> None:
        repository = UserStateRepository(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        persistence = Mock(
            load=repository.load, save=repository.save, delete=repository.delete
        )
        store = UserStateStore(persistence=persistence, wall_clock=lambda: 1000)
        async with store.use(1) as state:
            state.history.append(ConversationTurn("OLD", "reply"))
            state.roleplay_active = True

        async def blocked(*args: Any, **kwargs: Any) -> None:
            entered.set()
            await release.wait()
            await (repository.save if operation == "save" else repository.delete)(
                *args, **kwargs
            )

        setattr(
            persistence,
            "save" if operation == "save" else "delete",
            AsyncMock(side_effect=blocked),
        )

        async def writer() -> None:
            async with store.use(1) as state:
                state.history.append(ConversationTurn("DELIVERED", "reply"))

        task = asyncio.create_task(
            writer() if operation == "save" else store.reset_user(1)
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert store.get(1).lock.locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        if operation == "save":
            await store.reset_user(1)
        assert not store.get(1).history and not store.get(1).roleplay_active
        assert await repository.load(1) is None

    asyncio.run(scenario())
