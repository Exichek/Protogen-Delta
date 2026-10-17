"""Стабильный срок после паузы, один повтор и память собственной инициативы."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, User

from protogen_delta.core.input_operations import InputOperations
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.proactive_activity import (
    ProactiveIncomingMiddleware,
    ProactiveOutgoingMiddleware,
    proactive_delivery,
)
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.proactive import (
    ProactiveConfig,
    ProactiveMessenger,
    _moscow_datetime,
)


def test_model_history_has_assistant_only_for_saved_initiative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_tool_loop import response

    import protogen_delta.services.deepseek as module
    from protogen_delta.core.user_state import ConversationTurn
    from protogen_delta.services.prompt_composer import PromptComposer

    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=response())
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService("test", "https://unused.invalid", "test")
    turn = ConversationTurn("", "Хочешь обсудить наш проект?", proactive=True)
    asyncio.run(service.chat("core", "Ты мне уже писал?", history=(turn,)))
    messages = client.chat.completions.create.await_args.kwargs["messages"]
    assert [item["content"] for item in messages if item["role"] == "user"] == [
        "Ты мне уже писал?"
    ]
    assert any(
        item["role"] == "assistant" and item["content"] == turn.assistant_message
        for item in messages
    )
    summary = PromptComposer.compact_history(
        (turn, ConversationTurn("Ответ", "Продолжение")), live_turns=1
    ).summary
    assert "Самостоятельная реплика Дельты" in summary
    assert "Пользователь: ''" not in summary


def deadline(root: Path) -> tuple[float | None, int]:
    with closing(sqlite3.connect(root / "memories.db")) as db:
        return cast(
            tuple[float | None, int],
            db.execute(
                "SELECT next_proactive_at,unanswered_count FROM engagement WHERE user_id=7"
            ).fetchone(),
        )


async def due(repo: MemoriesRepository, now: float) -> list[Any]:
    return await repo.due_candidates(
        now=now,
        idle_seconds=5400,
        idle_max_seconds=10800,
        cooldown_seconds=72000,
        limit=10,
    )


@pytest.mark.parametrize("delay", [5400, 7200, 10800])
def test_random_deadline_is_chosen_once_and_survives_restart(
    tmp_path: Path, delay: float
) -> None:
    async def run() -> None:
        random = Mock(return_value=delay)
        repo = MemoriesRepository(tmp_path, random_delay=random)
        await repo.note_user_activity(7, 100)
        assert not await due(repo, 100)
        assert deadline(tmp_path) == (100 + delay, 0)
        assert not await due(repo, 100 + delay - 1)
        reopened = MemoriesRepository(
            tmp_path, random_delay=Mock(side_effect=AssertionError("must not reroll"))
        )
        assert [item.user_id for item in await due(reopened, 100 + delay)] == [7]
        random.assert_called_once_with(5400, 10800)

    asyncio.run(run())


def test_one_followup_after_twenty_hours_then_wait_for_user(tmp_path: Path) -> None:
    async def run() -> None:
        repo = MemoriesRepository(tmp_path, random_delay=lambda *_: 5400)
        await repo.note_user_activity(7, 100)
        first = (await due(repo, 5500))[0]
        await repo.note_proactive_sent(7, 5500, activity_version=first.activity_version)
        assert deadline(tmp_path) == (77500, 1)
        await repo.note_dialogue_activity(
            7, 7000
        )  # Другая доставка не считается ответом человека.
        assert not await due(repo, 77499)
        second = (await due(repo, 77500))[0]
        await repo.note_proactive_sent(
            7, 77500, activity_version=second.activity_version
        )
        assert deadline(tmp_path) == (None, 2)
        assert not await due(repo, 99999999)
        await repo.note_user_activity(7, 100000)
        assert not await due(repo, 105399)
        assert deadline(tmp_path) == (105400, 0)
        assert await due(repo, 105400)

    asyncio.run(run())


def test_timer_starts_after_delivered_reply_and_stale_generation_is_cancelled(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = MemoriesRepository(tmp_path, random_delay=lambda *_: 5400)
        await repo.note_user_activity(7, 100)
        await repo.note_dialogue_activity(7, 200)
        assert not await due(repo, 5599)
        old = (await due(repo, 5600))[0]
        await repo.note_user_activity(7, 5601)
        assert not await repo.is_due(
            7,
            now=6000,
            idle_seconds=5400,
            idle_max_seconds=10800,
            cooldown_seconds=72000,
            activity_version=old.activity_version,
        )
        await repo.note_proactive_sent(7, 6000, activity_version=old.activity_version)
        assert deadline(tmp_path) == (11001, 0)
        await repo.delete_user(7)
        await repo.note_dialogue_activity(7, 8000)
        await repo.note_proactive_sent(7, 8000)
        assert not await due(repo, 999999)
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute("SELECT COUNT(*) FROM engagement").fetchone()[0] == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "minimum,maximum", [(0, 1), (-1, 1), (10, 5), (float("nan"), 10), (5, float("inf"))]
)
def test_invalid_schedule_is_rejected(
    tmp_path: Path, minimum: float, maximum: float
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            MemoriesRepository(tmp_path).due_candidates(
                now=1,
                idle_seconds=minimum,
                idle_max_seconds=maximum,
                cooldown_seconds=72000,
                limit=1,
            )
        )


@pytest.mark.parametrize("random_value", [5399, 10801, float("nan")])
def test_random_source_cannot_violate_minimum_or_maximum(
    tmp_path: Path, random_value: float
) -> None:
    async def run() -> None:
        repo = MemoriesRepository(tmp_path, random_delay=lambda *_: random_value)
        await repo.note_user_activity(7, 100)
        with pytest.raises(ValueError):
            await due(repo, 200)
        assert deadline(tmp_path) == (None, 0)

    asyncio.run(run())


@pytest.mark.parametrize("activity_during_generation", [False, True])
def test_delivered_initiative_survives_restart_without_fake_user_reply(
    tmp_path: Path, activity_during_generation: bool
) -> None:
    async def run() -> None:
        now = [5500.0]
        repo = MemoriesRepository(tmp_path, random_delay=lambda *_: 5400)
        persistence = UserStateRepository(tmp_path)
        states = UserStateStore(persistence=persistence, wall_clock=lambda: now[0])
        bot = AsyncMock(spec=Bot)
        model = AsyncMock(spec=DeepSeekService)

        async def generate(**_: Any) -> str:
            if activity_during_generation:
                await repo.note_user_activity(7, now[0] + 1)
            return "Хочешь обсудить наш проект?"

        model.chat.side_effect = generate
        service = ProactiveMessenger(
            bot=cast(Bot, bot),
            deepseek=cast(DeepSeekService, model),
            repository=repo,
            user_states=states,
            system_prompt="core",
            config=ProactiveConfig(
                idle_seconds=5400, idle_max_seconds=10800, cooldown_seconds=72000
            ),
            clock=lambda: now[0],
            local_datetime=lambda: datetime(2026, 10, 13, 12),
        )
        await repo.note_user_activity(7, 100)
        counts = await asyncio.gather(service.run_once(), service.run_once())
        assert sum(counts) == (0 if activity_during_generation else 1)
        row = await persistence.load(7)
        assert row is not None
        if activity_during_generation:
            bot.send_message.assert_not_awaited()
            assert not row.history
        else:
            assert len(row.history) == 1 and row.history[0].proactive
            assert (
                row.history[0].user_message == ""
                and row.history[0].assistant_message == "Хочешь обсудить наш проект?"
            )
            assert row.history_updated_at == 5500
            restored = UserStateStore(
                persistence=persistence, wall_clock=lambda: now[0]
            )
            async with restored.use(7) as state:
                assert state.history[-1].proactive
            bot.send_message.assert_awaited_once()
        assert proactive_delivery.get() is False

    asyncio.run(run())


@pytest.mark.parametrize("change", ["none", "message", "mode"])
def test_emotion_decay_during_generation_does_not_cancel_initiative(
    tmp_path: Path, change: str
) -> None:
    """Затухание эмоций не отменяет отправку; новый ввод и смена режима отменяют."""

    async def run() -> None:
        now = [5500.0]
        repo = MemoriesRepository(tmp_path, random_delay=lambda *_: 5400)
        persistence = UserStateRepository(tmp_path)
        states = UserStateStore(persistence=persistence, wall_clock=lambda: now[0])
        async with states.use_conversation(7) as state:
            state.emotions.memory.joy = 0.8
            state.emotions.memory.tiredness = 0.4
            state.emotions.warmth = 0.7
        bot = AsyncMock(spec=Bot)
        model = AsyncMock(spec=DeepSeekService)

        async def generate(**_: Any) -> str:
            now[0] += 10
            if change == "message":
                await repo.note_user_activity(7, now[0])
            elif change == "mode":
                async with states.use_conversation(7) as current:
                    current.content_mode = "adult"
            return "Есть настроение поболтать :3"

        model.chat.side_effect = generate
        messenger = ProactiveMessenger(
            bot=cast(Bot, bot),
            deepseek=cast(DeepSeekService, model),
            repository=repo,
            user_states=states,
            system_prompt="core",
            config=ProactiveConfig(idle_seconds=5400, cooldown_seconds=72000),
            clock=lambda: now[0],
            local_datetime=lambda: datetime(2026, 10, 17, 12),
        )
        await repo.note_user_activity(7, 100)
        sent = await messenger.run_once()
        assert sent == (1 if change == "none" else 0)
        model.chat.assert_awaited_once()
        row = await persistence.load(7)
        assert row is not None
        assert row.emotions.memory.joy < 0.8
        assert row.emotions.memory.tiredness < 0.4
        if change == "none":
            bot.send_message.assert_awaited_once_with(7, "Есть настроение поболтать :3")
            assert len(row.history) == 1 and row.history[0].proactive
            assert await messenger.run_once() == 0
            model.chat.assert_awaited_once()
        else:
            bot.send_message.assert_not_awaited()
            assert not row.history

    asyncio.run(run())


@pytest.mark.parametrize(
    "group,bot_user", [(False, False), (True, False), (False, True)]
)
def test_incoming_activity_and_confirmed_outgoing_delivery(
    tmp_path: Path, group: bool, bot_user: bool
) -> None:
    async def run() -> None:
        repo = MemoriesRepository(tmp_path)
        event = Message(
            message_id=1,
            date=datetime.now(timezone.utc),
            from_user=User(id=7, is_bot=bot_user, first_name="Тест"),
            chat=Chat(id=-10 if group else 7, type="group" if group else "private"),
            text="/status",
        )
        incoming = ProactiveIncomingMiddleware(
            repo, UserStateStore(), InputOperations(), clock=lambda: 100
        )
        handler = AsyncMock()
        await incoming(handler, event, {})
        handler.assert_awaited_once()
        outgoing = ProactiveOutgoingMiddleware(repo, clock=lambda: 200)
        method = SendMessage(chat_id=7, text="Ответ")
        await outgoing(AsyncMock(return_value=event), cast(Bot, object()), method)
        if group or bot_user:
            assert not await due(repo, 100000)
        else:
            assert not await due(repo, 5500)
            scheduled = deadline(tmp_path)[0]
            assert scheduled is not None and 5600 <= scheduled <= 11000
            token = proactive_delivery.set(True)
            try:
                await outgoing(
                    AsyncMock(return_value=event), cast(Bot, object()), method
                )
            finally:
                proactive_delivery.reset(token)
            assert deadline(tmp_path)[0] == scheduled
        with pytest.raises(RuntimeError):
            await outgoing(
                AsyncMock(side_effect=RuntimeError("failure")),
                cast(Bot, object()),
                method,
            )

    asyncio.run(run())


def test_moscow_quiet_clock_has_explicit_timezone() -> None:
    assert _moscow_datetime().utcoffset() is not None
    assert _moscow_datetime().utcoffset().total_seconds() == 10800  # type: ignore[union-attr]
