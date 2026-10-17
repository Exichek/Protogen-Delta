"""Побуждение, разные поводы, память намерения и право на молчание без внешних API."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import SendMessage

from protogen_delta.config.settings import load_settings
from protogen_delta.core.llm_budget import LlmDailyLimitError
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.repositories.initiative import InitiativeRepository
from protogen_delta.repositories.memories import MemoriesRepository, Memory
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.services.deepseek import DeepSeekAPIError, DeepSeekService
from protogen_delta.services.initiative_context import initiative_reply_context
from protogen_delta.services.initiative_planner import (
    SILENCE,
    choose_initiative,
    initiative_drive,
    repeats_initiative,
)
from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger
from protogen_delta.services.stickers import ContextualStickerService


def saved(root: Path, user: int = 7) -> tuple[Any, ...]:
    with closing(sqlite3.connect(root / "memories.db")) as db:
        return cast(
            tuple[Any, ...],
            db.execute(
                "SELECT urge,updated_at,last_kind,last_sent_at,sent_today,deferred_until "
                "FROM initiative_state WHERE user_id=?",
                (user,),
            ).fetchone(),
        )


def test_accumulation_depends_on_elapsed_time_and_survives_restart(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeRepository(tmp_path)
        for user in (7, 8):
            await memories.note_user_activity(user, 100)
            assert not (await repo.assess(user, 100, 1)).ready
        for at in range(700, 7301, 600):
            await repo.assess(7, at, 1)
        repo = InitiativeRepository(tmp_path)
        assert (await repo.assess(8, 7300, 1)).ready
        assert saved(tmp_path, 7)[0] == pytest.approx(saved(tmp_path, 8)[0])
        previous = saved(tmp_path, 7)
        await repo.assess(7, 7000, 1)
        assert saved(tmp_path, 7) == previous
        assert initiative_drive(1, 1, 1, 0) == 1.5
        assert initiative_drive(0, 0, 0, 1) < initiative_drive(0, 0, 0, 0)
        assert initiative_drive(0, 0, 0, 100) == 0.5
        with pytest.raises(ValueError):
            await repo.assess(7, float("nan"), 1)
        with pytest.raises(ValueError):
            await repo.assess(7, 7300, 2)

    asyncio.run(run())


def test_cooldown_daily_cap_legacy_delivery_and_reset(tmp_path: Path) -> None:
    async def run() -> None:
        now = datetime(2026, 10, 17, 7, tzinfo=timezone.utc).timestamp()
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeRepository(tmp_path)
        await memories.note_user_activity(7, now - 10800)
        await memories.note_proactive_sent(7, now - 3600)
        await memories.note_user_activity(7, now - 100)
        assert not (await repo.assess(7, now, 1)).ready
        now += 14400
        assert (await repo.assess(7, now, 1)).ready
        await repo.note_sent(7, now, "play")
        await repo.note_sent(7, now, "play")
        assert saved(tmp_path)[4] == 2  # Включая прежнюю отправку сегодня.
        await memories.note_user_activity(7, now + 10)
        assert not (await repo.assess(7, now + 14399, 1)).ready
        assert not (await repo.assess(7, now + 14400, 1)).ready
        assert (await repo.assess(7, now + 86400, 1)).ready
        await repo.note_sent(7, now + 86400, "thought")
        assert saved(tmp_path)[4] == 1
        await memories.delete_user(7)
        assert saved(tmp_path) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "restriction", ["missing", "disabled", "suspended", "unanswered"]
)
def test_ineligible_accounts_do_not_accumulate(
    tmp_path: Path, restriction: str
) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeRepository(tmp_path)
        if restriction != "missing":
            await memories.note_user_activity(7, 100)
            if restriction == "disabled":
                await memories.set_proactive(7, False, 100)
            elif restriction == "suspended":
                await memories.suspend_proactive_delivery(7)
            else:
                await memories.note_proactive_sent(7, 200)
        assert not (await repo.assess(7, 20000, 1)).ready
        assert saved(tmp_path) is None

    asyncio.run(run())


def test_defer_honors_activity_version_and_restart(tmp_path: Path) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeRepository(tmp_path)
        await memories.note_user_activity(7, 100)
        assert (await repo.assess(7, 10000, 1)).ready
        await memories.note_user_activity(7, 10001)
        await repo.defer(7, 20000, 1)
        assert saved(tmp_path)[5] == 0
        await repo.defer(7, 20000, 2)
        repo = InitiativeRepository(tmp_path)
        assert not (await repo.assess(7, 19999, 1)).ready
        assert (await repo.assess(7, 20000, 1)).ready

    asyncio.run(run())


def test_planner_has_variety_grounded_callbacks_and_no_immediate_kind_repeat() -> None:
    def choose(value: float, **kwargs: Any) -> Any:
        return choose_initiative(
            joy=0.4,
            warmth=0.5,
            playfulness=0.4,
            last_kind=kwargs.pop("last_kind", ""),
            history=kwargs.pop("history", ()),
            value=value,
            **kwargs,
        )

    empty = {choose(i / 100, memories=[]).kind for i in range(100)}
    assert empty == {"thought", "play", "mood"}
    memory = Memory("funny", "Смешная история про кота и ноутбук", 100)
    plan = choose(0.999, memories=[memory])
    assert plan.kind == "callback" and memory.text in plan.request()
    assert SILENCE in plan.request()
    assert choose(0.999, memories=[memory], last_kind="callback").kind != "callback"
    closed = (ConversationTurn("Закроем код", "Хорошо", technical_topic_closed=True),)
    assert (
        choose(
            0.999,
            memories=[Memory("funny", "шутка про Python и код", 100)],
            history=closed,
        ).kind
        != "callback"
    )
    with pytest.raises(ValueError):
        choose(1, memories=[])
    history = (
        ConversationTurn(
            "", "Хочешь придумать имя для очень сонного дракона?", proactive=True
        ),
    )
    assert repeats_initiative(
        "Хочешь придумать имя для очень сонного дракона?!", history
    )
    assert not repeats_initiative("Новая мысль про облака", history)


def build(
    root: Path,
    now: list[float],
    response: str = "Есть идея для маленькой игры :3",
    *,
    stickers: bool = False,
    value: float = 0,
) -> tuple[Any, ...]:
    memories = MemoriesRepository(root, random_delay=lambda lo, hi: lo)
    initiative = InitiativeRepository(root)
    persistence = UserStateRepository(root)
    states = UserStateStore(persistence=persistence, wall_clock=lambda: now[0])
    bot = AsyncMock(spec=Bot)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = response
    sticker_service = Mock(spec=ContextualStickerService)
    sticker_service.proactive_available.return_value = True
    sticker_service.send_proactive = AsyncMock(return_value="[Стикер: улыбка]")
    service = ProactiveMessenger(
        bot=cast(Bot, bot),
        deepseek=cast(DeepSeekService, model),
        repository=memories,
        initiative=initiative,
        user_states=states,
        system_prompt="core",
        clock=lambda: now[0],
        random_value=lambda: value,
        local_datetime=lambda: datetime(2026, 10, 17, 12),
        stickers=sticker_service if stickers else None,
        config=ProactiveConfig(
            idle_seconds=5400,
            idle_max_seconds=10800,
            sticker_idle_seconds=1800 if stickers else None,
        ),
    )
    return memories, initiative, persistence, states, bot, model, service


def test_live_path_saves_reason_and_waits_for_answer(tmp_path: Path) -> None:
    async def run() -> None:
        now = [20000.0]
        memories, _, persistence, states, bot, model, service = build(tmp_path, now)
        await memories.note_user_activity(7, 100)
        states.get(7).emotions.memory.joy = 0.5
        assert await service.run_once() == 1
        assert saved(tmp_path)[0] == 0 and saved(tmp_path)[4] == 1
        fresh = UserStateStore(persistence=persistence, wall_clock=lambda: now[0])
        async with fresh.use_conversation(7) as state:
            assert (
                state.history[-1].initiative_reason
                == "Поделиться новой мыслью для разговора"
            )
            context = initiative_reply_context("Чего хотел?", state)
            assert state.history[-1].initiative_reason in context
        assert (
            "новую мысль" in model.chat.await_args.kwargs["user_message"]
            or "свежую лёгкую мысль" in model.chat.await_args.kwargs["user_message"]
        )
        now[0] += 72001
        assert await service.run_once() == 0
        bot.send_message.assert_awaited_once()
        model.chat.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize(
    "reply", ["", SILENCE, "Почему ты молчишь?", "Хочешь придумать имя для дракона?"]
)
def test_silence_pressure_and_repetition_back_off_without_delivery(
    tmp_path: Path, reply: str
) -> None:
    async def run() -> None:
        now = [20000.0]
        memories, _, _, states, bot, model, service = build(tmp_path, now, reply)
        async with states.use_conversation(7) as state:
            state.history.append(
                ConversationTurn(
                    "", "Хочешь придумать имя для дракона?", proactive=True
                )
            )
        await memories.note_user_activity(7, 100)
        assert await service.run_once() == 0
        assert saved(tmp_path)[5] == 23600
        now[0] += 30
        assert await service.run_once() == 0
        model.chat.assert_awaited_once()
        bot.send_message.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "error,delay", [(LlmDailyLimitError(), 3600), (DeepSeekAPIError("test"), 900)]
)
def test_api_errors_defer_without_repeated_calls(
    tmp_path: Path, error: Exception, delay: int
) -> None:
    async def run() -> None:
        now = [20000.0]
        memories, _, _, _, bot, model, service = build(tmp_path, now)
        model.chat.side_effect = error
        await memories.note_user_activity(7, 100)
        assert await service.run_once() == 0
        assert saved(tmp_path)[5] == now[0] + delay
        assert await service.run_once() == 0
        model.chat.assert_awaited_once()
        bot.send_message.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("change", ["message", "mode", "roleplay"])
def test_new_context_cancels_and_roleplay_keeps_its_invitation(
    tmp_path: Path, change: str
) -> None:
    async def run() -> None:
        now = [20000.0]
        memories, _, persistence, states, bot, model, service = build(tmp_path, now)
        if change == "roleplay":
            async with states.use_conversation(7) as state:
                state.roleplay_active = True

        async def generate(**_: Any) -> str:
            now[0] += 10
            if change == "message":
                await memories.note_user_activity(7, now[0])
            elif change == "mode":
                async with states.use_conversation(7) as state:
                    state.content_mode = "adult"
            return "Продолжим сцену?"

        model.chat.side_effect = generate
        await memories.note_user_activity(7, 100)
        assert await service.run_once() == int(change == "roleplay")
        row = await persistence.load(7)
        assert row is not None
        if change == "roleplay":
            assert "RP-сцене" in row.history[-1].initiative_reason
            assert SILENCE in model.chat.await_args.kwargs["user_message"]
            bot.send_message.assert_awaited_once()
        else:
            assert not row.history
            assert saved(tmp_path)[4] == 0
            bot.send_message.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("blocked", ["unanswered", "deferred", "daily_cap"])
def test_blocked_accounts_do_not_starve_other_candidates(
    tmp_path: Path, blocked: str
) -> None:
    async def run() -> None:
        now = [100000.0]
        memories, initiative, _, _, bot, model, service = build(tmp_path, now)
        for user in range(20, 35):
            await memories.note_user_activity(user, 100)
            await initiative.assess(user, 20000, 1)
            if blocked == "unanswered":
                await memories.note_proactive_sent(user, 200)
            elif blocked == "deferred":
                await initiative.defer(user, 110000, 1)
            else:
                await initiative.note_sent(user, 80000, "play")
                await initiative.note_sent(user, 80001, "mood")
        await memories.note_user_activity(7, 90000)
        assert await service.run_once() == 1
        bot.send_message.assert_awaited_once()
        assert bot.send_message.await_args.args[0] == 7
        model.chat.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize(
    "error",
    [
        TelegramBadRequest(SendMessage(chat_id=7, text="test"), "test"),
        RuntimeError("test"),
    ],
)
def test_delivery_failure_does_not_record_success_and_backs_off(
    tmp_path: Path, error: Exception
) -> None:
    async def run() -> None:
        now = [20000.0]
        memories, _, persistence, _, bot, model, service = build(tmp_path, now)
        bot.send_message.side_effect = error
        await memories.note_user_activity(7, 100)
        assert await service.run_once() == 0
        assert saved(tmp_path)[4] == 0 and saved(tmp_path)[5] == 20900
        row = await persistence.load(7)
        assert row is None or not row.history
        assert await service.run_once() == 0
        model.chat.assert_awaited_once()

    asyncio.run(run())


def test_sticker_initiative_uses_no_model_and_remembers_its_reason(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [20000.0]
        memories, _, persistence, _, bot, model, service = build(
            tmp_path, now, stickers=True
        )
        await memories.note_user_activity(7, 100)
        assert await service.run_once() == 1
        row = await persistence.load(7)
        assert (
            row is not None
            and row.history[-1].initiative_reason
            == "Выразить текущее настроение стикером"
        )
        model.chat.assert_not_awaited()
        bot.send_message.assert_not_awaited()
        assert saved(tmp_path)[4] == 1

    asyncio.run(run())


def test_not_enough_urge_waits_without_calling_model(tmp_path: Path) -> None:
    async def run() -> None:
        now = [5500.0]
        memories, _, _, _, bot, model, service = build(tmp_path, now)
        await memories.note_user_activity(7, 100)
        assert await service.run_once() == 0
        model.chat.assert_not_awaited()
        now[0] = 7300
        assert await service.run_once() == 1
        bot.send_message.assert_awaited_once()

    asyncio.run(run())


def test_text_and_sticker_are_one_initiative(tmp_path: Path) -> None:
    async def run() -> None:
        now = [20000.0]
        memories, _, persistence, _, bot, model, service = build(
            tmp_path, now, stickers=True, value=0.4
        )
        await memories.note_user_activity(7, 100)
        assert await service.run_once() == 1
        row = await persistence.load(7)
        assert row is not None and len(row.history) == 2
        assert all(turn.initiative_reason for turn in row.history)
        assert saved(tmp_path)[4] == 1
        bot.send_message.assert_awaited_once()
        model.chat.assert_awaited_once()

    asyncio.run(run())


def test_lively_setting_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("PROACTIVE_LIVELY_ENABLED", "false")
    assert not load_settings().proactive_lively_enabled
    monkeypatch.setenv("PROACTIVE_LIVELY_ENABLED", "true")
    assert load_settings().proactive_lively_enabled


@pytest.mark.parametrize(
    "reason,proactive,expected",
    [("Сыграть", True, "Сыграть"), (123, True, ""), ("Подмена", False, "")],
)
def test_history_reason_requires_proactive_record(
    reason: Any, proactive: bool, expected: str
) -> None:
    import json

    history = UserStateRepository._decode_history(
        json.dumps(
            [
                {
                    "user": "",
                    "assistant": "Привет",
                    "proactive": proactive,
                    "initiative_reason": reason,
                }
            ]
        )
    )
    assert history[0].initiative_reason == expected
