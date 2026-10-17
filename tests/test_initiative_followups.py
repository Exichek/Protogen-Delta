"""Повод по реальному плану ограничен сроком, источником и одной попыткой."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_lively_initiative import build
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.core.initiative_followup import extract_followup_plan
from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.repositories.initiative_followups import (
    DueFollowup,
    InitiativeFollowupsRepository,
)
from protogen_delta.repositories.memories import MemoriesRepository, Memory
from protogen_delta.repositories.user_facts import FactsUpdate
from protogen_delta.services.deepseek import DeepSeekAPIError
from protogen_delta.services.initiative_planner import SILENCE, choose_initiative
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.user_facts import UserFactsService

NOW = datetime(2026, 10, 17, 9, tzinfo=timezone.utc).timestamp()
TEXT = "Завтра у меня экзамен"
DUE = datetime(2026, 10, 19, 9, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize(
    "text,day",
    [
        (TEXT, "2026-10-18"),
        ("Сегодня пойду на собеседование", "2026-10-17"),
        ("Послезавтра участвую в турнире", "2026-10-19"),
        ("2026-10-25 у меня презентация", "2026-10-25"),
        ("Я завтра на экзамене", "2026-10-18"),
    ],
)
def test_explicit_own_plan_gets_day_after_event(text: str, day: str) -> None:
    plan = extract_followup_plan(text, NOW)
    assert plan and plan.source == text and plan.event_day == day
    assert plan.due_at > NOW and plan.expires_at == plan.due_at + 3 * 86400


@pytest.mark.parametrize(
    "text",
    [
        "У меня экзамен",
        "Может, завтра у меня экзамен",
        "Если завтра пойду на экзамен",
        "Завтра у меня экзамен?",
        "Завтра у друга экзамен",
        "Я думаю, завтра у друга экзамен",
        "Он сказал: «Завтра у меня экзамен»",
        "`Завтра у меня экзамен`",
        "```\nЗавтра у меня экзамен\n```",
        "/remind завтра у меня экзамен",
        "*завтра пойду на экзамен*",
        "Завтра у меня экзамен у врача",
        "Завтра у меня экзамен по сексу",
        "Завтра у меня экзамен, не спрашивай",
        "Завтра у меня экзамен, пароль secret",
        "2026-11-20 у меня экзамен",
        "2026-10-10 у меня экзамен",
        "2026-02-30 у меня экзамен",
        "Завтра и послезавтра у меня экзамен",
        "Завтра у меня экзамен и презентация",
    ],
)
def test_ambiguous_quoted_sensitive_or_undated_text_is_not_a_plan(text: str) -> None:
    assert extract_followup_plan(text, NOW) is None


def test_persistent_due_once_owner_scope_expiry_and_reset(tmp_path: Path) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT, TEXT), NOW)
        assert await repo.due(7, DUE - 1) is None
        repo = InitiativeFollowupsRepository(tmp_path)
        row = await repo.due(7, DUE)
        assert row and row.source == TEXT
        assert await repo.due(8, DUE) is None
        assert not await repo.select(8, row.id, DUE)
        results = await asyncio.gather(
            repo.select(7, row.id, DUE), repo.select(7, row.id, DUE)
        )
        assert sorted(results) == [False, True]
        await repo.observe(7, (TEXT,), NOW)
        assert await repo.due(7, DUE) is None
        await memories.delete_user(7)
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert (
                db.execute("SELECT count(*) FROM initiative_followups").fetchone()[0]
                == 0
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "correction", ["Экзамен отменили", "Экзамен сдал", "Не спрашивай об этом"]
)
def test_correction_or_completion_closes_pending_plan(
    tmp_path: Path, correction: str
) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT,), NOW)
        await repo.observe(7, (correction,), NOW + 100)
        assert await repo.due(7, DUE) is None

    asyncio.run(run())


def test_rescheduling_forgetting_and_expiry(tmp_path: Path) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT,), NOW)
        await repo.observe(7, ("Послезавтра у меня экзамен",), NOW)
        assert await repo.due(7, DUE) is None
        assert await repo.due(7, DUE + 86400)
        await repo.forget_sources(7, ("экзамен",))
        assert await repo.due(7, DUE + 86400) is None
        await repo.observe(7, (TEXT,), NOW)
        assert await repo.due(7, DUE + 3 * 86400) is None
        await repo.observe(7, (TEXT,), NOW)
        await repo.forget_sources(7, ())
        assert await repo.due(7, DUE) is None

    asyncio.run(run())


@pytest.mark.parametrize("response", ["Как всё прошло с твоим экзаменом?", SILENCE])
def test_followup_uses_existing_delivery_limits_and_saves_reason(
    tmp_path: Path, response: str
) -> None:
    async def run() -> None:
        now = [DUE]
        memories, _, _, states, bot, model, service = build(
            tmp_path, now, response, value=0.999
        )
        await memories.note_user_activity(7, DUE - 10000)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT,), NOW)
        service._followups = repo
        assert await service.run_once() == (0 if response == SILENCE else 1)
        assert TEXT in model.chat.await_args.kwargs["user_message"]
        assert "не доказывает" in model.chat.await_args.kwargs["user_message"]
        assert await repo.due(7, DUE) is None
        if response != SILENCE:
            assert "планом" in states.get(7).history[-1].initiative_reason
            assert await service.run_once() == 0
            bot.send_message.assert_awaited_once()
        else:
            bot.send_message.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("scope", ["own", "forwarded", "group", "attachment", "rp"])
def test_engine_records_only_own_non_rp_plaintext_plans(
    tmp_path: Path, scope: str
) -> None:
    async def run() -> None:
        engine, _, _, *_ = _create_engine()
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeFollowupsRepository(tmp_path)
        engine._memory = MemoryService(memories, clock=lambda: NOW, followups=repo)
        if scope == "rp":
            engine._user_states.get(TEST_USER_ID).roleplay_active = True
        token = current_reply_sources.set(
            (ReplySource(TEXT, TEXT, scope != "forwarded", scope != "forwarded"),)
        )
        try:
            if scope == "group":
                await engine.respond(TEST_USER_ID, TEXT, chat_id=-100)
            else:
                await engine.respond(
                    TEST_USER_ID,
                    TEXT,
                    attachment_text=TEXT if scope == "attachment" else None,
                )
        finally:
            current_reply_sources.reset(token)
        assert bool(await repo.due(TEST_USER_ID, DUE)) is (scope == "own")

    asyncio.run(run())


def test_cancelled_snapshot_cannot_be_selected_and_pending_cap_is_bounded(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        texts = tuple(
            "Завтра у меня " + topic
            for topic in (
                "экзамен",
                "собеседование",
                "презентация",
                "защита проекта",
                "релиз",
                "турнир",
            )
        )
        await repo.observe(7, texts, NOW)
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM initiative_followups WHERE closed_at=0"
                ).fetchone()[0]
                == 5
            )
        row = await repo.due(7, DUE)
        assert row
        await repo.observe(7, ("Забудь все планы",), DUE)
        assert not await repo.select(7, row.id, DUE)
        assert await repo.due(7, DUE) is None

    asyncio.run(run())


@pytest.mark.parametrize("blocked", ["quiet", "disabled", "rp"])
def test_existing_background_guards_do_not_consume_followup(
    tmp_path: Path, blocked: str
) -> None:
    async def run() -> None:
        now = [DUE]
        memories, _, _, states, _, model, service = build(tmp_path, now, value=0.999)
        await memories.note_user_activity(7, DUE - 10000)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT,), NOW)
        service._followups = repo
        if blocked == "quiet":
            service._local_datetime = lambda: datetime(2026, 10, 19, 23)
        elif blocked == "disabled":
            await memories.set_proactive(7, False, now[0])
        else:
            states.get(7).roleplay_active = True
        await service.run_once()
        assert await repo.due(7, DUE) is not None
        if blocked != "rp":
            model.chat.assert_not_awaited()
        else:
            assert TEXT not in model.chat.await_args.kwargs["user_message"]

    asyncio.run(run())


def test_future_plan_is_not_a_past_funny_callback() -> None:
    plan = choose_initiative(
        joy=0.5,
        warmth=0.5,
        playfulness=0.5,
        last_kind="",
        memories=[Memory("funny", TEXT, NOW)],
        history=(),
        value=0.999,
    )
    assert plan.kind != "callback" and TEXT not in plan.request()


def test_sensitive_plan_cannot_bypass_existing_topic_filter(tmp_path: Path) -> None:
    async def run() -> None:
        now = [DUE]
        memories, _, _, _, _, model, service = build(tmp_path, now, value=0.999)
        await memories.note_user_activity(7, DUE - 10000)
        repo = InitiativeFollowupsRepository(tmp_path)
        text = TEXT + ", тревожно"
        await repo.observe(7, (text,), NOW)
        service._followups = repo
        await service.run_once()
        assert text not in model.chat.await_args.kwargs["user_message"]
        assert await repo.due(7, DUE) is not None

    asyncio.run(run())


def test_followup_opt_out_survives_restart_and_needs_explicit_opt_in(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT, "Не спрашивай о моих планах"), NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, ("Завтра у меня собеседование",), NOW)
        assert await repo.due(7, DUE) is None
        await repo.observe(
            7,
            ("Можешь снова спрашивать о моих планах", "Завтра у меня презентация"),
            NOW,
        )
        row = await repo.due(7, DUE)
        assert row and "презентация" in row.source
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            before = db.execute(
                "SELECT proactive_enabled FROM engagement WHERE user_id=7"
            ).fetchone()[0]
        assert before == 1

    asyncio.run(run())


def test_forgetting_facts_also_removes_pending_plan_sources(tmp_path: Path) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT,), NOW)
        facts = AsyncMock(spec=UserFactsService)
        facts.observe.return_value = FactsUpdate(forgotten=True)
        memory = MemoryService(memories, facts=facts, followups=repo)
        await memory.observe_facts(7, "Забудь всё")
        assert await repo.due(7, DUE) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "message",
    [
        "Экзамен отменили?",
        "Если экзамен отменили, то хорошо",
        "Экзамен не отменили",
        "Он сказал: «Не спрашивай о моих планах»",
        "Если я скажу не спрашивай о моих планах, что будет?",
    ],
)
def test_question_hypothesis_or_quote_does_not_cancel_plan(
    tmp_path: Path, message: str
) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, NOW)
        repo = InitiativeFollowupsRepository(tmp_path)
        await repo.observe(7, (TEXT, message), NOW)
        assert await repo.due(7, DUE) is not None

    asyncio.run(run())


@pytest.mark.parametrize("closed", [True, False])
def test_followup_respects_closed_technical_topic(closed: bool) -> None:
    followup = DueFollowup(1, "Завтра у меня релиз Python-пакета", "2026-10-18")
    history = (
        (ConversationTurn("С кодом пока всё", "Хорошо", technical_topic_closed=True),)
        if closed
        else ()
    )
    plan = choose_initiative(
        joy=0.5,
        warmth=0.5,
        playfulness=0.5,
        last_kind="",
        memories=(),
        history=history,
        value=0.999,
        followup=followup,
    )
    assert (plan.kind == "followup") is not closed


def test_closed_plan_stays_closed_after_short_history_disappears(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        engine, _, _, *_ = _create_engine()
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeFollowupsRepository(tmp_path)
        memory = MemoryService(memories, clock=lambda: NOW, followups=repo)
        engine._memory = memory
        await memory.observe_followups(
            TEST_USER_ID, ("Завтра у меня релиз Python-пакета", TEXT)
        )
        await engine.respond(TEST_USER_ID, "С кодом пока всё")
        engine._user_states.get(TEST_USER_ID).history.clear()
        row = await repo.due(TEST_USER_ID, DUE)
        assert (
            row and row.source == TEXT
        )  # Нейтральный план не закрывается вместе с кодом.
        assert await repo.select(TEST_USER_ID, row.id, DUE)
        assert await repo.due(TEST_USER_ID, DUE) is None

    asyncio.run(run())


def test_opt_out_is_saved_even_if_reply_model_fails(tmp_path: Path) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeFollowupsRepository(tmp_path)
        memory = MemoryService(memories, clock=lambda: NOW, followups=repo)
        engine._memory = memory
        await memory.observe_followups(TEST_USER_ID, (TEXT,))
        model.chat.side_effect = DeepSeekAPIError("synthetic")
        await engine.respond(TEST_USER_ID, "Не спрашивай о моих планах")
        assert await repo.due(TEST_USER_ID, DUE) is None
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert (
                db.execute(
                    "SELECT enabled FROM initiative_followup_preferences WHERE user_id=?",
                    (TEST_USER_ID,),
                ).fetchone()[0]
                == 0
            )

    asyncio.run(run())
