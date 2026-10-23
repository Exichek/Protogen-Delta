"""Одна попытка новости за сутки, сохранение после перезапуска и удаление при сбросе."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from test_initiative_news import NOW, URL

from protogen_delta.repositories.initiative import InitiativeRepository
from protogen_delta.repositories.memories import MemoriesRepository


def test_news_claim_is_atomic_survives_restart_and_reopens_next_day(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        at = NOW.timestamp()
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeRepository(tmp_path)
        await memories.note_user_activity(7, at - 10800)
        assert (await repo.assess(7, at, 1)).news_available
        claims = await asyncio.gather(
            repo.claim_news(7, at, 1), repo.claim_news(7, at, 1)
        )
        assert sorted(claims) == [False, True]
        repo = InitiativeRepository(tmp_path)
        assert not (await repo.assess(7, at + 10, 1)).news_available
        assert (await repo.assess(7, at + 86400, 1)).news_available
        assert await repo.claim_news(7, at + 86400, 1)

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["stale", "disabled", "suspended", "unanswered"])
def test_news_claim_does_not_bypass_activity_or_permissions(
    tmp_path: Path, reason: str
) -> None:
    async def run() -> None:
        at = NOW.timestamp()
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeRepository(tmp_path)
        await memories.note_user_activity(7, at - 10800)
        await repo.assess(7, at, 1)
        if reason == "stale":
            await memories.note_user_activity(7, at)
        elif reason == "disabled":
            await memories.set_proactive(7, False, at)
        elif reason == "suspended":
            await memories.suspend_proactive_delivery(7)
        else:
            await memories.note_proactive_sent(7, at)
        assert not await repo.claim_news(7, at, 1)
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute(
                "SELECT news_attempted_day FROM initiative_state"
            ).fetchone() == ("",)

    asyncio.run(run())


def test_delivered_urls_expire_and_reset_removes_news_metadata(tmp_path: Path) -> None:
    async def run() -> None:
        at = NOW.timestamp()
        memories = MemoriesRepository(tmp_path)
        repo = InitiativeRepository(tmp_path)
        await memories.note_user_activity(7, at - 10800)
        await repo.assess(7, at, 1)
        await repo.note_sent(7, at, "news", URL)
        await repo.note_sent(7, at, "news", URL)
        repo = InitiativeRepository(tmp_path)
        assert await repo.seen_news(7, at) == (URL,)
        assert await repo.seen_news(8, at) == ()
        assert await repo.seen_news(7, at + 8 * 86400) == ()
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute("SELECT sent_today FROM initiative_state").fetchone() == (
                1,
            )
        await repo.note_sent(7, at + 8 * 86400, "news", URL)
        assert await repo.seen_news(7, at + 8 * 86400) == (URL,)
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute("SELECT COUNT(*) FROM initiative_news").fetchone() == (1,)
        await memories.delete_user(7)
        assert await repo.seen_news(7, at) == ()
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute("SELECT COUNT(*) FROM initiative_state").fetchone() == (
                0,
            )

    asyncio.run(run())


def test_existing_initiative_state_is_migrated_without_losing_values(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        memories = MemoriesRepository(tmp_path)
        await memories.note_user_activity(7, 100)
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db, db:
            db.execute(
                "CREATE TABLE initiative_state (user_id INTEGER PRIMARY KEY "
                "REFERENCES engagement(user_id) ON DELETE CASCADE,"
                "urge REAL NOT NULL DEFAULT 0,updated_at REAL NOT NULL,"
                "last_kind TEXT NOT NULL DEFAULT '',last_sent_at REAL NOT NULL DEFAULT 0,"
                "sent_day TEXT NOT NULL DEFAULT '',sent_today INTEGER NOT NULL DEFAULT 0,"
                "deferred_until REAL NOT NULL DEFAULT 0)"
            )
            db.execute(
                "INSERT INTO initiative_state VALUES (7,0.8,100,'thought',0,'',0,200)"
            )
        InitiativeRepository(tmp_path)
        InitiativeRepository(tmp_path)
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
            assert db.execute(
                "SELECT urge,last_kind,deferred_until,news_attempted_day FROM initiative_state"
            ).fetchone() == (0.8, "thought", 200, "")

    asyncio.run(run())
