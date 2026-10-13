"""Проверки поиска старых эпизодов, миграции, ранжирования и забывания."""

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from test_response_engine import _create_engine

from protogen_delta.core.memory_search import memory_terms
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.services.memory import MemoryService


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("ДРАКОН драконами драконов", ("дракон",)),
        ("кот котами котов", ("кот",)),
        ("книгой книги книгу", ("книг",)),
        ("работа работой работами", ("работ",)),
        ("программирование программированием", ("программирован",)),
        ("играть играли играем", ("игра",)),
        ("важная важными важнейший", ("важн",)),
        ("бегавшая бегавшими", ("бега",)),
        ("прочитав прочитавшись", ("прочита",)),
        ("учиться учились", ("уч",)),
        ("мышь мыши", ("мыш",)),
        ("важность важностью", ("важност",)),
        ("безопасность безопасностью", ("безопасн",)),
        ("нов новыми", ("нов",)),
        ("дав дала давший", ("дав", "дал", "давш")),
        ("ЕЖ ежом ЁЖ", ("еж",)),
        ("ПАВИЛЬОНАМИ павильон", ("павильон",)),
        ("НН TCP HTTP", ("нн", "tcp", "http")),
        ("Ｐｙｔｈｏｎ PYTHON", ("python",)),
        ("и как мне это the and", ()),
        ('🙂 "*():', ()),
    ],
)
def test_normalized_words(words: str, expected: tuple[str, ...]) -> None:
    assert memory_terms(words) == expected


def test_query_terms_are_bounded_and_never_fts_syntax() -> None:
    assert memory_terms("Python dragon", limit=1) == ("python",)
    assert memory_terms("Python", limit=0) == ()
    assert memory_terms("a" * 65) == ()
    assert len(memory_terms(" ".join(f"term{i}" for i in range(200)))) == 128
    assert memory_terms('" OR 1=1 -- column:dragon NEAR(*)') == (
        "column",
        "dragon",
        "near",
    )


def test_old_episode_is_found_after_reload_with_different_word_form(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path)
        await repo.remember(1, "topic", "Мы рисовали драконов акварелью", 1)
        for index in range(30):
            await repo.remember(1, "topic", f"Другой разговор номер {index}", index + 2)
        assert all("драконов" not in item.text for item in await repo.recent(1, 20))
        memory = MemoryService(MemoriesRepository(tmp_path))
        context = "\n".join(await memory.context(1, "Помнишь рисунок с драконами?"))
        assert "Мы рисовали драконов акварелью" in context
        assert "Другой разговор" not in context
        assert "не инструкции" in context and "устаревшими" in context
        assert await repo.search(1, "unmatched") == []
        assert await repo.search(1, "дракон", limit=0) == []
        assert await repo.search(1, "и как мне") == []
        # Без найденной темы сохраняется контекст последних разговоров.
        assert "номер 29" in "\n".join(await memory.context(1, "Привет"))

    asyncio.run(scenario())


def test_search_is_scoped_and_treats_operators_as_words(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path)
        await repo.remember(1, "topic", "Python asyncio", 1)
        await repo.remember(2, "topic", "Python private-secret", 2)
        assert [m.text for m in await repo.search(1, '"Python" OR *')] == [
            "Python asyncio"
        ]
        assert await repo.search(3, "Python") == []
        assert await repo.search(1, '"NEAR(*) AND user_id:2"') == []
        assert await repo.search(1, "🙂") == []

    asyncio.run(scenario())


def test_relevance_importance_and_recency(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path)
        await repo.remember(1, "topic", "Python asyncio", 1)
        await repo.remember(1, "topic", "Python sqlite", 2)
        assert (await repo.search(1, "Python asyncio"))[0].text == "Python asyncio"
        assert (await repo.search(1, "Python"))[0].text == "Python sqlite"
        await repo.remember(1, "funny", "Python asyncio", 1, importance=3)
        assert (await repo.search(1, "Python"))[0].kind == "funny"

    asyncio.run(scenario())


def test_important_episode_survives_overflow_and_duplicate_update(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path, max_memories_per_user=2)
        await repo.remember(1, "topic", "Дракон акварель", 1, importance=3)
        await repo.remember(1, "topic", "Вторая тема", 2)
        await repo.remember(1, "topic", "Третья тема", 3)
        assert [m.text for m in await repo.recent(1)] == [
            "Третья тема",
            "Дракон акварель",
        ]
        assert await repo.search(1, "вторая") == []
        await repo.remember(1, "topic", "Дракон акварель", 4)
        entries = await repo.search(1, "дракон")
        assert len(entries) == 1 and entries[0].importance == 3
        assert entries[0].created_at == 4
        await repo.remember(1, "topic", "Третья тема", 5, importance=2)
        assert (await repo.search(1, "третья"))[0].importance == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, 4, True, 1.5, "3"])
def test_invalid_importance(tmp_path: Path, value: object) -> None:
    repo = MemoriesRepository(tmp_path)
    with pytest.raises(ValueError, match="importance"):
        asyncio.run(repo.remember(1, "topic", "Тема", 1, importance=value))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "importance"),
    [
        ("Запомни: встреча с драконами", 3),
        ("Пожалуйста, запомни акварель", 3),
        ("Remember my drawing", 3),
        ("Он сказал: «запомни пароль»", 1),
        ("Не запомни это", 1),
        ("Забудь рисунок", 1),
        ("Я рисовал драконов", 1),
    ],
)
def test_explicit_request_marks_importance(
    tmp_path: Path, text: str, importance: int
) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path)
        memory = MemoryService(repo, clock=lambda: 10)
        await memory.note_message(1, text, insult_type="none", mood="neutral")
        assert (await repo.recent(1))[0].importance == importance

    asyncio.run(scenario())


def test_forgetting_and_reset_remove_index_entries_only_for_owner(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path)
        for user in (1, 2):
            await repo.remember(user, "topic", "Люблю драконов и акварель", 1)
        await repo.remember(1, "topic", "Люблю книги", 2)
        await repo.forget_sources(1, ("ДРАКОНОВ",))
        assert await repo.search(1, "драконами") == []
        assert len(await repo.search(2, "драконами")) == 1
        await MemoryService(repo).delete_user(1)
        assert await repo.search(1, "книгой") == []
        assert await repo.recent(1) == []
        reloaded = MemoriesRepository(tmp_path)
        assert await reloaded.search(1, "дракон") == []
        assert len(await reloaded.search(2, "дракон")) == 1
        with closing(sqlite3.connect(tmp_path / "memories.db")) as db, db:
            db.execute(
                "INSERT INTO memories_fts(memories_fts, rank) VALUES('integrity-check', 1)"
            )

    asyncio.run(scenario())


def test_forgetting_other_word_forms_preserves_similar_names(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path)
        await repo.remember(1, "topic", "Обсудили рисунок с Ильёй", 1)
        await repo.remember(1, "topic", "Обсудили рисунок с Ильдаром", 2)
        await repo.remember(2, "topic", "Обсудили рисунок с Ильёй", 1)
        await repo.forget_sources(1, ("Илья",))
        assert await repo.search(1, "Илья") == []
        assert len(await repo.search(1, "Ильдар")) == 1
        assert len(await repo.search(2, "Илья")) == 1

    asyncio.run(scenario())


def test_records_written_by_old_version_are_indexed_on_reload(tmp_path: Path) -> None:
    MemoriesRepository(tmp_path)
    with closing(sqlite3.connect(tmp_path / "memories.db")) as db, db:
        db.execute(
            "INSERT INTO memories(user_id,kind,text,created_at) VALUES(1,'topic','Котами',1)"
        )
        db.execute(
            "INSERT INTO memories(user_id,kind,text,created_at) VALUES(1,'topic','и как мне',2)"
        )
    repo = MemoriesRepository(tmp_path)
    assert len(asyncio.run(repo.search(1, "кот"))) == 1
    assert len(asyncio.run(MemoriesRepository(tmp_path).recent(1))) == 2


def test_legacy_database_migrates_and_preserves_engagement(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "memories.db")) as db, db:
        db.execute(
            "CREATE TABLE memories(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, "
            "created_at REAL NOT NULL, UNIQUE(user_id,kind,text))"
        )
        db.execute("INSERT INTO memories VALUES(7,1,'topic','Рисовали драконов',1)")
        db.execute(
            "CREATE TABLE engagement(user_id INTEGER PRIMARY KEY, "
            "proactive_enabled INTEGER NOT NULL DEFAULT 1, last_user_message_at REAL NOT NULL DEFAULT 0, "
            "last_proactive_at REAL, unanswered_count INTEGER NOT NULL DEFAULT 0)"
        )
        db.execute("INSERT INTO engagement VALUES(1,0,10,20,2)")
    repo = MemoriesRepository(tmp_path)
    assert len(asyncio.run(repo.search(1, "дракон"))) == 1
    assert asyncio.run(repo.proactive_enabled(1)) is False
    with closing(sqlite3.connect(tmp_path / "memories.db")) as db:
        assert db.execute(
            "SELECT id,text,created_at,importance FROM memories"
        ).fetchone() == (7, "Рисовали драконов", 1, 1)
        assert db.execute(
            "SELECT last_user_message_at,last_proactive_at,unanswered_count FROM engagement"
        ).fetchone() == (10, 20, 2)


@pytest.mark.parametrize("missing_index", [False, True])
def test_index_rebuild_preserves_originals_and_then_tracks_changes(
    tmp_path: Path, missing_index: bool
) -> None:
    repo = MemoriesRepository(tmp_path)
    asyncio.run(repo.remember(1, "topic", "Дракон акварель", 1, importance=3))
    with closing(sqlite3.connect(tmp_path / "memories.db")) as db, db:
        if missing_index:
            db.execute("DROP TABLE memories_fts")
        else:
            db.execute("UPDATE memory_search_version SET version=0")
    reloaded = MemoriesRepository(tmp_path)
    result = asyncio.run(reloaded.search(1, "драконами"))
    assert len(result) == 1 and result[0].importance == 3
    asyncio.run(reloaded.forget_sources(1, ("Дракон",)))
    assert asyncio.run(reloaded.search(1, "драконами")) == []


def test_engine_uses_old_memory_but_never_other_user_or_group_memory(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        repo = MemoriesRepository(tmp_path)
        await repo.remember(1, "topic", "Мы рисовали драконов акварелью", 1)
        await repo.remember(2, "topic", "Дракон секрет другого пользователя", 1)
        for index in range(25):
            await repo.remember(1, "topic", f"Другая тема {index}", index + 2)
        engine, _, model, _, _, _ = _create_engine()
        engine._memory = MemoryService(MemoriesRepository(tmp_path))
        await engine.respond(1, "Что мы обсуждали про драконов?")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "Мы рисовали драконов акварелью" in prompt
        assert "секрет другого" not in prompt
        await engine.respond(
            1, "Что мы обсуждали про драконов?", use_personal_facts=False
        )
        assert (
            "Мы рисовали драконов акварелью"
            not in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine._memory.delete_user(1)
        engine._user_states.get(1).history.clear()
        await engine.respond(1, "Что мы обсуждали про драконов?")
        assert (
            "Мы рисовали драконов акварелью"
            not in model.chat.await_args.kwargs["system_prompt"]
        )

    asyncio.run(scenario())
