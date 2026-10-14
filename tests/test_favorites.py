"""Избранное изолировано, долговечно, ограничено и выбирается без нового запроса."""

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_action_handlers import message
from test_maintenance import seed
from test_response_engine import _create_engine

from protogen_delta.core.favorite_messages import FAVORITE_LABELS, FavoriteReason
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.favorites import create_favorites_router
from protogen_delta.maintenance import copy_snapshot
from protogen_delta.repositories.favorites import FavoritesRepository
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.services.deepseek import DeepSeekConnectionError, DeepSeekService
from protogen_delta.services.dialogue_archive import dialogue_owner
from protogen_delta.services.favorites import FavoritesService
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.tools import ToolExecutor, default_registry

TEXT = "Наш проект ORBIT заработал: теперь баги работают на нас, а не мы на баги :)"


@pytest.mark.parametrize("reason", list(FAVORITE_LABELS))
def test_exact_text_reason_owner_and_search_survive_restart(
    tmp_path: Path, reason: FavoriteReason
) -> None:
    async def run() -> None:
        repo = FavoritesRepository(tmp_path)
        original = TEXT + "\nВторой абзац с тестами."
        assert await repo.add(1, original, reason, 10000)
        assert not await repo.add(1, original, reason, 11000)
        reopened = FavoritesRepository(tmp_path)
        result = await reopened.search(1, "тесты ORBIT")
        assert (
            len(result) == 1
            and result[0].text == original
            and result[0].reason == reason
        )
        assert await reopened.get(1, result[0].id) == result[0]
        assert await reopened.get(2, result[0].id) is None
        assert not await reopened.search(2, "ORBIT")
        assert not await reopened.recent(1, limit=0)
        assert not await reopened.search(1, "ORBIT", limit=0)
        assert not await reopened.search(1, "!!!")
        assert not await reopened.remove(2, result[0].id)
        assert await reopened.remove(1, result[0].id)
        assert not await reopened.search(1, "ORBIT")
        assert not await reopened.remove(1, result[0].id)

    asyncio.run(run())


def test_limits_are_atomic_durable_and_moscow_day_does_not_bypass_cooldown(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = FavoritesRepository(tmp_path)
        chosen = await asyncio.gather(
            *(repo.add(1, TEXT + str(index), "humor", 10000) for index in range(8))
        )
        assert sum(chosen) == 1
        for index in range(1, 5):
            assert await repo.add(
                1, TEXT + f" next{index}", "humor", 10000 + index * 900
            )
        assert not await FavoritesRepository(tmp_path).add(
            1, TEXT + " sixth", "humor", 14500
        )
        assert await repo.add(1, TEXT + " next-day", "humor", 75600)
        assert await repo.add(2, TEXT + " midnight", "support", 75599)
        assert not await repo.add(2, TEXT + " midnight2", "support", 75601)
        assert await repo.add(2, TEXT + " midnight3", "support", 76499)

    asyncio.run(run())


@pytest.mark.parametrize("owner", [0, -1, True, 2**63])
def test_invalid_owners_cannot_read_or_write(tmp_path: Path, owner: int) -> None:
    async def run() -> None:
        repo = FavoritesRepository(tmp_path)
        for operation in (
            repo.add(owner, TEXT, "humor", 0),
            repo.recent(owner),
            repo.search(owner, "x"),
            repo.get(owner, 1),
            repo.remove(owner, 1),
            repo.delete_user(owner),
            repo.forget_sources(owner, (TEXT,)),
        ):
            with pytest.raises(ValueError):
                await operation

    asyncio.run(run())


@pytest.mark.parametrize(
    "text,reason,at",
    [
        ("", "humor", 0),
        ("x" * 8001, "support", 0),
        (TEXT, "fake", 0),
        (TEXT, "humor", -1),
        (TEXT, "humor", float("nan")),
        (TEXT, "humor", True),
        (TEXT, "humor", float("inf")),
        (TEXT, "humor", 253402300799),
    ],
)
def test_invalid_records_never_reach_database(
    tmp_path: Path, text: str, reason: FavoriteReason, at: float
) -> None:
    with pytest.raises(ValueError):
        asyncio.run(FavoritesRepository(tmp_path).add(1, text, reason, at))


def test_backup_restore_and_fts_rebuild_include_favorites(tmp_path: Path) -> None:
    # seed() использует asyncio.run, поэтому подготовка до цикла.
    source = tmp_path / "data"
    seed(source)

    async def prepared() -> None:
        repo = FavoritesRepository(source)
        await repo.add(1, TEXT, "shared_success", 10000)
        initial = await repo.recent(1)
        copy_snapshot(source, tmp_path / "backup")
        await repo.delete_user(1)
        copy_snapshot(tmp_path / "backup", tmp_path / "restored")
        assert (
            await FavoritesRepository(tmp_path / "restored").search(1, "ORBIT")
            == initial
        )
        with closing(sqlite3.connect(tmp_path / "restored/favorites.db")) as db, db:
            db.executescript(
                "DROP TRIGGER favorites_insert; DROP TRIGGER favorites_delete; DROP TABLE favorites_fts;"
            )
        assert (
            await FavoritesRepository(tmp_path / "restored").search(1, "ORBIT")
            == initial
        )

    asyncio.run(prepared())


@pytest.mark.parametrize(
    "favorite,mood,insult,expected",
    [(reason, "playful", "none", reason) for reason in FAVORITE_LABELS]
    + [
        (None, "neutral", "none", None),
        ("unknown", "sweet", "none", None),
        ({"instruction": "x"}, "sweet", "none", None),
        ("support", "angry", "none", None),
        ("humor", "playful", "direct", None),
        ("support", "horny", "none", None),
    ],
)
def test_selector_is_bounded_json_and_reuses_one_call(
    favorite: object, mood: str, insult: str, expected: object
) -> None:
    model = AsyncMock(spec=DeepSeekService)
    model.classify_interaction.return_value = json.dumps(
        {"mood": mood, "insult": insult, "favorite": favorite}
    )
    classifier = InteractionClassifier(
        model, "mood", "insult", favorite_prompt="rare choices"
    )
    result = asyncio.run(classifier.classify(TEXT))
    assert result.favorite == expected and result.mood == mood
    model.classify_interaction.assert_awaited_once()
    model.chat.assert_not_awaited()


def test_empty_selector_rules_rejected_and_legacy_flags_do_not_create_favorites() -> (
    None
):
    model = AsyncMock(spec=DeepSeekService)
    with pytest.raises(ValueError):
        InteractionClassifier(model, "mood", "insult", favorite_prompt="")
    model.classify_interaction.return_value = (
        '{"mood":"sweet","insult":"none","favorite":"support"}'
    )
    assert (
        asyncio.run(
            InteractionClassifier(model, "mood", "insult").classify(TEXT)
        ).favorite
        is None
    )


def test_scoped_tool_context_forget_and_reset_never_mix_owners(tmp_path: Path) -> None:
    async def run() -> None:
        repo = FavoritesRepository(tmp_path)
        service = FavoritesService(repo, frozenset({1, 2}), clock=lambda: 10000)
        assert await service.observe(1, TEXT, "shared_success")
        assert await service.observe(
            2, "Поддержка SECOND_OWNER_440 для другого человека", "support"
        )
        assert await service.context(1, "ORBIT")
        assert await service.context(1, "Что тебе понравилось?")
        assert not await service.context(3, "ORBIT")
        assert not await service.context(1, "UNKNOWN_339")
        assert not (await service.search_tool({"query": "ORBIT"}))["available"]
        with dialogue_owner(1):
            assert len((await service.search_tool({"query": ""}))["messages"]) == 1
            assert (
                len((await service.search_tool({"query": "последние"}))["messages"])
                == 1
            )
            assert not (await service.search_tool({"query": "SECOND_OWNER_440"}))[
                "messages"
            ]
            with pytest.raises(ValueError):
                await service.search_tool({"query": "x" * 201})
            executor = ToolExecutor(default_registry(favorites=service))
            assert '"ok": false' in await executor.execute(
                "search_favorites", '{"query":"ORBIT","user_id":"2"}'
            )
        memory = MemoryService(MemoriesRepository(tmp_path), favorites=service)
        assert len("".join(await memory.context(1, "ORBIT", char_limit=35))) <= 35
        await repo.forget_sources(1, ("работает совершенно другой проект",))
        assert await repo.recent(1)
        await repo.forget_sources(1, ("ORBIT",))
        assert not await repo.search(1, "ORBIT") and await repo.recent(2)
        await memory.delete_user(2)
        assert not await repo.recent(2)

    asyncio.run(run())


@pytest.mark.parametrize(
    "text",
    [
        "/command " + TEXT,
        "> " + TEXT,
        "```python\n" + TEXT + "\n```",
        "«" + TEXT + "»",
        "API_KEY=sk-abcdefghijklmnopqrst secret text",
        "Пароль: hunter2 very long secret text",
        "Спасибо!",
        "x" * 8001,
    ],
)
def test_commands_quotes_and_secrets_are_not_silently_liked(
    tmp_path: Path, text: str
) -> None:
    async def run() -> None:
        service = FavoritesService(
            FavoritesRepository(tmp_path), frozenset({1}), clock=lambda: 10000
        )
        assert not await service.observe(1, text, "support")
        assert not await service.repository.recent(1)
        assert not await service.observe(3, TEXT, "humor")
        assert not await service.observe(1, TEXT, None)

    asyncio.run(run())


@pytest.mark.parametrize(
    "scope",
    [
        "private",
        "group",
        "other_owner",
        "forward",
        "attachment",
        "override",
        "failed",
        "empty",
    ],
)
def test_engine_only_confirms_real_selection_for_eligible_private_input(
    tmp_path: Path, scope: str
) -> None:
    async def run() -> None:
        engine, _, model, insult, mood, _ = _create_engine()
        service = FavoritesService(
            FavoritesRepository(tmp_path), frozenset({1}), clock=lambda: 10000
        )
        engine._memory = MemoryService(MemoriesRepository(tmp_path), favorites=service)
        engine._interaction_classifier = InteractionClassifier(
            model, "mood", "insult", favorite_prompt="rare choices"
        )
        model.classify_interaction.return_value = (
            '{"mood":"playful","insult":"none","favorite":"humor"}'
        )
        kwargs: dict[str, object] = {}
        owner = 2 if scope == "other_owner" else 1
        if scope == "group":
            kwargs["chat_id"] = -10
        if scope == "forward":
            kwargs["use_personal_facts"] = False
        if scope == "attachment":
            kwargs["attachment_text"] = "A document"
        if scope == "override":
            kwargs["model_message_override"] = "trusted media input"
        if scope == "failed":
            model.chat.side_effect = DeepSeekConnectionError("failed")
        if scope == "empty":
            model.chat.return_value = "   "
        reply = await engine.respond(owner, TEXT, **kwargs)  # type: ignore[arg-type]
        assert bool(await service.repository.recent(1)) is (scope == "private")
        assert ("⭐ В избранном" in reply) is (scope == "private")
        assert model.classify_interaction.await_count == 1
        insult.classify.assert_not_awaited()
        mood.classify.assert_not_awaited()
        if scope == "private":
            again = await engine.respond(1, TEXT)
            assert (
                "⭐ В избранном" not in again
                and len(await service.repository.recent(1)) == 1
            )
            await engine.reset_user(1, cleanup=lambda: engine._memory.delete_user(1))  # type: ignore[union-attr]
            assert not await service.repository.recent(1)

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["allowed", "outsider", "group", "forward", "bot"])
def test_command_is_private_and_never_calls_llm(tmp_path: Path, reason: str) -> None:
    async def run() -> None:
        service = FavoritesService(
            FavoritesRepository(tmp_path), frozenset({1}), clock=lambda: 10000
        )
        await service.observe(1, TEXT, "humor")
        router = create_favorites_router(service, UserStateStore())
        incoming = message("/favorites", user=2 if reason == "outsider" else 1)
        if reason == "group":
            incoming.chat.type = "group"
        if reason == "forward":
            incoming.forward_origin = object()
        if reason == "bot":
            incoming.from_user.is_bot = True
        handler = router.message.handlers[0].callback
        await handler(incoming)
        assert ("ORBIT" in incoming.answer.await_args.args[0]) is (reason == "allowed")
        if reason == "allowed":
            entry = (await service.repository.recent(1))[0]
            for query in (
                "ORBIT",
                str(entry.id),
                "x" * 201,
                "remove nope",
                "0",
                "9" * 30,
            ):
                incoming.text = "/favorites " + query
                await handler(incoming)
            incoming.text = "/favorites remove " + str(entry.id)
            await handler(incoming)
            assert not await service.repository.recent(1)
            await handler(incoming)
            assert "Такой записи" in incoming.answer.await_args.args[0]

    asyncio.run(run())
