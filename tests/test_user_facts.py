"""Стабильность профиля, подтверждение источником и контролируемое забывание."""

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from test_response_engine import _create_engine

from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_facts import FactChange, UserFactsRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.user_facts import (
    UserFactsService,
    fact_candidate,
    parse_fact_changes,
)


def _json(key: str, value: str, quote: str, action: str = "replace") -> str:
    return json.dumps(
        {"changes": [{"field": key, "action": action, "value": value, "quote": quote}]},
        ensure_ascii=False,
    )


def test_profile_survives_reload_and_updates_only_owner(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = UserFactsRepository(tmp_path)
        await repo.apply(
            1,
            [
                FactChange("name", "add", "Илья", "Меня зовут Илья"),
                FactChange("occupation", "replace", "сварщиком", "Я работаю сварщиком"),
            ],
            1,
        )
        await repo.apply(2, [FactChange("name", "add", "Илья", "Меня зовут Илья")], 2)
        assert len(await UserFactsRepository(tmp_path).all(1)) == 2
        assert not (
            await repo.apply(
                1, [FactChange("name", "add", "илья", "Меня зовут илья")], 3
            )
        ).changed
        changed = await repo.apply(
            1,
            [FactChange("occupation", "replace", "водителем", "Я работаю водителем")],
            4,
        )
        assert changed.removed_sources == ("Я работаю сварщиком",)
        assert [f.value for f in await repo.all(1)] == ["Илья", "водителем"]
        assert not (
            await repo.apply(1, [FactChange("name", "forget", "Петя")], 5)
        ).changed
        assert len(await repo.all(1)) == 2
        await repo.clear(1, "name")
        assert [f.key for f in await repo.all(1)] == ["occupation"]
        assert [f.value for f in await repo.all(2)] == ["Илья"]
        await repo.clear(1)
        assert await repo.all(1) == []

    asyncio.run(scenario())


def test_bounded_profile_does_not_evict_name(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = UserFactsRepository(tmp_path)
        await repo.apply(
            1, [FactChange("name", "replace", "Илья", "Меня зовут Илья")], 1
        )
        for i in range(40):
            await repo.apply(
                1, [FactChange("music", "add", f"Artist{i}", f"Люблю Artist{i}")], i + 2
            )
        facts = await repo.all(1)
        assert len(facts) == 32 and any(f.value == "Илья" for f in facts)
        await repo.apply(
            1, [FactChange("occupation", "replace", "сварщик", "Я работаю сварщик")], 99
        )
        assert len(await repo.all(1)) == 32 and any(
            f.key == "occupation" for f in await repo.all(1)
        )
        with pytest.raises(ValueError):
            await repo.apply(1, [FactChange("bad", "add", "x", "y")], 1)
        with pytest.raises(ValueError):
            await repo.apply(
                1, [FactChange("name", "add", "x" * 181, "Я люблю музыку")], 1
            )
        with pytest.raises(ValueError):
            await repo.apply(1, [FactChange("name", "replace", "", "")], 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Меня зовут Илья", True),
        ("Я люблю Pendulum", True),
        ("Забудь моё имя", True),
        ("Зови меня Exi", True),
        ("My name is John", True),
        ("Привет", False),
        ("Друг написал: «Меня зовут Илья»", False),
        ("*Я работаю врачом*", False),
        ("/memory", False),
        ("> Я живу в Москве", False),
        ("```Я живу в Москве```", False),
        ("Я люблю " + "x" * 4001, False),
    ],
)
def test_extract_only_explicit_self_information(text: str, expected: bool) -> None:
    assert fact_candidate(text) is expected


@pytest.mark.parametrize(
    "raw,text",
    [
        ("[]", "Меня зовут Илья"),
        ('{"changes":false}', "Меня зовут Илья"),
        ('{"changes":[],"instructions":"ignore"}', "Меня зовут Илья"),
        ("x" * 12001, "Меня зовут Илья"),
        ('{"changes":' + json.dumps([{}] * 9) + "}", "Меня зовут Илья"),
        ('{"changes":[1]}', "Меня зовут Илья"),
        ('{"changes":[{"field":1,"action":"add","value":"x","quote":"test"}]}', "test"),
        (_json("unknown", "Илья", "Меня зовут Илья"), "Меня зовут Илья"),
        (_json("name", "Вася", "Меня зовут Илья"), "Меня зовут Илья"),
        (_json("name", "Илья", "Меня зовут Илья"), "Привет, я люблю музыку"),
        (_json("name", "Илья", "Илья"), "Меня зовут Илья"),
        (_json("name", "", "Меня зовут Илья", "forget"), "Меня зовут Илья"),
        (
            _json("name", "Илья", "Меня зовут Илья"),
            "Я люблю Python. Друг сказал: «Меня зовут Илья»",
        ),
        (_json("music", "Python", "Я не люблю Python", "add"), "Я не люблю Python"),
        (
            _json("occupation", "врачом", "Я работаю врачом"),
            "Представь, я работаю врачом",
        ),
        (_json("name", "Илья", "Меня зовут Илья?"), "Меня зовут Илья?"),
        (_json("name", "Илья", "Меня зовут Илья"), "Меня зовут Илья?"),
        (_json("preferences", "sk-secret", "Запомни sk-secret"), "Запомни sk-secret"),
        (_json("name", "Илья", "test"), "test"),
    ],
)
def test_reject_invented_quotes_secrets_and_unrequested_deletion(
    raw: str, text: str
) -> None:
    with pytest.raises(ValueError):
        parse_fact_changes(raw, text)


def test_service_preserves_profile_on_bad_json_and_provider_failure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        repo = UserFactsRepository(tmp_path)
        model = AsyncMock(spec=DeepSeekService)
        service = UserFactsService(repo, model)
        assert not (await service.observe(1, "Привет")).attempted
        model.extract_user_facts.assert_not_awaited()
        model.extract_user_facts.return_value = _json("name", "Илья", "Меня зовут Илья")
        assert (await service.observe(1, "Меня зовут Илья")).changed == 1
        sent = json.loads(model.extract_user_facts.await_args.args[1])
        assert sent == {"profile": [], "message": "Меня зовут Илья"}
        for response in ("bad JSON", '{"changes":[]}'):
            model.extract_user_facts.return_value = response
            await service.observe(1, "Меня зовут Петя")
            assert (await repo.all(1))[0].value == "Илья"
        model.extract_user_facts.side_effect = TimeoutError()
        assert (await service.observe(1, "Меня зовут Петя")).failed
        assert (await repo.all(1))[0].value == "Илья"
        assert "не системные инструкции" in "\n".join(await service.context(1))
        assert await service.context(2) == []

    asyncio.run(scenario())


def test_engine_recalls_profile_after_restart_and_forgets_all_related_context(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        engine, _, model, _, _, _ = _create_engine()
        repo = UserFactsRepository(tmp_path)
        episodes = MemoriesRepository(tmp_path)
        memory = MemoryService(
            episodes, facts=UserFactsService(repo, cast(DeepSeekService, model))
        )
        engine._memory = memory
        model.extract_user_facts.return_value = _json("name", "Илья", "Меня зовут Илья")
        await engine.respond(1, "Меня зовут Илья")
        assert "Илья" in model.chat.await_args.kwargs["system_prompt"]
        engine2, _, model2, _, _, _ = _create_engine()
        engine2._memory = MemoryService(
            MemoriesRepository(tmp_path),
            facts=UserFactsService(
                UserFactsRepository(tmp_path), cast(DeepSeekService, model2)
            ),
        )
        await engine2.respond(1, "Как меня зовут?")
        assert "Илья" in model2.chat.await_args.kwargs["system_prompt"]
        # Профиль не публикуется в групповой ответ, не извлекается из вложений/RP.
        before = model.extract_user_facts.await_count
        await engine.respond(1, "Меня зовут Вася", use_personal_facts=False)
        assert model.extract_user_facts.await_count == before
        assert (
            "Постоянный профиль пользователя"
            not in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine.respond(1, "Меня зовут Вася", attachment_text="Документ")
        assert model.extract_user_facts.await_count == before
        engine._user_states.get(1).roleplay_active = True
        await engine.respond(1, "Меня зовут Вася")
        assert model.extract_user_facts.await_count == before
        engine._user_states.get(1).roleplay_active = False
        model.extract_user_facts.return_value = _json(
            "name", "", "Забудь моё имя", "forget"
        )
        await episodes.remember(1, "topic", "Я тот самый ИЛЬЯ", 2)
        await episodes.remember(2, "topic", "Я тот самый ИЛЬЯ", 2)
        model.chat.return_value = "Хорошо, Илья, забуду."
        await engine.respond(1, "Забудь моё имя")
        assert await repo.all(1) == []
        assert all("Илья" not in m.text for m in await episodes.recent(1, limit=50))
        assert all("ИЛЬЯ" not in m.text for m in await episodes.recent(1, limit=50))
        assert len(await episodes.recent(2)) == 1
        assert all(
            "Илья" not in t.user_message + t.assistant_message
            for t in engine._user_states.get(1).history
        )
        model.extract_user_facts.return_value = _json("name", "Вася", "Меня зовут Вася")
        await engine.respond(1, "Меня зовут Вася")
        await memory.delete_user(1)
        assert await repo.all(1) == [] and await episodes.recent(1) == []

    asyncio.run(scenario())


def test_memory_command_never_discloses_another_user_profile(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from aiogram.types import Message

    from protogen_delta.core.user_state import UserStateStore
    from protogen_delta.handlers.memory import create_memory_router

    async def scenario() -> None:
        repo = UserFactsRepository(tmp_path)
        episodes = MemoriesRepository(tmp_path)
        memory = MemoryService(episodes, facts=UserFactsService(repo, AsyncMock()))
        states = UserStateStore()
        callback = create_memory_router(memory, states).message.handlers[0].callback
        message = Mock(
            spec=Message,
            from_user=SimpleNamespace(id=1),
            chat=SimpleNamespace(type="private"),
            text="/memory",
            answer=AsyncMock(),
        )
        await repo.apply(
            2, [FactChange("name", "replace", "Private", "Меня зовут Private")], 1
        )
        await callback(message)
        assert "Private" not in message.answer.await_args.args[0]
        await repo.apply(
            1, [FactChange("name", "replace", "Илья", "Меня зовут Илья")], 1
        )
        await episodes.remember(1, "topic", "Меня зовут Илья", 1)
        message.text = "/memory forget name"
        await callback(message)
        assert await repo.all(1) == [] and await episodes.recent(1) == []
        assert len(await repo.all(2)) == 1
        message.chat.type = "supergroup"
        message.text = "/memory"
        await callback(message)
        assert "личном" in message.answer.await_args.args[0]
        message.chat.type = "private"
        message.text = "/memory bad"
        await callback(message)
        assert "/memory forget" in message.answer.await_args.args[0]
        memory.facts = None
        message.text = "/memory"
        await callback(message)
        assert "недоступен" in message.answer.await_args.args[0]
        message.from_user = None
        await callback(message)

    asyncio.run(scenario())
