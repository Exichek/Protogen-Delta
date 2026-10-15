"""Происхождение частей пачки, постоянные факты и явные команды сцены."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from test_input_batches import Harness, message
from test_response_engine import _create_engine

from protogen_delta.core.reply_capture import current_reply_sources
from protogen_delta.core.sender import Sender, current_sender
from protogen_delta.handlers.documents import create_document_router
from protogen_delta.handlers.text import create_text_router
from protogen_delta.repositories.favorites import FavoritesRepository
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_facts import FactChange, UserFactsRepository
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.favorites import FavoritesService
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.user_facts import UserFactsService


def forwarded() -> dict[str, Any]:
    """Задать подтверждённое Telegram происхождение чужой реплики."""
    return {
        "type": "hidden_user",
        "date": datetime.now(timezone.utc),
        "sender_user_name": "Другой собеседник",
    }


def install_memory(h: Harness, path: Path) -> UserFactsRepository:
    """Подключить настоящую память и сброс, оставив извлечение фактов мокированным."""
    facts = UserFactsRepository(path)
    h.engine._memory = MemoryService(
        MemoriesRepository(path),
        facts=UserFactsService(facts, h.model),
        inputs=h.inputs,
        delivery_ledger=h.ledger,
    )
    return facts


def changes(field: str, value: str, quote: str, action: str = "add") -> str:
    """Собрать подтверждённое исходной репликой изменение профиля."""
    return json.dumps(
        {
            "changes": [
                {"field": field, "action": action, "value": value, "quote": quote}
            ]
        },
        ensure_ascii=False,
    )


def test_batch_saves_each_own_fact_once_without_presentation_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        facts = install_memory(h, tmp_path)
        h.model.extract_user_facts.side_effect = [
            changes("name", "Тестовый", "Меня зовут Тестовый"),
            changes("occupation", "программистом", "Я работаю программистом"),
        ]
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(message(1, "Меня зовут Тестовый"))
            await h.feed(message(2, "Я работаю программистом"))
            await h.drain()
            assert {fact.key: fact.value for fact in await facts.all(123456)} == {
                "name": "Тестовый",
                "occupation": "программистом",
            }
            assert [
                json.loads(call.args[1])["message"]
                for call in h.model.extract_user_facts.await_args_list
            ] == ["Меня зовут Тестовый", "Я работаю программистом"]
            assert h.model.chat.await_count == h.answer.await_count == 1
            assert current_reply_sources.get() is None
        finally:
            await h.close()

    asyncio.run(run())


@pytest.mark.parametrize("with_forward", [False, True])
def test_batch_forgets_existing_fact_without_restoring_it_to_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_forward: bool
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        facts = install_memory(h, tmp_path)
        await facts.apply(
            123456, [FactChange("name", "add", "Тестовый", "Меня зовут Тестовый")], 1
        )
        h.model.extract_user_facts.return_value = changes(
            "name", "", "Забудь моё имя", "forget"
        )
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(message(1, "Забудь моё имя"))
            extra: dict[str, Any] = (
                {"forward_origin": forwarded()} if with_forward else {}
            )
            await h.feed(message(2, "Давай сменим тему", **extra))
            await h.drain()
            assert not await facts.all(123456)
            assert h.model.extract_user_facts.await_count == 1
            assert h.model.chat.await_count == h.answer.await_count == 1
            assert "Тестовый" not in str(h.model.chat.await_args)
            history = h.engine._user_states.get(123456).history
            assert (
                not history
                or history[-1].user_message == "Пользователь попросил забыть сведения."
            )
        finally:
            await h.close()

    asyncio.run(run())


def test_forwarded_fact_does_not_block_or_contaminate_own_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        facts = install_memory(h, tmp_path)
        h.model.extract_user_facts.return_value = changes(
            "name", "Тестовый", "Меня зовут Тестовый"
        )
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(message(1, "Меня зовут Чужой", forward_origin=forwarded()))
            await h.feed(message(2, "Меня зовут Тестовый"))
            await h.drain()
            assert [fact.value for fact in await facts.all(123456)] == ["Тестовый"]
            assert h.model.extract_user_facts.await_count == 1
            assert (
                json.loads(h.model.extract_user_facts.await_args.args[1])["message"]
                == "Меня зовут Тестовый"
            )
        finally:
            await h.close()

    asyncio.run(run())


def test_document_and_caption_are_not_fact_sources_but_followup_text_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        facts = install_memory(h, tmp_path)
        h.model.extract_user_facts.return_value = changes(
            "name", "Тестовый", "Меня зовут Тестовый"
        )

        async def download(file_id: str, *, destination: Any) -> None:
            destination.write("Меня зовут Содержимое файла".encode())

        monkeypatch.setattr(h.bot, "download", AsyncMock(side_effect=download))
        h.routers(
            create_document_router(h.engine, h.bot, native_work=BlockingWorkPool(2)),
            create_text_router(h.engine),
        )
        try:
            await h.feed(
                message(
                    1,
                    caption="Меня зовут Подпись",
                    document={
                        "file_id": "text",
                        "file_unique_id": "unique",
                        "file_name": "note.txt",
                    },
                )
            )
            await h.feed(message(2, "Меня зовут Тестовый"))
            await h.drain()
            assert [fact.value for fact in await facts.all(123456)] == ["Тестовый"]
            assert h.model.extract_user_facts.await_count == 1
            assert "Содержимое файла" in str(h.model.chat.await_args)
            assert h.model.chat.await_count == h.answer.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_batch_scene_commands_apply_in_source_order_and_survive_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        try:
            for index, text in enumerate(
                (
                    "Ты в мужской конфигурации",
                    "Мой персонаж: взрослый исследователь",
                    "Ты в женской конфигурации",
                    "Продолжим сцену",
                ),
                1,
            ):
                await h.feed(message(index, text))
            await h.drain()
            state = h.engine._user_states.get(123456)
            assert state.roleplay_active and state.roleplay_configuration == "female"
            assert state.roleplay_character == "взрослый исследователь"
            assert h.model.chat.await_count == h.answer.await_count == 1
            await h.feed(message(5, "*оглядываюсь вокруг*"))
            await h.drain()
            assert state.roleplay_active and state.roleplay_configuration == "female"
            assert state.roleplay_character == "взрослый исследователь"
        finally:
            await h.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "text",
    [
        "*обнял собеседника*",
        "Давай начнём рп",
        "Ты в женской конфигурации",
        "Мой персонаж: взрослый исследователь",
        "Сбрось облик",
        "Мне 15 лет",
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_forwarded_content_does_not_change_scene_appearance_or_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, batched: bool
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        state = h.engine._user_states.get(123456)
        state.delta_appearance = "Взрослый протоген с красным визором"
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(message(1, text, forward_origin=forwarded()))
            if batched:
                await h.feed(message(2, "Просто прочитай, что он написал"))
            await h.drain()
            assert not state.roleplay_active and not state.age_restricted
            assert (
                state.roleplay_configuration == "male" and not state.roleplay_character
            )
            assert state.delta_appearance == "Взрослый протоген с красным визором"
            assert h.model.chat.await_count == h.answer.await_count == 1
            assert text in str(h.model.chat.await_args)
        finally:
            await h.close()

    asyncio.run(run())


def test_forwarded_stop_keeps_active_scene_and_queued_own_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        state = h.engine._user_states.get(123456)
        state.roleplay_active = True
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(message(1, "*смотрю на окно*"))
            await h.feed(message(2, "стоп рп", forward_origin=forwarded()))
            await h.drain()
            assert state.roleplay_active
            assert h.model.chat.await_count == h.answer.await_count == 1
            assert "смотрю на окно" in str(h.model.chat.await_args)
        finally:
            await h.close()

    asyncio.run(run())


def test_own_scene_command_still_applies_after_forwarded_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(
                message(1, "Ты в мужской конфигурации", forward_origin=forwarded())
            )
            await h.feed(message(2, "Ты в женской конфигурации"))
            await h.drain()
            state = h.engine._user_states.get(123456)
            assert state.roleplay_active and state.roleplay_configuration == "female"
            assert h.model.chat.await_count == h.answer.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_direct_engine_call_uses_forward_metadata_without_batch() -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        token = current_sender.set(Sender(123456, forwarded=True))
        try:
            await engine.respond(123456, "Ты в женской конфигурации")
            state = engine._user_states.get(123456)
            assert not state.roleplay_active and state.roleplay_configuration == "male"
            assert model.chat.await_count == 1
        finally:
            current_sender.reset(token)

    asyncio.run(run())


@pytest.mark.parametrize("with_forward", [False, True])
def test_favorites_use_only_complete_batch_of_own_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_forward: bool
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        favorites = FavoritesService(FavoritesRepository(tmp_path), frozenset({123456}))
        h.engine._memory = MemoryService(
            MemoriesRepository(tmp_path), favorites=favorites
        )
        h.engine._interaction_classifier = InteractionClassifier(
            h.model, "Настроение", "Оскорбления", favorite_prompt="Избранное"
        )
        h.model.classify_interaction.return_value = json.dumps(
            {
                "mood": "playful",
                "insult": "none",
                "favorite": "humor",
                "reasoning": False,
            }
        )
        h.routers(create_text_router(h.engine))
        try:
            first, second = (
                "Тостер собрался в отпуск",
                "и попросил розетку поливать цветы",
            )
            await h.feed(message(1, first))
            extra: dict[str, Any] = (
                {"forward_origin": forwarded()} if with_forward else {}
            )
            await h.feed(message(2, second, **extra))
            await h.drain()
            stored = await favorites.repository.recent(123456)
            assert len(stored) == (0 if with_forward else 1)
            if stored:
                assert stored[0].text == first + "\n" + second
            assert h.model.chat.await_count == h.answer.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())


def test_group_batch_never_extracts_private_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        h = Harness(tmp_path, monkeypatch)
        facts = install_memory(h, tmp_path)
        h.routers(create_text_router(h.engine))
        try:
            await h.feed(message(1, "Меня зовут Тестовый", chat=-100))
            await h.feed(message(2, "Я работаю программистом", chat=-100))
            await h.drain()
            h.model.extract_user_facts.assert_not_awaited()
            assert not await facts.all(123456)
            assert h.model.chat.await_count == h.answer.await_count == 1
        finally:
            await h.close()

    asyncio.run(run())
