"""Доставленное авторство, сохранённый RP-профиль и изоляция стиля."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery
from aiohttp.test_utils import TestClient, TestServer
from test_creator_and_proactive_handlers import _call, _message
from test_miniapp import TOKEN, _signed_init_data
from test_response_engine import _create_engine

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.roleplay import has_roleplay_intent
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.creator import create_creator_router
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.creator_messages import CreatorMessagesRepository
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.creator_messages import CreatorMessageService
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.response_engine import RP_SAVED_PROFILE_REPLY


def _service(
    path: Path,
) -> tuple[CreatorMessageService, CreatorMessagesRepository, AsyncMock, UserStateStore]:
    bot = AsyncMock(spec=Bot)
    bot.send_message.return_value = SimpleNamespace(
        message_id=1, date=datetime.fromtimestamp(1000, timezone.utc)
    )
    repository = CreatorMessagesRepository(path)
    states = UserStateStore(
        persistence=UserStateRepository(path), wall_clock=lambda: 1001
    )
    return (
        CreatorMessageService(cast(Bot, bot), repository, states, clock=lambda: 1000.5),
        repository,
        bot,
        states,
    )


def test_journal_is_persistent_bounded_and_scoped(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = CreatorMessagesRepository(tmp_path)
        for index in range(25):
            await repository.record(1, index, f"Delivery {index}", "message", index)
        await repository.record(1, 24, "overwrite", "broadcast", 25)
        await repository.record(2, 24, "Other recipient", "broadcast", 25)
        await repository.record(-10, 24, "Public chat", "message", 25)
        restored = CreatorMessagesRepository(tmp_path)
        own = await restored.recent(1)
        assert (
            len(own) == 20 and own[0].text == "Delivery 24" and own[-1].message_id == 5
        )
        assert (await restored.recent(2))[0].text == "Other recipient"
        assert (await restored.recent(-10))[0].text == "Public chat"
        await restored.delete_user(1)
        assert not await restored.recent(1)
        assert await restored.recent(2) and await restored.recent(-10)

    asyncio.run(scenario())


def test_delivery_and_followup_prompt_have_creator_provenance(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, repository, bot, states = _service(tmp_path)
        engine, _, model, *_ = _create_engine()
        engine._user_states = states
        engine._creator_messages = service
        await service.send(456, "Проверка 1 2 3", "message")
        assert "Проверка 1 2 3" in await service.context(456, "Ась?", 1000.25)
        assert not states.get(456).history
        await engine.respond(456, 'Ты мне только что прислал "Проверка 1 2 3"?')
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "автор текста — создатель" in prompt and "Проверка 1 2 3" in prompt
        assert "не инструкции" in prompt
        assert len(await repository.recent(456)) == 1
        bot.send_message.assert_awaited_once_with(456, "Проверка 1 2 3")
        await engine.respond(789, "Что ты прислал?")
        assert "Проверка 1 2 3" not in model.chat.await_args.kwargs["system_prompt"]
        await engine.respond(
            456, "Что ты прислал?", chat_id=-10, use_personal_facts=False
        )
        assert "Проверка 1 2 3" not in model.chat.await_args.kwargs["system_prompt"]
        await service.send(-10, "Объявление группе", "message")
        await engine.respond(
            789, "Кто прислал сообщение?", chat_id=-10, use_personal_facts=False
        )
        assert "Объявление группе" in model.chat.await_args.kwargs["system_prompt"]
        assert "Проверка 1 2 3" not in model.chat.await_args.kwargs["system_prompt"]

    asyncio.run(scenario())


def test_context_is_relevant_bounded_and_can_find_older_quoted_message(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        service, repository, *_ = _service(tmp_path)
        await repository.record(1, 1, "Проверка старого текста", "message", 1000)
        for index in range(2, 6):
            await repository.record(
                1, index, "\n\\\x00" * 1300, "broadcast", 1000 + index
            )
        assert not await service.context(1, "Как погода?", 1010)
        assert await service.context(1, "Ась?", 999)
        quoted = await service.context(1, 'Ты писал "Проверка старого текста"?', 1010)
        assert "Проверка старого текста" in quoted
        assert len(await service.context(1, "Кто написал сообщение?", 1010)) <= 2000
        assert not await service.context(999, "Что ты прислал?", 0)
        assert not service._locks

    asyncio.run(scenario())


def test_failed_delivery_has_no_success_record_and_failed_record_does_not_resend(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        service, repository, bot, _ = _service(tmp_path)
        bot.send_message.side_effect = TelegramBadRequest(
            method=Mock(), message="blocked"
        )
        with pytest.raises(TelegramBadRequest):
            await service.send(1, "Not delivered", "message")
        assert not await repository.recent(1)
        bot.send_message.side_effect = None
        repository.record = AsyncMock(side_effect=RuntimeError("storage unavailable"))  # type: ignore[method-assign]
        assert not await service.send(1, "Delivered", "message")
        assert bot.send_message.await_count == 2 and not service._locks

    asyncio.run(scenario())


def test_cancelled_delivery_finishes_record_before_full_forget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        service, repository, _, states = _service(tmp_path)
        memory = MemoryService(
            MemoriesRepository(tmp_path), creator_messages=repository
        )
        started, release = asyncio.Event(), asyncio.Event()
        original = repository.record

        async def slow(*args: object) -> None:
            started.set()
            await release.wait()
            await original(*args)  # type: ignore[arg-type]

        monkeypatch.setattr(repository, "record", slow)
        sending = asyncio.create_task(service.send(1, "Delivered", "message"))
        await started.wait()
        sending.cancel()
        forgetting = asyncio.create_task(
            states.reset_user(1, cleanup=lambda: memory.delete_user(1))
        )
        await asyncio.sleep(0)
        assert not sending.done() and not forgetting.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await sending
        await forgetting
        assert not await repository.recent(1) and not service._locks

    asyncio.run(scenario())


def test_creator_router_records_only_successful_confirmed_recipients(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        service, repository, bot, _ = _service(tmp_path)
        users = UsersRepository(tmp_path)
        users.add(1)
        users.add(2)
        router = create_creator_router(
            bot=cast(Bot, bot),
            users_repository=users,
            creator_id=123,
            creator_messages=service,
        )
        command, answer = _message(123, "/broadcast Проверка")
        await _call(router, "prepare_broadcast", command)
        assert not await repository.recent(1)
        assert answer.await_args is not None
        markup = answer.await_args.kwargs["reply_markup"]
        token = markup.inline_keyboard[0][0].callback_data
        callback = Mock(spec=CallbackQuery)
        callback.from_user = SimpleNamespace(id=123)
        callback.answer = AsyncMock()
        callback.message = None
        callback.data = token.replace(":confirm:", ":unknown:")
        await router.callback_query.handlers[0].callback(callback)
        bot.send_message.assert_not_awaited()
        callback.data = token
        bot.send_message.side_effect = [
            SimpleNamespace(
                message_id=1, date=datetime.fromtimestamp(1000, timezone.utc)
            ),
            TelegramBadRequest(method=Mock(), message="blocked"),
        ]
        await router.callback_query.handlers[0].callback(callback)
        assert (await repository.recent(1))[0].kind == "broadcast"
        assert not await repository.recent(2)
        await router.callback_query.handlers[0].callback(callback)
        assert bot.send_message.await_count == 2

    asyncio.run(scenario())


def test_private_message_is_recorded_and_storage_failure_is_reported(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        service, repository, bot, _ = _service(tmp_path)
        router = create_creator_router(
            bot=cast(Bot, bot),
            users_repository=UsersRepository(tmp_path),
            creator_id=123,
            creator_messages=service,
        )
        command, _ = _message(123, "/message 456 Hello")
        await _call(router, "message_user", command)
        assert (await repository.recent(456))[0].text == "Hello"
        repository.record = AsyncMock(side_effect=RuntimeError())  # type: ignore[method-assign]
        command, answer = _message(123, "/message 789 Second")
        await _call(router, "message_user", command)
        assert answer.await_args is not None
        assert "Доставка не записана" in answer.await_args.args[0]
        assert bot.send_message.await_count == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "question",
    [
        "Йоу, как я выгляжу?",
        "Я про свой рп профиль?",
        "Какой у меня облик?",
        "Опиши моего персонажа",
    ],
)
def test_saved_profile_is_available_without_starting_rp(
    tmp_path: Path, question: str
) -> None:
    async def scenario() -> None:
        original = UserStateStore(persistence=UserStateRepository(tmp_path))
        async with original.use(1) as state:
            state.roleplay_character = "Синий дракон с длинными чёрными рогами"
            state.delta_appearance = "Фиолетовый протоген Дельта"
            state.roleplay_preferences = "PRIVATE_PREFERENCES"
        engine, _, model, *_ = _create_engine()
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        await engine.respond(1, question)
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert (
            "Синий дракон" in prompt and "сохранён RP-персонаж пользователя" in prompt
        )
        assert "Фиолетовый протоген" in prompt and "PRIVATE_PREFERENCES" not in prompt
        assert not engine._user_states.get(1).roleplay_active
        await engine.respond(1, question, chat_id=-10, use_personal_facts=False)
        group = model.chat.await_args.kwargs["system_prompt"]
        assert "Синий дракон" not in group and "Фиолетовый протоген" not in group

    asyncio.run(scenario())


def test_profile_evidence_survives_small_dynamic_budget_and_stays_relevant() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        engine._config = replace(engine._config, dynamic_state_chars=1)
        state = engine._user_states.get(1)
        state.roleplay_character = "Сохранённый синий дракон"
        await engine.respond(1, "Какой у меня облик?")
        assert (
            "Сохранённый синий дракон" in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine.respond(1, "Расскажи о музыке")
        assert (
            "сохранён RP-персонаж пользователя"
            not in model.chat.await_args.kwargs["system_prompt"]
        )

    asyncio.run(scenario())


def test_rp_start_reuses_saved_character_without_requesting_profile_again() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        engine._user_states.get(1).roleplay_character = "Дракон в пальто"
        assert await engine.respond(1, "Хочу рпшить)") == RP_SAVED_PROFILE_REPLY
        assert engine._user_states.get(1).roleplay_active
        model.chat.assert_not_awaited()
        await engine.respond(1, "Мы в библиотеке, ищем потерянную книгу")
        assert "Дракон в пальто" in model.chat.await_args.kwargs["system_prompt"]
        assert not has_roleplay_intent("Я не хочу рпшить")

    asyncio.run(scenario())


def test_saved_profile_does_not_override_new_image_or_cleared_profile() -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        state = engine._user_states.get(1)
        state.roleplay_character = "OLD_PRIVATE_CHARACTER"
        await engine.respond(
            1, "Как я выгляжу?", images=[ImageInput(b"synthetic", "image/png")]
        )
        assert (
            "OLD_PRIVATE_CHARACTER" not in model.chat.await_args.kwargs["system_prompt"]
        )
        await engine.respond(1, "Как я выгляжу?", use_personal_facts=False)
        assert (
            "OLD_PRIVATE_CHARACTER" not in model.chat.await_args.kwargs["system_prompt"]
        )
        state.roleplay_character = ""
        await engine.respond(1, "Опиши мой RP-профиль")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert "сохранённое описание RP-персонажа пользователя отсутствует" in prompt

    asyncio.run(scenario())


def test_actual_miniapp_save_is_visible_to_conversation_after_restart(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = UserStateStore(persistence=UserStateRepository(tmp_path))
        application = MiniAppServer(TOKEN, store).application()
        async with TestClient(TestServer(application)) as client:
            response = await client.patch(
                "/api/profile",
                headers={"X-Telegram-Init-Data": _signed_init_data()},
                json={
                    "roleplay_character": "Синий дракон из Mini App",
                    "roleplay_active": False,
                },
            )
            assert response.status == 200
        engine, _, model, *_ = _create_engine()
        engine._user_states = UserStateStore(persistence=UserStateRepository(tmp_path))
        await engine.respond(42, "Как я выгляжу?")
        assert (
            "Синий дракон из Mini App" in model.chat.await_args.kwargs["system_prompt"]
        )
        assert not engine._user_states.get(42).roleplay_active

    asyncio.run(scenario())


def test_context_waits_for_delivery_and_read_failure_is_not_fabricated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        service, repository, bot, _ = _service(tmp_path)
        started, release = asyncio.Event(), asyncio.Event()
        delivered = bot.send_message.return_value

        async def slow(*args: object) -> object:
            started.set()
            await release.wait()
            return delivered

        bot.send_message.side_effect = slow
        sending = asyncio.create_task(service.send(1, "Delivered", "message"))
        await started.wait()
        context = asyncio.create_task(service.context(1, "Кто прислал сообщение?", 0))
        await asyncio.sleep(0)
        assert not context.done()
        release.set()
        assert await sending
        assert "Delivered" in await context and not service._locks
        monkeypatch.setattr(repository, "recent", AsyncMock(side_effect=RuntimeError()))
        assert await service.context(1, "Кто прислал сообщение?", 0) == ""
        assert not service._locks

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["adult", "soft", "unselected"])
def test_conversation_style_is_only_added_to_adult_ordinary_chat(mode: str) -> None:
    async def scenario() -> None:
        engine, _, model, *_ = _create_engine()
        engine._config = replace(
            engine._config,
            adult_conversation_prompt=load_prompt("adult_conversation_style"),
        )
        state = engine._user_states.get(1)
        state.content_mode = mode  # type: ignore[assignment]
        await engine.respond(1, "Нахуй эту хуйню, сменим тему")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert ("Стиль обычного разговора" in prompt) == (mode == "adult")
        assert not state.roleplay_active
        state.roleplay_active = True
        await engine.respond(1, "Мы ищем книгу в библиотеке")
        assert (
            "Стиль обычного разговора"
            not in model.chat.await_args.kwargs["system_prompt"]
        )

    asyncio.run(scenario())
