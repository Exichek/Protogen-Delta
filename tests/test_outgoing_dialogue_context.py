"""Реальные отправки вне движка становятся доступными для следующего ответа."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import EditMessageText, Response, SendMessage
from test_input_batches import message
from test_response_engine import _create_engine

from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.handlers.outgoing_context import (
    OutgoingContextMiddleware,
    outgoing_kind,
)
from protogen_delta.repositories.outgoing_messages import OutgoingMessagesRepository
from protogen_delta.repositories.user_state import UserStateRepository


@pytest.mark.parametrize("wrapped", [False, True])
def test_middleware_records_only_confirmed_sends(tmp_path: Path, wrapped: bool) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        middleware = OutgoingContextMiddleware(repo, clock=lambda: 100)
        bot = Bot("987654:synthetic")
        delivered = message(1, "Реальный текст", user=10)
        result: Any = Response[Any](ok=True, result=delivered) if wrapped else delivered
        send = AsyncMock(return_value=result)
        method = SendMessage(chat_id=10, text="Другой исходный текст")
        try:
            assert await middleware(send, bot, method) is result
            # Повтор того же подтверждения не дублируется.
            await middleware(send, bot, method)
            rows = await repo.pending(10, 0, 100)
            assert len(rows) == 1 and rows[0].text == "Реальный текст"
            assert await repo.pending(11, 0, 100) == ()
            with outgoing_kind("dialogue"):
                send.return_value = message(2, "Обычный ответ", user=10)
                await middleware(send, bot, method)
            await middleware(
                send, bot, EditMessageText(chat_id=10, message_id=1, text="Правка")
            )
            send.side_effect = RuntimeError("Сеть")
            with pytest.raises(RuntimeError):
                await middleware(send, bot, method)
            assert await repo.pending(10, 0, 100) == rows
        finally:
            await bot.session.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload", ["group", "caption", "sticker", "markup", "empty", "creator"]
)
def test_payload_and_authorship(tmp_path: Path, payload: str) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        middleware = OutgoingContextMiddleware(repo, clock=lambda: 100)
        bot = Bot("987654:synthetic")
        extra: dict[str, Any] = {}
        if payload == "caption":
            extra["caption"] = "Подпись к видео"
        if payload == "sticker":
            extra["sticker"] = {
                "file_id": "fake",
                "file_unique_id": "unique",
                "type": "regular",
                "width": 1,
                "height": 1,
                "is_animated": False,
                "is_video": False,
                "emoji": "😆",
            }
        delivered = message(
            1,
            (
                "<｜DSML｜calls>"
                if payload == "markup"
                else "От создателя" if payload == "creator" else None
            ),
            user=10,
            chat=-123 if payload == "group" else None,
            **extra,
        )
        try:
            with outgoing_kind("creator" if payload == "creator" else "bot"):
                await middleware(
                    AsyncMock(return_value=delivered),
                    bot,
                    SendMessage(chat_id=10, text="x"),
                )
            rows = await repo.pending(10, 0, 100)
            assert bool(rows) == (payload in {"caption", "sticker", "creator"})
            if rows and payload == "creator":
                assert rows[0].kind == "creator"
        finally:
            await bot.session.close()

    asyncio.run(run())


def test_repository_bounds_restarts_and_reset(tmp_path: Path) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        await repo.record(11, 1, "Чужой чат", "bot", 100)
        for n in range(40):
            await repo.record(10, n, "x" * 5000, "bot", 100 + n)
        restored = OutgoingMessagesRepository(tmp_path)
        rows = await restored.pending(10, 0, 200)
        assert len(rows) == 8 and all(len(m.text) == 4096 for m in rows)
        assert [m.delivered_at for m in rows] == list(range(132, 140))
        assert await restored.pending(10, rows[-1].id, 200) == ()
        assert await restored.pending(10, 0, 8 * 86400) == ()
        await restored.delete_user(10)
        assert await restored.pending(10, 0, 200) == ()
        assert await restored.pending(11, 0, 200)

    asyncio.run(run())


@pytest.mark.parametrize(
    "text",
    [
        "Автообзор: проверить датчик",
        "Новость об ИИ: ссылка",
        "Напоминаю про встречу",
        "Дополнение: полезная деталь",
        "Давай обсудим игру",
        "[Отправлен стикер 😆]",
    ],
)
def test_any_background_message_survives_active_scene_and_restart(
    tmp_path: Path, text: str
) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        await repo.record(10, 1, text, "bot", 10000)
        engine, _, model, *_ = _create_engine()
        engine._outgoing_messages = repo
        engine._user_states = UserStateStore(
            persistence=UserStateRepository(tmp_path), wall_clock=lambda: 11000
        )
        async with engine._user_states.use_conversation(10) as state:
            state.roleplay_active = True
            state.history.append(ConversationTurn("*сижу рядом*", "Старая сцена"))
        # Доставка действительно новее сохранённой сцены.
        await repo.record(10, 1, text, "bot", 10000)
        engine._user_states.get(10).history_updated_at = 9000
        # Сохранить начальные часы без перезаписи истории.
        engine._user_states._wall_clock = lambda: 9000
        async with engine._user_states.use_conversation(10):
            pass
        engine._user_states._wall_clock = lambda: 11000
        await engine.respond(10, "И как думаешь, это поможет?")
        req = model.chat.await_args.kwargs
        assert "RP PROMPT" not in req["system_prompt"]
        assert req["history"][-1].assistant_message == text
        assert req["history"][-1].user_message == ""
        assert engine._user_states.get(10).roleplay_active is True
        assert engine._user_states.get(10).history[-1].outside_roleplay
        engine._user_states = UserStateStore(
            persistence=UserStateRepository(tmp_path), wall_clock=lambda: 12000
        )
        await engine.respond(10, "А подробнее?")
        req = model.chat.await_args.kwargs
        assert "RP PROMPT" not in req["system_prompt"]
        assert sum(t.assistant_message == text for t in req["history"]) == 1
        await engine.respond(10, "*подхожу к тебе*")
        assert "RP PROMPT" in model.chat.await_args.kwargs["system_prompt"]
        assert not engine._user_states.get(10).history[-1].outside_roleplay

    asyncio.run(run())


def test_multiple_messages_actual_api_order_and_send_during_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_deepseek_service import _create_response, _create_service

    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        await repo.record(10, 1, "Первое самостоятельное", "bot", 100)
        await repo.record(10, 2, "Второе самостоятельное", "bot", 101)
        backend, _, api, _ = _create_service(monkeypatch)

        async def generate(**kwargs: Any) -> Any:
            await repo.record(10, 3, "Пришло во время генерации", "bot", 109)
            return _create_response("Разберём второе")

        api.side_effect = generate
        engine, *_ = _create_engine()
        engine._deepseek = backend
        engine._outgoing_messages = repo
        engine._user_states._wall_clock = lambda: 110
        try:
            await engine.respond(10, "Это о чём?")
            assert api.await_args is not None
            assert api.await_args.kwargs["messages"][-3:] == [
                {"role": "assistant", "content": "Первое самостоятельное"},
                {"role": "assistant", "content": "Второе самостоятельное"},
                {"role": "user", "content": "Это о чём?", "name": "tg_10"},
            ]
            await engine.respond(10, "А теперь?")
            assert api.await_args is not None
            assert (
                api.await_args.kwargs["messages"][-2]["content"]
                == "Пришло во время генерации"
            )
        finally:
            await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "scope", ["group", "no_facts", "forward", "reply", "failed_delivery"]
)
def test_other_scopes_and_delivery_failures(tmp_path: Path, scope: str) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        await repo.record(10, 1, "PRIVATE_MARKER", "bot", 100)
        engine, _, model, *_ = _create_engine()
        engine._outgoing_messages = repo
        engine._user_states._wall_clock = lambda: 110
        token = current_reply_sources.set(
            (
                ReplySource(
                    "Это?",
                    "Это?",
                    scope != "forward",
                    True,
                    reply_target_present=scope == "reply",
                ),
            )
        )
        try:
            kwargs: dict[str, Any] = {}
            if scope == "failed_delivery":
                kwargs["deliver"] = AsyncMock(side_effect=RuntimeError("не доставлено"))
                with pytest.raises(RuntimeError):
                    await engine.respond(10, "Это?", **kwargs)
                assert not engine._user_states.get(10).history
                await engine.respond(10, "Это?")
                assert (
                    model.chat.await_args.kwargs["history"][-1].assistant_message
                    == "PRIVATE_MARKER"
                )
            else:
                await engine.respond(
                    10,
                    "Это?",
                    chat_id=-123 if scope == "group" else None,
                    use_personal_facts=scope != "no_facts",
                )
                assert "PRIVATE_MARKER" not in str(model.chat.await_args)
        finally:
            current_reply_sources.reset(token)

    asyncio.run(run())


def test_real_bot_session_captures_background_services_and_excludes_normal_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from protogen_delta.repositories.creator_messages import CreatorMessagesRepository
    from protogen_delta.repositories.memories import MemoriesRepository
    from protogen_delta.repositories.reminders import RemindersRepository
    from protogen_delta.services.creator_messages import CreatorMessageService
    from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger
    from protogen_delta.services.reminders import ReminderService

    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        bot = Bot("987654:synthetic")
        counter = 0

        async def api(bot: Bot, method: Any, timeout: Any = None) -> Any:
            nonlocal counter
            counter += 1
            return message(counter, method.text, user=10)

        monkeypatch.setattr(bot.session, "make_request", api)
        bot.session.middleware.register(
            OutgoingContextMiddleware(repo, clock=lambda: 100)
        )
        engine, *_ = _create_engine()
        engine._outgoing_messages = repo
        engine._user_states._wall_clock = lambda: 110
        creator = CreatorMessageService(
            bot, CreatorMessagesRepository(tmp_path), engine._user_states
        )
        try:
            await creator.send(10, "Сообщение создателя", "message")
            reminders_repo = RemindersRepository(tmp_path)
            await reminders_repo.create(10, 10, 1, None, "Проверить встречу", 90, 1)
            await ReminderService(bot, reminders_repo, clock=lambda: 100).deliver_due()
            proactive = ProactiveMessenger(
                bot=bot,
                deepseek=AsyncMock(),
                repository=AsyncMock(spec=MemoriesRepository),
                system_prompt="Тест",
                config=ProactiveConfig(),
            )
            assert await proactive._deliver(10, "Самостоятельная тема", "text", 0)
            from test_source_access import source

            from protogen_delta.repositories.code_reviews import CodeReviewsRepository
            from protogen_delta.services.code_review import CodeReviewService

            model = AsyncMock()
            model.chat.return_value = "Автообзор: гипотеза для обсуждения"
            access = source(tmp_path)
            reviews = CodeReviewService(
                bot, model, access, CodeReviewsRepository(tmp_path), clock=lambda: 100
            )
            await reviews.run_once()
            (tmp_path / "services/sample.py").write_text("value = 43", "utf-8")
            await reviews.run_once()

            # Обычный ответ проходит тот же Telegram-клиент, но уже сохранится движком.
            async def deliver(text: str) -> None:
                await bot.send_message(10, text)

            await engine.respond(10, "Спасибо", deliver=deliver)
            rows = await repo.pending(10, 0, 110)
            assert len(rows) == 4
            assert rows[0].kind == "creator" and "встречу" in rows[1].text
            assert rows[2].text == "Самостоятельная тема"
            assert "Автообзор" in rows[3].text
            assert engine._user_states.get(10).history[-1].outgoing_id == rows[-1].id
        finally:
            await bot.session.close()

    asyncio.run(run())


def test_explicit_reply_consumes_cursor_without_later_topic_hijack(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        await repo.record(10, 1, "Уже обсуждённый отчёт", "bot", 100)
        engine, _, model, *_ = _create_engine()
        engine._outgoing_messages = repo
        engine._user_states._wall_clock = lambda: 110
        await engine.respond(10, "Понятно")
        await repo.record(10, 2, "Другой отчёт", "bot", 111)
        engine._user_states._wall_clock = lambda: 120
        token = current_reply_sources.set(
            (
                ReplySource(
                    "Про другое", "Про другое", True, True, reply_target_present=True
                ),
            )
        )
        try:
            await engine.respond(10, "Про другое")
        finally:
            current_reply_sources.reset(token)
        assert "Другой отчёт" not in str(model.chat.await_args)
        await engine.respond(10, "Продолжай эту тему")
        assert "Другой отчёт" not in str(model.chat.await_args)

    asyncio.run(run())


def test_record_failure_does_not_repeat_delivery_and_cancellation_finishes_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        middleware = OutgoingContextMiddleware(repo, clock=lambda: 100)
        bot = Bot("987654:synthetic")
        send = AsyncMock(return_value=message(1, "Подтверждено", user=10))
        original = repo.record
        monkeypatch.setattr(repo, "record", AsyncMock(side_effect=RuntimeError("база")))
        try:
            await middleware(send, bot, SendMessage(chat_id=10, text="x"))
            send.assert_awaited_once()
            entered, release = asyncio.Event(), asyncio.Event()

            async def held(*args: Any) -> None:
                entered.set()
                await release.wait()
                await original(*args)

            monkeypatch.setattr(repo, "record", held)
            task = asyncio.create_task(
                middleware(send, bot, SendMessage(chat_id=10, text="x"))
            )
            await entered.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(await repo.pending(10, 0, 100)) == 1
        finally:
            await bot.session.close()

    asyncio.run(run())


def test_review_with_active_scene_uses_report_not_release_journal(
    tmp_path: Path,
) -> None:
    from test_source_access import source

    from protogen_delta.repositories.code_reviews import CodeReviewsRepository
    from protogen_delta.services.code_review import CodeReviewService

    async def run() -> None:
        service = CodeReviewService(
            AsyncMock(),
            AsyncMock(),
            source(tmp_path),
            CodeReviewsRepository(tmp_path),
            clock=lambda: 10000,
        )
        await service.record_delivery(
            10, "Автообзор: проверить runtime_health", automatic=True
        )
        engine, _, model, *_ = _create_engine()
        engine._code_reviews = service
        engine._user_states._wall_clock = lambda: 11000
        engine._user_states.get(10).roleplay_active = True
        engine._release_journal = journal = AsyncMock()
        journal.context.return_value = "НЕ ТОТ ПАТЧНОУТ: Instagram"
        await engine.respond(10, "И как думаешь эти изменения тебе помогут?")
        req = model.chat.await_args.kwargs
        assert "RP PROMPT" not in req["system_prompt"]
        assert (
            req["history"][-1].assistant_message
            == "Автообзор: проверить runtime_health"
        )
        assert "НЕ ТОТ ПАТЧНОУТ" not in req["system_prompt"]
        journal.context.assert_not_awaited()
        await engine.respond(10, "А теперь расскажи патчноут установленной сборки")
        journal.context.assert_awaited_once()

    asyncio.run(run())


def test_first_background_message_arriving_during_reply_is_not_lost(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        engine, _, model, *_ = _create_engine()
        engine._outgoing_messages = repo
        engine._user_states._wall_clock = lambda: 110

        async def deliver(text: str) -> None:
            await repo.record(10, 1, "Первое фоновое во время ответа", "bot", 109)

        await engine.respond(10, "Вопрос", deliver=deliver)
        await engine.respond(10, "А это что?")
        assert (
            model.chat.await_args.kwargs["history"][-1].assistant_message
            == "Первое фоновое во время ответа"
        )

    asyncio.run(run())


def test_pending_messages_interleave_with_already_saved_initiative(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        for n, text in enumerate(
            ("Раннее напоминание", "Сохранённая инициатива", "Поздний обзор"), 1
        ):
            await repo.record(10, n, text, "bot", 100 + n)
        engine, _, model, *_ = _create_engine()
        engine._outgoing_messages = repo
        engine._user_states._wall_clock = lambda: 110
        state = engine._user_states.get(10)
        state.history.append(
            ConversationTurn(
                "",
                "Сохранённая инициатива",
                proactive=True,
                initiative_reason="Обсудить игру",
            )
        )
        await engine.respond(10, "Давай обсудим это")
        texts = [t.assistant_message for t in model.chat.await_args.kwargs["history"]]
        assert texts == [
            "Раннее напоминание",
            "Сохранённая инициатива",
            "Поздний обзор",
        ]
        assert [t.assistant_message for t in state.history][:-1] == texts

    asyncio.run(run())


def test_scene_invitation_keeps_scene_route(tmp_path: Path) -> None:
    async def run() -> None:
        repo = OutgoingMessagesRepository(tmp_path)
        await repo.record(10, 1, "Продолжим сцену?", "scene", 100)
        engine, _, model, *_ = _create_engine()
        engine._outgoing_messages = repo
        engine._user_states._wall_clock = lambda: 110
        state = engine._user_states.get(10)
        state.roleplay_active = True
        await engine.respond(10, "Да, продолжим")
        assert "RP PROMPT" in model.chat.await_args.kwargs["system_prompt"]
        assert not state.history[-1].outside_roleplay

    asyncio.run(run())
