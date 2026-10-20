"""Отправлять только полезное дополнение к ещё актуальному доставленному ответу."""

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot
from test_input_operations import _event
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.core.input_operations import (
    InputOperations,
    ReceiveInputsMiddleware,
)
from protogen_delta.core.llm_budget import current_llm_user
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.repositories.reply_supplements import ReplySupplementLimits
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.delivery_ledger import current_delivery
from protogen_delta.services.reply_supplements import (
    ReplySupplements,
    supplement_eligible,
    supplement_text,
)


def test_receiving_a_command_cancels_addition_before_handler_runs() -> None:
    async def run() -> None:
        middleware = ReceiveInputsMiddleware(InputOperations())
        cancel = Mock()
        middleware.on_input = cancel

        async def handler(event: object, data: object) -> None:
            cancel.assert_called_once_with(TEST_USER_ID)

        await middleware(handler, _event(TEST_USER_ID, "/reset"), {})

    asyncio.run(run())


def test_successful_private_delivery_schedules_exact_confirmed_turn() -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        model.chat.return_value = ANSWER
        service = Mock(spec=ReplySupplements)
        engine.supplements = service
        await engine.respond_and_deliver(TEST_USER_ID, TURN.user_message, AsyncMock())
        service.schedule.assert_called_once_with(
            TEST_USER_ID,
            engine._user_states.get(TEST_USER_ID).history[-1],
            service.cancel.return_value,
        )

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["silence", "model", "send"])
def test_optional_failure_never_retries_or_adds_false_history(
    tmp_path: Path, failure: str
) -> None:
    async def run() -> None:
        states = UserStateStore()
        states.get(TEST_USER_ID).history.append(TURN)
        model = AsyncMock(spec=DeepSeekService)
        model.draft_supplement.return_value = json.dumps(
            {"text": "" if failure == "silence" else EXTRA}
        )
        if failure == "model":
            model.draft_supplement.side_effect = RuntimeError("model unavailable")
        bot = Mock(spec=Bot)
        bot.send_message = AsyncMock(side_effect=RuntimeError("unknown delivery"))
        service = ReplySupplements(
            bot,
            model,
            states,
            InputOperations(),
            ReplySupplementLimits(tmp_path),
            style_prompt="persona",
            sleep=AsyncMock(),
        )
        service.schedule(TEST_USER_ID, TURN, service.cancel(TEST_USER_ID))
        await asyncio.gather(*service._workers)
        assert list(states.get(TEST_USER_ID).history) == [TURN]
        assert bot.send_message.await_count == int(failure == "send")
        assert model.draft_supplement.await_count == 1
        await service.close()

    asyncio.run(run())


ANSWER = "DNS сопоставляет имя сайта с адресом сервера. " * 5
EXTRA = "Можно представить DNS как адресную книгу: имя знакомое, а нужный адрес она подсказывает за тебя."
TURN = ConversationTurn("Объясни, что такое DNS?", ANSWER)


@pytest.mark.parametrize(
    "raw",
    [
        "bad json",
        "[]",
        "{}",
        '{"text":""}',
        json.dumps({"text": "длинно " * 100}),
        json.dumps({"text": "Открой https://example.org"}),
        json.dumps({"text": "Число 42"}),
        json.dumps({"text": "две\nстроки"}),
        json.dumps({"text": ANSWER[:80]}),
    ],
)
def test_unhelpful_or_unbounded_addition_is_silent(raw: str) -> None:
    assert not supplement_text(raw, ANSWER)


@pytest.mark.parametrize(
    "question",
    [
        "Как дела?",
        "Спасибо!",
        "Расскажи одну короткую шутку",
        "Объясни DNS коротко",
        "Объясни DNS без дополнений",
    ],
)
def test_short_requests_do_not_get_a_second_answer(question: str) -> None:
    assert not supplement_eligible(ConversationTurn(question, ANSWER))


def test_cooldown_is_atomic_persistent_and_limited_per_moscow_day(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [1_800_000_000.0]
        limits = ReplySupplementLimits(tmp_path, clock=lambda: now[0])
        assert (
            sum(await asyncio.gather(*(limits.reserve(TEST_USER_ID) for _ in range(8))))
            == 1
        )
        limits = ReplySupplementLimits(tmp_path, clock=lambda: now[0])
        assert not await limits.reserve(TEST_USER_ID)
        for _ in range(3):
            now[0] += 600
            assert await limits.reserve(TEST_USER_ID)
        now[0] += 600
        assert not await limits.reserve(TEST_USER_ID)
        now[0] += 86400
        assert await limits.reserve(TEST_USER_ID)

    asyncio.run(run())


def test_addition_is_delayed_remembered_and_has_clean_context(tmp_path: Path) -> None:
    async def run() -> None:
        states = UserStateStore()
        states.get(TEST_USER_ID).history.append(TURN)
        model = AsyncMock(spec=DeepSeekService)
        bot = Mock(spec=Bot)
        bot.send_message = AsyncMock()
        clock = [0.0]
        delays = []

        async def draft(*args: str) -> str:
            assert current_delivery.get() is None
            assert current_llm_user.get() == TEST_USER_ID
            clock[0] += 2
            return json.dumps({"text": EXTRA})

        async def sleep(seconds: float) -> None:
            delays.append(seconds)
            clock[0] += seconds

        model.draft_supplement.side_effect = draft
        service = ReplySupplements(
            bot,
            model,
            states,
            InputOperations(),
            ReplySupplementLimits(tmp_path),
            style_prompt="persona",
            delay=lambda: 10,
            sleep=sleep,
            clock=lambda: clock[0],
        )
        token = current_delivery.set(Mock())
        try:
            service.schedule(TEST_USER_ID, TURN, service.cancel(TEST_USER_ID))
        finally:
            current_delivery.reset(token)
        await asyncio.gather(*service._workers)
        assert delays == [8]
        bot.send_message.assert_awaited_once()
        assert bot.send_message.await_args.args[:2] == (TEST_USER_ID, EXTRA)
        history = states.get(TEST_USER_ID).history
        assert len(history) == 1 and history[0].user_message == TURN.user_message
        assert history[0].assistant_message == ANSWER + "\n\n" + EXTRA
        await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("action", ["new_input", "reset", "close", "history_changed"])
def test_pending_addition_cannot_survive_new_activity_or_reset(
    tmp_path: Path, action: str
) -> None:
    async def run() -> None:
        states = UserStateStore()
        states.get(TEST_USER_ID).history.append(TURN)
        model = AsyncMock(spec=DeepSeekService)
        model.draft_supplement.return_value = json.dumps({"text": EXTRA})
        bot = Mock(spec=Bot)
        bot.send_message = AsyncMock()
        operations = InputOperations()
        waiting = asyncio.Event()
        release = asyncio.Event()

        async def sleep(seconds: float) -> None:
            waiting.set()
            await release.wait()

        service = ReplySupplements(
            bot,
            model,
            states,
            operations,
            ReplySupplementLimits(tmp_path),
            style_prompt="persona",
            sleep=sleep,
        )
        service.schedule(TEST_USER_ID, TURN, service.cancel(TEST_USER_ID))
        async with asyncio.timeout(3):
            await waiting.wait()
        if action == "new_input":
            service.cancel(TEST_USER_ID)
        elif action == "reset":
            async with operations.resetting(TEST_USER_ID):
                await states.reset_user(TEST_USER_ID)
        elif action == "close":
            await service.close()
        else:
            states.get(TEST_USER_ID).history.append(
                ConversationTurn("Другая тема", "Ответ")
            )
            release.set()
        await asyncio.gather(*service._workers, return_exceptions=True)
        bot.send_message.assert_not_awaited()
        await service.close()

    asyncio.run(run())


def test_input_received_during_main_answer_prevents_scheduling(tmp_path: Path) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        model.chat.return_value = ANSWER
        service = ReplySupplements(
            cast(Bot, Mock()),
            model,
            engine._user_states,
            InputOperations(),
            ReplySupplementLimits(tmp_path),
            style_prompt="persona",
        )
        engine.supplements = service

        async def deliver(text: str) -> None:
            # Следующее сообщение уже принято, пока заканчивается доставка первого.
            service.cancel(TEST_USER_ID)

        await engine.respond_and_deliver(TEST_USER_ID, TURN.user_message, deliver)
        assert not service._workers
        model.draft_supplement.assert_not_awaited()
        await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("route", ["direct_api", "failed_delivery", "rp", "group"])
def test_only_successfully_delivered_private_regular_reply_is_scheduled(
    tmp_path: Path, route: str
) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        model.chat.return_value = ANSWER
        service = Mock(spec=ReplySupplements)
        engine.supplements = service
        deliver = AsyncMock()
        if route == "failed_delivery":
            deliver.side_effect = RuntimeError("delivery failed")
            with pytest.raises(RuntimeError):
                await engine.respond_and_deliver(
                    TEST_USER_ID, TURN.user_message, deliver
                )
        elif route == "direct_api":
            await engine.respond(TEST_USER_ID, TURN.user_message)
        elif route == "group":
            await engine.respond_and_deliver(
                TEST_USER_ID, TURN.user_message, deliver, chat_id=-100
            )
        else:
            engine._user_states.get(TEST_USER_ID).roleplay_active = True
            await engine.respond_and_deliver(TEST_USER_ID, "*машу рукой*", deliver)
        service.schedule.assert_not_called()

    asyncio.run(run())
