"""Одно необязательное дополнение к доставленному ответу через короткую паузу."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from contextvars import Context
from dataclasses import replace
from difflib import SequenceMatcher
from random import uniform
from time import monotonic

from aiogram import Bot

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.input_operations import InputOperations
from protogen_delta.core.llm_budget import llm_user
from protogen_delta.core.operation_metrics import OperationMetrics, OperationSnapshot
from protogen_delta.core.reply_formatting import format_reply_text
from protogen_delta.core.shared_dialogue import SharedDialogueAccounts
from protogen_delta.core.tool_markup import contains_unquoted_tool_markup
from protogen_delta.core.user_state import ConversationTurn, UserStateStore
from protogen_delta.repositories.reply_supplements import ReplySupplementLimits
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.turn_contract import turn_contract_context
from protogen_delta.services.web_verification import UNVERIFIED_WEB_REPLY

logger = logging.getLogger(__name__)


def supplement_eligible(turn: ConversationTurn) -> bool:
    """Не добавлять вторую реплику к краткой просьбе, ошибке или приветствию."""
    return (
        not turn.proactive
        and len(turn.assistant_message) >= 160
        and len(turn.user_message) >= 12
        and UNVERIFIED_WEB_REPLY not in turn.assistant_message
        and not turn_contract_context([turn.user_message])
        and not re.search(
            r"без\s+(?:дополнени\w*|продолжени\w*)|не\s+дополняй",
            turn.user_message,
            re.I,
        )
        and bool(
            re.search(
                r"\?|\b(?:расскажи|объясни|почему|посоветуй|как|что)\b",
                turn.user_message,
                re.I,
            )
        )
    )


def supplement_text(raw: str, previous: str) -> str:
    try:
        value = json.loads(raw)
    except ValueError, TypeError:
        return ""
    text = value.get("text") if isinstance(value, dict) else None
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 350:
        return ""
    text = text.strip()
    if (
        "\n" in text
        or re.search(r"https?://|\d|```", text)
        or contains_unquoted_tool_markup(text)
        or text.casefold() in previous.casefold()
        or SequenceMatcher(None, text.casefold(), previous.casefold()).ratio() > 0.65
        or any(
            SequenceMatcher(None, text.casefold(), part.casefold()).ratio() > 0.75
            for part in re.split(r"(?<=[.!?])\s+|\n+", previous)
            if part.strip()
        )
    ):
        return ""
    return text


class ReplySupplements:
    def __init__(
        self,
        bot: Bot,
        model: DeepSeekService,
        states: UserStateStore,
        operations: InputOperations,
        limits: ReplySupplementLimits,
        *,
        style_prompt: str,
        shared: SharedDialogueAccounts | None = None,
        delay: Callable[[], float] = lambda: uniform(5, 15),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._bot, self._model, self._states, self._operations = (
            bot,
            model,
            states,
            operations,
        )
        self._limits, self._shared = limits, shared or SharedDialogueAccounts()
        self._prompt = style_prompt + "\n\n" + load_prompt("reply_supplement")
        self._delay, self._sleep, self._clock = delay, sleep, clock
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._workers: set[asyncio.Task[None]] = set()
        self._activity: dict[int, object] = {}
        self._slots = asyncio.Semaphore(2)
        self._closed = False
        self._metrics = OperationMetrics(capacity=32)

    def snapshot(self) -> OperationSnapshot:
        return self._metrics.snapshot()

    def cancel(self, owner: int) -> object:
        key = self._shared.key(owner)
        token = object()
        self._activity[key] = token
        if len(self._activity) > 512:
            self._activity.pop(next(iter(self._activity)))
        task = self._tasks.get(key)
        if task is not None:
            task.cancel()
        return token

    def schedule(self, owner: int, turn: ConversationTurn, activity: object) -> None:
        key = self._shared.key(owner)
        if self._activity.get(key) is not activity:
            return
        if self._closed or len(self._workers) >= 32 or not supplement_eligible(turn):
            return
        # Не наследовать завершённую доставку, архивный билет, блокировку RP
        # или владельца предыдущего запроса из ContextVar.
        task = asyncio.create_task(self._run(owner, turn, activity), context=Context())
        self._tasks[key] = task
        self._workers.add(task)
        task.add_done_callback(lambda done: self._done(key, done))

    def _done(self, key: int, task: asyncio.Task[None]) -> None:
        self._workers.discard(task)
        if self._tasks.get(key) is task:
            self._tasks.pop(key, None)
        if not task.cancelled():
            task.exception()

    async def _run(
        self, owner: int, expected: ConversationTurn, activity: object
    ) -> None:
        with self._metrics.measure():
            await self._generate(owner, expected, activity)

    async def _generate(
        self, owner: int, expected: ConversationTurn, activity: object
    ) -> None:
        started = self._clock()
        delay = min(15.0, max(5.0, self._delay()))
        try:
            async with self._operations.receive(owner) as ticket:
                if not self._operations.start_memory(ticket):
                    return
                if self._activity.get(self._shared.key(owner)) is not activity:
                    return
                async with self._states.use_conversation(owner) as state:
                    if (
                        not state.history
                        or state.history[-1] is not expected
                        or state.roleplay_active
                        or self._clock() - started > 15
                    ):
                        return
                if not await self._limits.reserve(owner):
                    return
                async with asyncio.timeout(12), self._slots:
                    with llm_user(owner):
                        raw = await self._model.draft_supplement(
                            self._prompt,
                            json.dumps(
                                {
                                    "question": expected.user_message[:4000],
                                    "sent_answer": expected.assistant_message[:10000],
                                },
                                ensure_ascii=False,
                            ),
                        )
                text = supplement_text(raw, expected.assistant_message)
                if not text or self._clock() - started > 15:
                    return
                await self._sleep(max(0, delay - (self._clock() - started)))
                async with self._states.use_conversation(owner) as state:
                    if (
                        not self._operations.is_current(ticket)
                        or not state.history
                        or self._activity.get(self._shared.key(owner)) is not activity
                        or state.history[-1] is not expected
                        or state.roleplay_active
                        or self._clock() - started > 15
                    ):
                        return
                    visible, entities = format_reply_text(text)
                    # Ровно одна попытка доставки. Сетевой таймаут не повторяем.
                    await self._bot.send_message(
                        owner, visible, entities=entities, parse_mode=None
                    )
                    state.history[-1] = replace(
                        expected,
                        assistant_message=expected.assistant_message + "\n\n" + text,
                    )
                    self._states.mark_history_updated(state)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info("Дополнение пропущено: генерация или доставка не завершена")

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._workers)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
