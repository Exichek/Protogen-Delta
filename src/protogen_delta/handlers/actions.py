"""Калькулятор и личные напоминания без запроса к модели."""

import re
from collections.abc import Callable
from time import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.calculation import calculate
from protogen_delta.core.message_utils import split_message
from protogen_delta.core.reminder_request import (
    REMINDER_HELP,
    is_reminder_message,
    parse_reminder,
    reminder_payload,
    reminder_time,
)
from protogen_delta.services.reminders import ReminderService


def create_actions_router(
    reminders: ReminderService,
    *,
    clock: Callable[[], float] = time,
) -> Router:
    router = Router(name=__name__)

    @router.message(Command("calc"))
    async def calculator(message: Message) -> None:
        if message.forward_origin is not None:
            await message.answer(
                "Пришли выражение своей командой /calc", parse_mode=None
            )
            return
        expression = re.sub(
            r"^/calc(?:@\w+)?(?:\s+|$)", "", message.text or "", flags=re.I
        )
        if not expression:
            await message.answer(
                "Калькулятор: /calc (25+17)*3\nДоступны + − × ÷ // % ^ и скобки.",
                parse_mode=None,
            )
            return
        try:
            result = calculate(expression)
        except ValueError as error:
            await message.answer(str(error), parse_mode=None)
        else:
            await message.answer(f"{expression}\n= {result}", parse_mode=None)

    async def handle_reminder(message: Message) -> None:
        if message.from_user is None or message.from_user.is_bot:
            return
        if message.chat.type != "private" or message.chat.id != message.from_user.id:
            await message.answer(
                "Личные напоминания доступны в личном чате со мной.", parse_mode=None
            )
            return
        if message.forward_origin is not None:
            await message.answer(
                "Чтобы создать напоминание, напиши собственную просьбу.\n"
                + REMINDER_HELP,
                parse_mode=None,
            )
            return
        value = (message.text or "").strip()
        command = re.match(
            r"^/(remind|reminders)(?:@\w+)?(?:\s+(.*))?$", value, re.I | re.S
        )
        payload = command[2] or "" if command else reminder_payload(value)
        listed = command and command[1].casefold() == "reminders" and not payload
        listed = listed or bool(
            re.fullmatch(
                r"(?:покажи|список)\s+(?:мои\s+)?напоминани[яй][.!?]*", payload, re.I
            )
        )
        cancelled = re.fullmatch(
            r"(?:cancel|отмени|удали)(?:\s+напоминание)?\s+#?(\d{1,10})[.!]?",
            payload,
            re.I,
        )
        try:
            if listed:
                jobs = await reminders.list_user(message.from_user.id)
                text = (
                    "Активных напоминаний нет."
                    if not jobs
                    else "Твои напоминания:\n"
                    + "\n".join(
                        f"#{job.id} — {reminder_time(job.due_at)}: {job.text}"
                        + (" — не удалось доставить" if job.status == "failed" else "")
                        for job in jobs
                    )
                )
                for chunk in split_message(text):
                    await message.answer(chunk, parse_mode=None)
            elif cancelled:
                removed = await reminders.cancel(
                    message.from_user.id, int(cancelled[1])
                )
                await message.answer(
                    (
                        "Напоминание отменено."
                        if removed
                        else "Такого активного напоминания у тебя нет: оно могло быть отправлено или отменено."
                    ),
                    parse_mode=None,
                )
            elif command and command[1].casefold() == "reminders":
                await message.answer(
                    "Отмена: /reminders cancel 12\nСписок: /reminders", parse_mode=None
                )
            else:
                request = parse_reminder(payload, clock())
                job = await reminders.create(
                    message.from_user.id,
                    message.chat.id,
                    message.message_id,
                    message.message_thread_id,
                    request,
                )
                if job.status in {"pending", "sending"}:
                    await message.answer(
                        f"Напоминание #{job.id} сохранено: {reminder_time(job.due_at)}.\n{job.text}",
                        parse_mode=None,
                    )
                else:
                    await message.answer(
                        "Эта просьба уже обработана. Новое напоминание можно создать отдельным сообщением.",
                        parse_mode=None,
                    )
        except ValueError as error:
            await message.answer(str(error), parse_mode=None)
        except Exception:
            await message.answer(
                "Напоминания сейчас недоступны. Не могу подтвердить сохранение — проверь /reminders позже.",
                parse_mode=None,
            )

    @router.message(Command("remind", "reminders"))
    async def reminder_command(message: Message) -> None:
        await handle_reminder(message)

    @router.message(F.text.func(is_reminder_message))
    async def natural_reminder(message: Message) -> None:
        await handle_reminder(message)

    return router
