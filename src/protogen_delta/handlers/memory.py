"""Просмотр и удаление собственного постоянного профиля."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.message_utils import split_message
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.user_facts import FACT_LABELS
from protogen_delta.services.memory import MemoryService


def create_memory_router(memory: MemoryService, states: UserStateStore) -> Router:
    router = Router(name=__name__)

    @router.message(Command("memory"))
    async def memory_command(message: Message) -> None:
        if message.from_user is None:
            return
        if message.chat.type != "private":
            await message.answer("Открой /memory в личном чате с Дельтой.")
            return
        user_id = message.from_user.id
        parts = (message.text or "").split()
        async with states.use(user_id) as state:
            if memory.facts is None:
                await message.answer("Профиль памяти сейчас недоступен.")
                return
            if len(parts) == 3 and parts[1] == "forget" and parts[2] in FACT_LABELS:
                result = await memory.forget_facts(user_id, parts[2])
                state.history.clear()
                await message.answer(
                    f"Убрано из профиля: {FACT_LABELS[parts[2]]}. Записей: {result.changed}."
                )
                return
            if len(parts) != 1:
                await message.answer(
                    "Просмотр: /memory\nУдаление поля: /memory forget occupation\n"
                    + "Поля: "
                    + ", ".join(FACT_LABELS)
                    + "\nПолный сброс всей памяти: /reset",
                    parse_mode=None,
                )
                return
            facts = await memory.facts.repository.all(user_id)
            lines = ["🧠 Постоянный профиль:"]
            lines.extend(f"• {FACT_LABELS[f.key]} ({f.key}): {f.value}" for f in facts)
            if not facts:
                lines.append(
                    "Пока пусто. Расскажи о себе: «Меня зовут…», «Я работаю…», «Люблю…»."
                )
            lines.append(
                "Удалить поле: /memory forget <поле>. Полностью забыть тебя: /reset."
            )
            # Профиль до 32 коротких записей; Telegram допускает 4096 символов.
            text = "\n".join(lines)
            for chunk in split_message(text):
                await message.answer(chunk, parse_mode=None)

    return router
