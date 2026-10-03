"""Административная разметка стикеров для контекстных реакций."""

import re
from collections import Counter
from typing import cast

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.repositories.stickers import StickerEntry, StickerRating
from protogen_delta.repositories.stickers import StickersRepository as Repository

_TAG_PATTERN = re.compile(r"^[a-z0-9_-]{1,32}$")
_HELP = (
    "🎭 Контекстные стикеры:\n\n"
    "Ответь на стикер: /stickers add playful happy\n"
    "Для взрослого: /stickers add horny rp rating:adult\n"
    "Удалить ответом: /stickers remove\n"
    "Проверить выбор: /stickers test playful\n"
    "Статистика: /stickers list\n\n"
    "Основные теги: sweet, playful, angry, horny, neutral, rp. "
    "Теги хранятся локально и не расходуют токены."
)


def _arguments(text: str | None) -> list[str]:
    """Вернуть аргументы после имени команды."""
    return (text or "").strip().split()[1:]


def _parse_tags(values: list[str]) -> tuple[tuple[str, ...], StickerRating] | None:
    """Разобрать теги и необязательный rating."""
    rating: StickerRating = "safe"
    tags: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized.startswith("rating:"):
            raw_rating = normalized.partition(":")[2]
            if raw_rating not in {"safe", "adult"}:
                return None
            rating = cast(StickerRating, raw_rating)
        elif not _TAG_PATTERN.fullmatch(normalized):
            return None
        else:
            tags.append(normalized)
    unique = tuple(dict.fromkeys(tags))
    return (unique, rating) if unique else None


def create_sticker_admin_router(
    repository: Repository,
    admin_ids: frozenset[int],
) -> Router:
    """Создать закрытый роутер управления стикерами Дельты."""
    router = Router(name=__name__)

    @router.message(Command("stickers"))
    async def stickers(message: Message) -> None:
        """Добавить, обновить, удалить или проверить стикер."""
        if message.from_user is None or message.from_user.id not in admin_ids:
            await message.answer("⛔ У тебя нет доступа к этой команде.")
            return

        arguments = _arguments(message.text)
        if not arguments or arguments[0] in {"help", "помощь"}:
            await message.answer(_HELP)
            return

        action = arguments[0].lower()
        if action == "list":
            entries = repository.get_all()
            if not entries:
                await message.answer("Стикеры ещё не размечены.\n\n" + _HELP)
                return
            tag_counts = Counter(tag for entry in entries for tag in entry.tags)
            adult = sum(entry.rating == "adult" for entry in entries)
            summary = ", ".join(
                f"{tag}: {count}" for tag, count in tag_counts.most_common()
            )
            await message.answer(
                f"🎭 Размечено: {len(entries)}\n"
                f"Safe: {len(entries) - adult}, adult: {adult}\n"
                f"Теги: {summary}"
            )
            return

        replied_sticker = (
            message.reply_to_message.sticker
            if message.reply_to_message is not None
            else None
        )
        if action == "add":
            parsed = _parse_tags(arguments[1:])
            if replied_sticker is None or parsed is None:
                await message.answer(
                    "Ответь этой командой на стикер и укажи английские теги.\n"
                    "Пример: /stickers add playful happy"
                )
                return
            tags, rating = parsed
            created = repository.upsert(
                StickerEntry(
                    file_id=replied_sticker.file_id,
                    file_unique_id=replied_sticker.file_unique_id,
                    tags=tags,
                    rating=rating,
                    emoji=replied_sticker.emoji,
                    set_name=replied_sticker.set_name,
                )
            )
            verb = "Добавил" if created else "Обновил"
            await message.answer(
                f"✅ {verb} стикер: {', '.join(tags)}; рейтинг {rating}."
            )
            return

        if action == "remove":
            unique_id = (
                replied_sticker.file_unique_id
                if replied_sticker is not None
                else arguments[1] if len(arguments) > 1 else ""
            )
            removed = bool(unique_id) and repository.remove(unique_id)
            await message.answer(
                "✅ Стикер удалён из реакций."
                if removed
                else "Такого стикера в разметке нет."
            )
            return

        if action == "test" and len(arguments) > 1:
            tag = arguments[1].lower()
            entry = next(
                (item for item in repository.get_all() if tag in item.tags),
                None,
            )
            if entry is None:
                await message.answer(f"Для тега {tag!r} ничего не найдено.")
            else:
                await message.answer_sticker(entry.file_id)
            return

        await message.answer(_HELP)

    return router
