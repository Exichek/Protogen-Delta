"""Импорт проверенного пака по постоянным Telegram ID, без вызовов LLM."""

import json
from dataclasses import dataclass, replace
from importlib.resources import files

from aiogram import Bot

from protogen_delta.repositories.stickers import (
    StickersRepository,
    parse_sticker_entries,
)

DEFAULT_STICKER_SET = "delta_sticksss"


@dataclass(frozen=True, slots=True)
class StickerPackImport:
    """Результат синхронизации без автоматической разметки новых картинок."""

    matched: int
    added: int
    unknown: int


class StickerPackImporter:
    """Обновлять Telegram file_id, сохраняя локальные смысловые теги."""

    def __init__(self, bot: Bot, repository: StickersRepository) -> None:
        self._bot = bot
        self._repository = repository

    async def sync(self) -> StickerPackImport:
        """Сопоставить текущий пак с проверенной разметкой по file_unique_id."""
        raw = json.loads(
            files("protogen_delta.config")
            .joinpath("data/delta_stickers.json")
            .read_text(encoding="utf-8")
        )
        definitions = {item.file_unique_id: item for item in parse_sticker_entries(raw)}
        pack = await self._bot.get_sticker_set(DEFAULT_STICKER_SET)
        entries = [
            replace(
                definitions[sticker.file_unique_id],
                file_id=sticker.file_id,
                emoji=sticker.emoji,
                set_name=pack.name,
            )
            for sticker in pack.stickers
            if sticker.file_unique_id in definitions
        ]
        added = self._repository.import_entries(entries)
        return StickerPackImport(len(entries), added, len(pack.stickers) - len(entries))
