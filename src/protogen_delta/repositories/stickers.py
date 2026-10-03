"""Хранилище размеченных стикеров Дельты."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from protogen_delta.repositories.json_file import JsonFileRepository

StickerRating = Literal["safe", "adult"]


@dataclass(frozen=True, slots=True)
class StickerEntry:
    """Один Telegram-стикер и его локальная смысловая разметка."""

    file_id: str
    file_unique_id: str
    tags: tuple[str, ...]
    rating: StickerRating = "safe"
    emoji: str | None = None
    set_name: str | None = None


def _entries(data: dict[str, Any]) -> list[StickerEntry]:
    """Проверить JSON и вернуть типизированные записи."""
    raw_entries = data.get("STICKERS", [])
    if not isinstance(raw_entries, list):
        raise TypeError("Поле STICKERS должно содержать список")

    entries: list[StickerEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise TypeError("Запись STICKERS должна быть объектом")
        file_id = raw.get("file_id")
        unique_id = raw.get("file_unique_id")
        tags = raw.get("tags")
        rating = raw.get("rating", "safe")
        emoji = raw.get("emoji")
        set_name = raw.get("set_name")
        if (
            not isinstance(file_id, str)
            or not file_id
            or not isinstance(unique_id, str)
            or not unique_id
            or not isinstance(tags, list)
            or not tags
            or not all(isinstance(tag, str) and tag for tag in tags)
            or rating not in {"safe", "adult"}
            or emoji is not None
            and not isinstance(emoji, str)
            or set_name is not None
            and not isinstance(set_name, str)
        ):
            raise TypeError("Некорректная запись STICKERS")
        entries.append(
            StickerEntry(
                file_id=file_id,
                file_unique_id=unique_id,
                tags=tuple(dict.fromkeys(tags)),
                rating=rating,
                emoji=emoji,
                set_name=set_name,
            )
        )
    return entries


class StickersRepository:
    """Сохранять file_id и смысловые теги стикеров между перезапусками."""

    def __init__(self, data_dir: Path) -> None:
        self._storage = JsonFileRepository(
            data_dir / "stickers.json",
            {"STICKERS": []},
        )

    def get_all(self) -> list[StickerEntry]:
        """Вернуть все размеченные стикеры."""
        return _entries(self._storage.load())

    def upsert(self, entry: StickerEntry) -> bool:
        """Добавить или обновить запись; вернуть True при создании."""
        created = True

        def update(data: dict[str, Any]) -> bool:
            nonlocal created
            entries = _entries(data)
            for index, existing in enumerate(entries):
                if existing.file_unique_id == entry.file_unique_id:
                    created = False
                    if existing == entry:
                        return False
                    entries[index] = entry
                    data["STICKERS"] = [asdict(item) for item in entries]
                    return True
            entries.append(entry)
            data["STICKERS"] = [asdict(item) for item in entries]
            return True

        self._storage.update(update)
        return created

    def remove(self, file_unique_id: str) -> bool:
        """Удалить стикер по постоянному Telegram ID."""

        def update(data: dict[str, Any]) -> bool:
            entries = _entries(data)
            remaining = [
                item for item in entries if item.file_unique_id != file_unique_id
            ]
            if len(remaining) == len(entries):
                return False
            data["STICKERS"] = [asdict(item) for item in remaining]
            return True

        return self._storage.update(update)
