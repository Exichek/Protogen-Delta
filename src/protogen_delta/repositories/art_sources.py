"""Сохраняемый список групп для загрузки артов."""

from pathlib import Path
from typing import Any

from protogen_delta.repositories.json_file import JsonFileRepository


class ArtSourcesRepository:
    def __init__(self, data_dir: Path, initial_chat_id: int) -> None:
        self._storage = JsonFileRepository(
            data_dir / "art_sources.json", {"CHATS": [initial_chat_id]}
        )

    def get_all(self) -> list[int]:
        values = self._storage.load().get("CHATS")
        if not isinstance(values, list) or not all(type(x) is int for x in values):
            raise ValueError("Некорректный список групп артов")
        return values

    def change(self, chat_id: int, *, add: bool) -> bool:
        if chat_id >= 0:
            raise ValueError("Нужен отрицательный ID группы")

        def update(data: dict[str, Any]) -> bool:
            chats = data["CHATS"]
            if add and chat_id not in chats:
                chats.append(chat_id)
                return True
            if not add and chat_id in chats:
                chats.remove(chat_id)
                return True
            return False

        return self._storage.update(update)
