"""Репозиторий изображений бота."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from protogen_delta.repositories.json_file import JsonFileRepository

_IMAGES_KEY = "IMAGES"
MediaKind = Literal["photo", "document", "video", "animation"]


def _get_images(data: dict[str, Any]) -> list[str]:
    """Получить и проверить список изображений из данных хранилища."""
    images = data.get(_IMAGES_KEY, [])

    if not isinstance(images, list) or not all(
        isinstance(file_id, str) for file_id in images
    ):
        raise TypeError("Поле IMAGES должно содержать список строк")

    return cast(list[str], images)


class ImagesRepository:
    """Хранилище Telegram file_id сохранённых изображений."""

    def __init__(self, data_dir: Path) -> None:
        """Инициализировать хранилище изображений."""
        self._storage = JsonFileRepository(
            path=data_dir / "images.json",
            default_data={_IMAGES_KEY: []},
        )

    def get_all(self) -> list[str]:
        """Вернуть file_id всех сохранённых изображений."""
        return _get_images(self._storage.load())

    def add(
        self,
        file_id: str,
        *,
        kind: MediaKind = "photo",
        file_unique_id: str | None = None,
    ) -> bool:
        """Добавить изображение и вернуть True, если его ещё не было."""
        return bool(self._add([file_id], kind, file_unique_id))

    def add_many(self, file_ids: Iterable[str], *, kind: MediaKind = "photo") -> int:
        """Добавить набор ID одной атомарной записью и вернуть число новых."""
        return self._add(list(dict.fromkeys(file_ids)), kind, None)

    def _add(
        self,
        file_ids: list[str],
        kind: MediaKind,
        file_unique_id: str | None,
    ) -> int:
        added = 0

        def add_image(data: dict[str, Any]) -> bool:
            """Добавить изображение внутри атомарной операции."""
            nonlocal added
            images = _get_images(data)
            existing = set(images)
            unique_ids = data.get("UNIQUE_IDS", {})
            changed = False
            for file_id in file_ids:
                if file_id in existing:
                    # Старой записи можно добавить идентификатор без смены кода.
                    if file_unique_id and unique_ids.get(file_id) != file_unique_id:
                        data.setdefault("UNIQUE_IDS", {})[file_id] = file_unique_id
                        changed = True
                    continue
                if file_unique_id and file_unique_id in unique_ids.values():
                    continue
                images.append(file_id)
                existing.add(file_id)
                if kind != "photo":
                    data.setdefault("KINDS", {})[file_id] = kind
                if file_unique_id:
                    data.setdefault("UNIQUE_IDS", {})[file_id] = file_unique_id
                added += 1
                changed = True
            return changed

        self._storage.update(add_image)
        return added

    def get_kind(self, file_id: str) -> MediaKind:
        kinds = self._storage.load().get("KINDS", {})
        kind = kinds.get(file_id)
        return (
            cast(MediaKind, kind)
            if kind in {"document", "video", "animation"}
            else "photo"
        )

    def remove(self, file_id: str) -> bool:
        """Удалить изображение и вернуть True, если оно существовало."""

        def remove_image(data: dict[str, Any]) -> bool:
            """Удалить изображение внутри атомарной операции."""
            images = _get_images(data)

            if file_id not in images:
                return False

            images.remove(file_id)
            data.get("KINDS", {}).pop(file_id, None)
            data.get("UNIQUE_IDS", {}).pop(file_id, None)
            return True

        return self._storage.update(remove_image)

    def count(self) -> int:
        """Вернуть количество сохранённых изображений."""
        return len(self.get_all())
