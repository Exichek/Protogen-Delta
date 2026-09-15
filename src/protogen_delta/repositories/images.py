"""Репозиторий изображений бота."""

from pathlib import Path
from typing import Any, cast

from protogen_delta.repositories.json_file import JsonFileRepository

_IMAGES_KEY = "IMAGES"


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

    def add(self, file_id: str) -> bool:
        """Добавить изображение и вернуть True, если его ещё не было."""

        def add_image(data: dict[str, Any]) -> bool:
            """Добавить изображение внутри атомарной операции."""
            images = _get_images(data)

            if file_id in images:
                return False

            images.append(file_id)
            return True

        return self._storage.update(add_image)

    def remove(self, file_id: str) -> bool:
        """Удалить изображение и вернуть True, если оно существовало."""

        def remove_image(data: dict[str, Any]) -> bool:
            """Удалить изображение внутри атомарной операции."""
            images = _get_images(data)

            if file_id not in images:
                return False

            images.remove(file_id)
            return True

        return self._storage.update(remove_image)

    def count(self) -> int:
        """Вернуть количество сохранённых изображений."""
        return len(self.get_all())
