"""Репозиторий пользователей бота."""

from pathlib import Path
from typing import Any, cast

from protogen_delta.repositories.json_file import JsonFileRepository

_USERS_KEY = "USERS"


def _get_users(data: dict[str, Any]) -> list[int]:
    """Получить и проверить список пользователей из данных хранилища."""
    users = data.get(_USERS_KEY, [])

    if not isinstance(users, list) or not all(
        isinstance(user_id, int) for user_id in users
    ):
        raise TypeError("Поле USERS должно содержать список целых чисел")

    return cast(list[int], users)


class UsersRepository:
    """Хранилище идентификаторов пользователей Telegram."""

    def __init__(self, data_dir: Path) -> None:
        """Инициализировать хранилище пользователей."""
        self._storage = JsonFileRepository(
            path=data_dir / "users.json",
            default_data={_USERS_KEY: []},
        )

    def get_all(self) -> list[int]:
        """Вернуть идентификаторы всех сохранённых пользователей."""
        return _get_users(self._storage.load())

    def add(self, user_id: int) -> bool:
        """Добавить пользователя и вернуть True, если его ещё не было."""

        def add_user(data: dict[str, Any]) -> bool:
            """Добавить пользователя внутри атомарной операции."""
            users = _get_users(data)

            if user_id in users:
                return False

            users.append(user_id)
            return True

        return self._storage.update(add_user)

    def remove(self, user_id: int) -> bool:
        """Удалить пользователя и вернуть True, если он был сохранён."""

        def remove_user(data: dict[str, Any]) -> bool:
            """Удалить пользователя внутри атомарной операции."""
            users = _get_users(data)

            if user_id not in users:
                return False

            users.remove(user_id)
            return True

        return self._storage.update(remove_user)

    def count(self) -> int:
        """Вернуть количество сохранённых пользователей."""
        return len(self.get_all())
