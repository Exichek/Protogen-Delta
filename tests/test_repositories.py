"""Тесты JSON-репозиториев приложения."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import sleep
from typing import Any

import pytest

from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.json_file import JsonFileRepository
from protogen_delta.repositories.users import UsersRepository


def test_json_repository_creates_default_file(tmp_path: Path) -> None:
    """При отсутствии файла репозиторий должен создать его с данными по умолчанию."""
    path = tmp_path / "data.json"

    repository = JsonFileRepository(
        path=path,
        default_data={"VALUE": []},
    )

    result = repository.load()

    assert result == {"VALUE": []}
    assert path.exists()

    saved_data = json.loads(path.read_text(encoding="utf-8"))

    assert saved_data == {"VALUE": []}


def test_json_repository_saves_and_loads_data(tmp_path: Path) -> None:
    """Сохранённые JSON-данные должны корректно загружаться."""
    path = tmp_path / "data.json"

    repository = JsonFileRepository(
        path=path,
        default_data={},
    )

    repository.save(
        {
            "name": "Дельта",
            "count": 5,
        }
    )

    assert repository.load() == {
        "name": "Дельта",
        "count": 5,
    }


def test_json_repository_rejects_non_object_root(tmp_path: Path) -> None:
    """Корневой элемент JSON должен быть объектом."""
    path = tmp_path / "data.json"

    path.write_text(
        '["one", "two"]',
        encoding="utf-8",
    )

    repository = JsonFileRepository(
        path=path,
        default_data={},
    )

    with pytest.raises(
        TypeError,
        match="Корневой элемент JSON-файла",
    ):
        repository.load()


def test_users_repository_adds_user(tmp_path: Path) -> None:
    """Новый пользователь должен добавляться в хранилище."""
    repository = UsersRepository(tmp_path)

    result = repository.add(123)

    assert result is True
    assert repository.get_all() == [123]
    assert repository.count() == 1


def test_users_repository_does_not_add_duplicate(tmp_path: Path) -> None:
    """Один Telegram ID нельзя добавить дважды."""
    repository = UsersRepository(tmp_path)

    repository.add(123)

    result = repository.add(123)

    assert result is False
    assert repository.get_all() == [123]
    assert repository.count() == 1


def test_users_repository_rejects_invalid_data(tmp_path: Path) -> None:
    """USERS должен содержать только целые Telegram ID."""
    path = tmp_path / "users.json"

    path.write_text(
        '{"USERS": [123, "456"]}',
        encoding="utf-8",
    )

    repository = UsersRepository(tmp_path)

    with pytest.raises(
        TypeError,
        match="USERS",
    ):
        repository.get_all()


def test_images_repository_adds_image(tmp_path: Path) -> None:
    """Новый file_id изображения должен сохраняться."""
    repository = ImagesRepository(tmp_path)

    result = repository.add("file-id-1")

    assert result is True
    assert repository.get_all() == ["file-id-1"]
    assert repository.count() == 1


def test_images_repository_does_not_add_duplicate(
    tmp_path: Path,
) -> None:
    """Одинаковый file_id нельзя сохранить дважды."""
    repository = ImagesRepository(tmp_path)

    repository.add("file-id-1")

    result = repository.add("file-id-1")

    assert result is False
    assert repository.get_all() == ["file-id-1"]


def test_images_repository_removes_image(tmp_path: Path) -> None:
    """Существующее изображение должно удаляться."""
    repository = ImagesRepository(tmp_path)

    repository.add("file-id-1")
    repository.add("file-id-2")

    result = repository.remove("file-id-1")

    assert result is True
    assert repository.get_all() == ["file-id-2"]


def test_images_repository_returns_false_for_missing_image(
    tmp_path: Path,
) -> None:
    """Удаление отсутствующего изображения должно вернуть False."""
    repository = ImagesRepository(tmp_path)

    result = repository.remove("missing-id")

    assert result is False
    assert repository.get_all() == []


def test_images_repository_rejects_invalid_data(
    tmp_path: Path,
) -> None:
    """IMAGES должен содержать только строковые file_id."""
    path = tmp_path / "images.json"

    path.write_text(
        '{"IMAGES": ["file-id", 123]}',
        encoding="utf-8",
    )

    repository = ImagesRepository(tmp_path)

    with pytest.raises(
        TypeError,
        match="IMAGES",
    ):
        repository.get_all()


def test_json_repository_does_not_leave_temporary_file(
    tmp_path: Path,
) -> None:
    """После успешного сохранения временный JSON-файл не должен оставаться."""
    path = tmp_path / "data.json"
    temp_path = tmp_path / "data.json.tmp"

    repository = JsonFileRepository(
        path=path,
        default_data={},
    )

    repository.save(
        {
            "name": "Дельта",
        }
    )

    assert path.exists()
    assert not temp_path.exists()
    assert repository.load() == {
        "name": "Дельта",
    }


def test_json_repository_skips_save_when_update_returns_false(
    tmp_path: Path,
) -> None:
    """Изменения не должны сохраняться, если updater вернул False."""
    path = tmp_path / "data.json"

    repository = JsonFileRepository(
        path=path,
        default_data={"VALUE": 1},
    )

    repository.save({"VALUE": 1})

    def updater(data: dict[str, Any]) -> bool:
        data["VALUE"] = 2
        return False

    result = repository.update(updater)

    assert result is False
    assert repository.load() == {"VALUE": 1}


def test_json_repository_serializes_concurrent_updates(
    tmp_path: Path,
) -> None:
    """Параллельные изменения не должны терять данные."""
    path = tmp_path / "counter.json"

    repository = JsonFileRepository(
        path=path,
        default_data={"COUNT": 0},
    )

    repository.load()

    workers = 8
    barrier = Barrier(workers)

    def increment() -> bool:
        barrier.wait()

        def updater(data: dict[str, Any]) -> bool:
            current = data["COUNT"]

            if not isinstance(current, int):
                raise TypeError("COUNT должен быть целым числом")

            sleep(0.01)

            data["COUNT"] = current + 1
            return True

        return repository.update(updater)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(increment) for _ in range(workers)]

        results = [future.result() for future in futures]

    assert all(results)
    assert repository.load() == {"COUNT": workers}


def test_users_repository_removes_user(tmp_path: Path) -> None:
    """Сохранённый пользователь должен удаляться из хранилища."""
    repository = UsersRepository(tmp_path)

    repository.add(111)
    repository.add(222)

    result = repository.remove(111)

    assert result is True
    assert repository.get_all() == [222]
    assert repository.count() == 1


def test_users_repository_returns_false_for_missing_user(
    tmp_path: Path,
) -> None:
    """Удаление неизвестного пользователя должно вернуть False."""
    repository = UsersRepository(tmp_path)

    repository.add(111)

    result = repository.remove(222)

    assert result is False
    assert repository.get_all() == [111]
