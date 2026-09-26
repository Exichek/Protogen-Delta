"""Проверки восстановления SQLite и JSON в отдельный каталог."""

import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.maintenance import copy_snapshot, main
from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.repositories.users import UsersRepository


def seed(path: Path) -> None:
    UsersRepository(path).add(123)
    ImagesRepository(path).add("test-image")
    states = UserStateStore(persistence=UserStateRepository(path))
    memories = MemoriesRepository(path)

    async def save() -> None:
        await memories.remember(123, "funny", "старый мем", 10.0)
        async with states.use(123) as state:
            state.relationship.trust = 0.7
            state.roleplay_active = True
            state.roleplay_configuration = "female"
            state.roleplay_character = "человек в пальто"

    asyncio.run(save())


def test_backup_restore_and_restart(tmp_path: Path) -> None:
    source, backup, restored = (tmp_path / name for name in ("data", "backup", "new"))
    seed(source)
    ArtSourcesRepository(source, -100).change(-200, add=True)
    copy_snapshot(source, backup)
    UsersRepository(source).remove(123)
    asyncio.run(UserStateRepository(source).delete(123))
    copy_snapshot(backup, restored)
    assert UsersRepository(restored).get_all() == [123]
    assert ImagesRepository(restored).get_all() == ["test-image"]
    assert ArtSourcesRepository(restored, -999).get_all() == [-100, -200]
    assert asyncio.run(MemoriesRepository(restored).recent(123))[0].text == "старый мем"

    async def check() -> None:
        states = UserStateStore(persistence=UserStateRepository(restored))
        async with states.use(123) as state:
            assert state.relationship.trust == 0.7
            assert state.roleplay_active
            assert state.roleplay_configuration == "female"
            assert state.roleplay_character == "человек в пальто"
            assert not state.history
        await states.reset_user(123)
        assert await UserStateRepository(restored).load(123) is None

    asyncio.run(check())


@pytest.mark.parametrize(
    "payload", ['{"USERS": [true]}', '{"USERS": "bad"}', "[]", "{"]
)
def test_invalid_json_never_creates_destination(tmp_path: Path, payload: str) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    seed(source)
    (source / "users.json").write_text(payload, encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        copy_snapshot(source, destination)
    assert not destination.exists()


def test_existing_destination_is_untouched(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    seed(source)
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        copy_snapshot(source, destination)
    assert marker.read_text() == "keep"


def test_corrupt_database_does_not_create_destination(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    seed(source)
    (source / "user_states.db").write_bytes(b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        copy_snapshot(source, destination)
    assert not destination.exists()


def test_failed_copy_removes_only_new_destination(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    seed(source)
    with patch("protogen_delta.maintenance.shutil.copyfile", side_effect=OSError):
        with pytest.raises(OSError):
            copy_snapshot(source, destination)
    assert not destination.exists()
    assert UsersRepository(source).get_all() == [123]


def test_cli_requires_stopped_bot_and_restores(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    seed(source)
    args = [
        "maintenance",
        "restore",
        "--source",
        str(source),
        "--destination",
        str(destination),
    ]
    with patch("sys.argv", args), pytest.raises(SystemExit):
        main()
    assert not destination.exists()
    with patch("sys.argv", args + ["--bot-stopped"]):
        main()
    assert UsersRepository(destination).get_all() == [123]
