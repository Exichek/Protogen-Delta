"""Однократный импорт, сохранение удаления и конкурирующие регистрации."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest
from test_maintenance import seed

from protogen_delta.maintenance import copy_snapshot
from protogen_delta.repositories.users import UsersRepository


def test_import_is_once_and_never_rewrites_legacy_file(tmp_path: Path) -> None:
    legacy = tmp_path / "users.json"
    original = b'{"USERS": [7, 42, 7, 3]}'
    legacy.write_bytes(original)
    users = UsersRepository(tmp_path)
    assert users.get_all() == [7, 42, 3]
    assert users.remove(42)
    assert users.add(99)
    assert UsersRepository(tmp_path).get_all() == [7, 3, 99]
    assert legacy.read_bytes() == original
    legacy.write_text("broken old file", encoding="utf-8")
    assert UsersRepository(tmp_path).get_all() == [7, 3, 99]


def test_concurrent_instances_register_exactly_once(tmp_path: Path) -> None:
    repositories = [UsersRepository(tmp_path) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda repo: repo.add(42), repositories))
    assert results.count(True) == 1
    assert repositories[0].get_all() == [42]
    assert repositories[0].count() == 1
    assert not (tmp_path / "users.json").exists()
    with closing(sqlite3.connect(tmp_path / "users.db")) as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_failed_import_can_be_retried_without_partial_users(tmp_path: Path) -> None:
    legacy = tmp_path / "users.json"
    legacy.write_text('{"USERS": [42, true]}', encoding="utf-8")
    repository = UsersRepository(tmp_path)
    with pytest.raises(TypeError, match="USERS"):
        repository.get_all()
    legacy.write_text('{"USERS": [7]}', encoding="utf-8")
    assert repository.get_all() == [7]


@pytest.mark.parametrize("user_id", [True, 0, -1, 2**63])
def test_invalid_user_id_cannot_be_registered(tmp_path: Path, user_id: int) -> None:
    with pytest.raises(ValueError, match="Telegram ID"):
        UsersRepository(tmp_path).add(user_id)


def test_snapshot_restores_current_sqlite_not_stale_legacy_json(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "restored"
    seed(source)
    assert UsersRepository(source).remove(123)
    assert UsersRepository(source).add(7)
    copy_snapshot(source, destination)
    assert UsersRepository(destination).get_all() == [7]
    assert (destination / "users.json").read_bytes() == (
        source / "users.json"
    ).read_bytes()


def test_snapshot_supports_new_installation_without_legacy_file(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "restored"
    seed(source)
    (source / "users.json").unlink()
    copy_snapshot(source, destination)
    assert UsersRepository(destination).get_all() == [123]
