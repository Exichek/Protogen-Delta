"""Онлайн-копия читает WAL, обнаруживает повреждения и сохраняет все базы."""

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import pytest

import protogen_delta.backup_snapshot as module
from protogen_delta.backup_snapshot import (
    create_online_snapshot,
    pack_snapshot,
    restore_archive,
    restore_snapshot,
    verify_snapshot,
)


def source_data(folder: Path) -> Path:
    source = folder / "source"
    source.mkdir()
    for name, table in [
        ("user_states.db", "age_restrictions"),
        ("delivery_ledger.db", "ledger"),
    ]:
        with closing(sqlite3.connect(source / name)) as db, db:
            db.execute(f"CREATE TABLE {table} (user_id INTEGER PRIMARY KEY)")
            db.execute(f"INSERT INTO {table} VALUES (7)")
    (source / "users.json").write_text('{"USERS":[7]}', "utf-8")
    (source / "stickers.json").write_text('{"STICKERS":[]}', "utf-8")
    (source / ".env").write_text("SECRET=never-copy", "utf-8")
    (source / "backups").mkdir()
    return source


def test_online_snapshot_reads_committed_wal_without_stopping_writer(
    tmp_path: Path,
) -> None:
    source = source_data(tmp_path)
    with closing(
        sqlite3.connect(source / "user_states.db", isolation_level=None)
    ) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO age_restrictions VALUES (8)")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO age_restrictions VALUES (9)")
        result = create_online_snapshot(source, tmp_path / "backup")
        writer.rollback()
    assert result["files"] == 4 and result["databases"] == 2
    assert not (tmp_path / "backup/.env").exists()
    with closing(sqlite3.connect(tmp_path / "backup/user_states.db")) as db:
        assert db.execute(
            "SELECT user_id FROM age_restrictions ORDER BY user_id"
        ).fetchall() == [(7,), (8,)]
    restored = tmp_path / "restored"
    assert restore_snapshot(tmp_path / "backup", restored) == result
    assert (restored / "delivery_ledger.db").is_file()
    with pytest.raises(FileExistsError):
        restore_snapshot(tmp_path / "backup", restored)
    assert verify_snapshot(restored) == result


def test_zip_round_trip_includes_restriction_and_ledger(tmp_path: Path) -> None:
    result = create_online_snapshot(source_data(tmp_path), tmp_path / "backup")
    digest = pack_snapshot(tmp_path / "backup", tmp_path / "copy.zip")
    assert len(digest) == 64
    assert restore_archive(tmp_path / "copy.zip", tmp_path / "restore") == result
    assert verify_snapshot(tmp_path / "restore") == result
    with pytest.raises(FileExistsError):
        pack_snapshot(tmp_path / "backup", tmp_path / "copy.zip")


@pytest.mark.parametrize(
    "change",
    [
        "sha",
        "extra",
        "version",
        "duplicate",
        "name",
        "missing",
        "bad_size",
        "bad_record",
    ],
)
def test_verification_refuses_incomplete_or_modified_snapshot(
    tmp_path: Path, change: str
) -> None:
    backup = tmp_path / "backup"
    create_online_snapshot(source_data(tmp_path), backup)
    p = backup / module.MANIFEST
    manifest = json.loads(p.read_text("utf-8"))
    if change == "sha":
        (backup / "users.json").write_text('{"USERS":[9]}', "utf-8")
    elif change == "extra":
        (backup / "unknown.json").write_text("{}", "utf-8")
    elif change == "version":
        manifest["version"] = 99
    elif change == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif change == "name":
        manifest["files"][0]["name"] = "../private.db"
    elif change == "missing":
        (backup / manifest["files"][0]["name"]).unlink()
    elif change == "bad_size":
        manifest["files"][0]["bytes"] = -1
    else:
        manifest["files"][0] = "invalid"
    p.write_text(json.dumps(manifest), "utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_snapshot(backup)
    with pytest.raises((ValueError, FileNotFoundError)):
        restore_snapshot(backup, tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


def test_invalid_source_never_publishes_snapshot_or_overwrites_data(
    tmp_path: Path,
) -> None:
    source = source_data(tmp_path)
    (source / "users.json").write_text("not json", "utf-8")
    with pytest.raises(ValueError):
        create_online_snapshot(source, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "important.txt").write_text("keep", "utf-8")
    with pytest.raises(FileExistsError):
        create_online_snapshot(source, existing)
    assert (existing / "important.txt").read_text("utf-8") == "keep"


@pytest.mark.parametrize(
    "name", ["../outside.db", "/absolute.db", "nested/state.db", ".."]
)
def test_zip_never_extracts_paths_outside_restore(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(name, b"bad")
    with pytest.raises(ValueError):
        restore_archive(archive, tmp_path / "restore")
    assert (
        not (tmp_path / "restore").exists() and not (tmp_path / "outside.db").exists()
    )


def test_archive_expansion_limit_is_checked_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_online_snapshot(source_data(tmp_path), tmp_path / "backup")
    pack_snapshot(tmp_path / "backup", tmp_path / "copy.zip")
    monkeypatch.setattr(module, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="ограничения"):
        restore_archive(tmp_path / "copy.zip", tmp_path / "restore")
    assert not (tmp_path / "restore").exists()


@pytest.mark.parametrize(
    "action", ["create", "verify", "restore", "pack", "restore-zip"]
)
def test_cli_actions_are_local_and_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
) -> None:
    source = source_data(tmp_path)
    if action != "create":
        create_online_snapshot(source, tmp_path / "backup")
        source = tmp_path / "backup"
    if action == "restore-zip":
        pack_snapshot(source, tmp_path / "copy.zip")
        source = tmp_path / "copy.zip"
    args = ["snapshot", action, "--source", str(source)]
    if action != "verify":
        args += ["--destination", str(tmp_path / "result")]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert "sha256" in result if action == "pack" else result["files"] == 4


def test_cli_requires_new_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["snapshot", "create", "--source", str(tmp_path)])
    with pytest.raises(SystemExit):
        module.main()
