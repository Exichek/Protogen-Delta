"""Проверяемая онлайн-копия SQLite/JSON и восстановление в новый каталог."""

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile

MANIFEST = "snapshot-manifest.json"
MAX_ARCHIVE_BYTES = 2 * 1024**3


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024**2), b""):
            digest.update(piece)
    return digest.hexdigest()


def _validate_file(path: Path) -> None:
    if path.suffix == ".db":
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite-копия не прошла проверку целостности")
    elif path.suffix == ".json":
        json.loads(path.read_text("utf-8"))
    else:
        raise ValueError("Неизвестный тип данных резервной копии")


def verify_snapshot(folder: Path) -> dict[str, int]:
    folder = folder.resolve(strict=True)
    manifest_path = folder / MANIFEST
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 2 * 1024**2:
        raise ValueError("Некорректный манифест копии")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Неизвестная версия резервной копии")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 1024:
        raise ValueError("Некорректный список файлов копии")
    names = set()
    total = databases = 0
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Некорректная запись манифеста")
        name = item.get("name")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[\w.-]{1,128}", name)
            or name in names
            or name == MANIFEST
        ):
            raise ValueError("Небезопасное или повторное имя в копии")
        names.add(name)
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Файл копии отсутствует или является ссылкой")
        size = item.get("bytes")
        if (
            type(size) is not int
            or size < 0
            or path.stat().st_size != size
            or file_hash(path) != item.get("sha256")
        ):
            raise ValueError("Размер или SHA-256 файла копии не совпадает")
        _validate_file(path)
        total += size
        databases += int(path.suffix == ".db")
    if {p.name for p in folder.iterdir()} != names | {MANIFEST}:
        raise ValueError("В копии есть незаявленные файлы")
    return {"files": len(entries), "databases": databases, "bytes": total}


def create_online_snapshot(source: Path, destination: Path) -> dict[str, int]:
    """Копировать каждую SQLite согласованно; общей транзакции между базами нет."""
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if not source.is_dir():
        raise ValueError("Источник должен быть каталогом")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Каталог назначения уже существует")
    files = sorted(p for p in source.iterdir() if p.suffix in {".db", ".json"})
    if not files or not (source / "user_states.db").is_file():
        raise ValueError("Нет базы пользовательского состояния")
    if any(p.is_symlink() or not p.is_file() or p.name == MANIFEST for p in files):
        raise ValueError("Источник содержит ссылки или неподдерживаемые файлы")
    destination.parent.mkdir(parents=True, exist_ok=True)
    deadline = monotonic() + 120

    def progress(status: int, remaining: int, pages: int) -> None:
        if monotonic() > deadline:
            raise TimeoutError("Онлайн-копирование превысило 120 секунд")

    with TemporaryDirectory(prefix=".delta-snapshot-", dir=destination.parent) as temp:
        stage = Path(temp)
        records = []
        for path in files:
            target = stage / path.name
            if path.suffix == ".db":
                with (
                    closing(
                        sqlite3.connect(
                            path.as_uri() + "?mode=ro", uri=True, timeout=10
                        )
                    ) as src,
                    closing(sqlite3.connect(target)) as dst,
                ):
                    src.backup(dst, pages=128, progress=progress, sleep=0.01)
                    # Копия переносится одним файлом, без внешних WAL/SHM.
                    dst.execute("PRAGMA journal_mode=DELETE")
            else:
                target.write_bytes(path.read_bytes())
            target.chmod(0o600)
            _validate_file(target)
            records.append(
                {
                    "name": path.name,
                    "bytes": target.stat().st_size,
                    "sha256": file_hash(target),
                }
            )
        manifest = {
            "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "consistency": "each_sqlite_database_online",
            "files": records,
        }
        (stage / MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
        (stage / MANIFEST).chmod(0o600)
        verify_snapshot(stage)
        destination.mkdir(mode=0o700)
        for path in stage.iterdir():
            shutil.copyfile(path, destination / path.name)
            (destination / path.name).chmod(0o600)
    return verify_snapshot(destination)


def restore_snapshot(source: Path, destination: Path) -> dict[str, int]:
    """Восстановить все перечисленные файлы, не подменяя существующие данные."""
    verify_snapshot(source)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Каталог назначения уже существует")
    destination.mkdir(mode=0o700, parents=True)
    for path in source.iterdir():
        shutil.copyfile(path, destination / path.name)
        (destination / path.name).chmod(0o600)
    return verify_snapshot(destination)


def pack_snapshot(source: Path, archive: Path) -> str:
    if verify_snapshot(source)["bytes"] > MAX_ARCHIVE_BYTES:
        raise ValueError("Копия превышает предел ZIP-восстановления 2 ГБ")
    with (
        archive.open("xb") as stream,
        ZipFile(stream, "w", compression=ZIP_DEFLATED) as zipped,
    ):
        for path in sorted(source.iterdir()):
            zipped.write(path, path.name)
    archive.chmod(0o600)
    return file_hash(archive)


def restore_archive(archive: Path, destination: Path) -> dict[str, int]:
    """Проверить ограниченный ZIP и восстановить его содержимое в новый каталог."""
    with (
        TemporaryDirectory(prefix="delta-restore-") as temp,
        ZipFile(archive) as zipped,
    ):
        items = zipped.infolist()
        if (
            not 1 <= len(items) <= 1025
            or sum(i.file_size for i in items) > MAX_ARCHIVE_BYTES
        ):
            raise ValueError("ZIP-копия превышает ограничения")
        names = set()
        for item in items:
            if (
                not re.fullmatch(r"[\w.-]{1,128}", item.filename)
                or item.filename in {".", ".."}
                or item.filename in names
            ):
                raise ValueError("Небезопасное или повторное имя в ZIP-копии")
            names.add(item.filename)
            with (
                zipped.open(item) as src,
                (Path(temp) / item.filename).open("xb") as dst,
            ):
                shutil.copyfileobj(src, dst, 1024**2)
        verify_snapshot(Path(temp))
        return restore_snapshot(Path(temp), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("create", "verify", "restore", "pack", "restore-zip")
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    result: dict[str, int] | dict[str, str]
    if args.action != "verify" and args.destination is None:
        parser.error("Нужен --destination")
    if args.action == "verify":
        result = verify_snapshot(args.source)
    else:
        assert args.destination is not None
        if args.action == "create":
            result = create_online_snapshot(args.source, args.destination)
        elif args.action == "restore":
            result = restore_snapshot(args.source, args.destination)
        elif args.action == "restore-zip":
            result = restore_archive(args.source, args.destination)
        else:
            result = {"sha256": pack_snapshot(args.source, args.destination)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
