"""Копирование данных остановленного бота в новый каталог без перезаписи."""

import argparse
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


def copy_snapshot(source: Path, destination: Path) -> None:
    """Проверить SQLite/JSON и создать отдельную копию для backup или restore.

    Бот должен быть остановлен: SQLite backup согласован сам по себе,
    но общей транзакции между SQLite и двумя JSON-файлами нет.
    """
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError("Каталог назначения уже существует")
    if not source.is_dir():
        raise ValueError("Источник должен быть каталогом данных")

    payloads: dict[str, bytes] = {}
    for filename, key, value_type in (
        ("users.json", "USERS", int),
        ("images.json", "IMAGES", str),
    ):
        payload = (source / filename).read_bytes()
        data = json.loads(payload)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get(key), list)
            or not all(type(item) is value_type for item in data[key])
        ):
            raise ValueError(f"Некорректный формат {filename}")
        payloads[filename] = payload

    database = source / "user_states.db"
    uri = database.as_uri() + "?mode=ro"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".delta-snapshot-", dir=destination.parent) as temp:
        staging = Path(temp) / "data"
        staging.mkdir()
        with (
            closing(sqlite3.connect(uri, uri=True)) as src,
            closing(sqlite3.connect(staging / database.name)) as dst,
        ):
            src.backup(dst)
            if dst.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Проверка целостности SQLite не прошла")
            dst.execute("SELECT user_id FROM user_states LIMIT 0")
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
        # Не подменяем существующие рабочие данные, даже пустой каталог.
        destination.mkdir()
        try:
            for item in staging.iterdir():
                shutil.copyfile(item, destination / item.name)
        except BaseException:
            shutil.rmtree(destination)
            raise


def main() -> None:
    """Создать резервную копию или восстановить её в отдельный каталог."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--bot-stopped",
        action="store_true",
        help="Подтверждение, что процесс бота остановлен",
    )
    args = parser.parse_args()
    if not args.bot_stopped:
        parser.error("Сначала остановите бот и добавьте --bot-stopped")
    copy_snapshot(args.source, args.destination)
    print(f"{args.action}: копия проверена и сохранена в {args.destination}")


if __name__ == "__main__":
    main()
