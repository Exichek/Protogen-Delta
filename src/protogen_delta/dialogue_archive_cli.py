"""Экспорт, восстановление или импорт личной переписки без запросов модели."""

import argparse
import asyncio
import json
from pathlib import Path

from protogen_delta.core.dialogue_import import telegram_text_messages
from protogen_delta.repositories.dialogue_archive import DialogueArchiveRepository


async def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "restore", "import-text"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--user-name", default="")
    parser.add_argument("--bot-name", default="")
    args = parser.parse_args()
    repository = DialogueArchiveRepository(args.data_dir)
    if args.action == "export":
        count = await repository.export(args.user_id, args.file)
    elif args.action == "restore":
        count = await repository.import_snapshot(args.user_id, args.file)
    else:
        messages = telegram_text_messages(args.file, args.user_name, args.bot_name)
        count = await repository.save_many(args.user_id, messages)
    print(
        json.dumps({"user_id": args.user_id, "messages": count, "action": args.action})
    )


if __name__ == "__main__":
    asyncio.run(run())
