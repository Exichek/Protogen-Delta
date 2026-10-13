"""Разобрать предоставленную владельцем текстовую выгрузку Telegram."""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protogen_delta.repositories.dialogue_archive import DialogueMessage

_ENTRY = re.compile(r"^\[(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})\] ([^:\r\n]+): ?", re.M)


def telegram_text_messages(
    source: Path, user_name: str, bot_name: str
) -> list[DialogueMessage]:
    """Считать только двух явно указанных участников; время в выгрузке — МСК."""
    if not user_name or not bot_name or user_name == bot_name:
        raise ValueError("Укажи разные имена пользователя и бота")
    if source.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Текстовая выгрузка превышает 20 МБ")
    raw = source.read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    entries = list(_ENTRY.finditer(text))
    if not entries or len(entries) > 10000:
        raise ValueError("Не найдены сообщения или превышен предел 10000 реплик")
    digest = hashlib.sha256(raw).hexdigest()
    messages = []
    for index, entry in enumerate(entries):
        author = entry.group(2)
        if author not in {user_name, bot_name}:
            raise ValueError("В выгрузке есть неизвестный участник")
        end = entries[index + 1].start() if index + 1 < len(entries) else len(text)
        start = entry.end()
        body = text[start:end].strip()
        if not body:
            continue
        date = datetime.strptime(entry.group(1), "%d.%m.%Y %H:%M").replace(
            tzinfo=timezone(timedelta(hours=3))
        )
        messages.append(
            DialogueMessage(
                f"import:{digest}:{index}",
                "user" if author == user_name else "assistant",
                body,
                date.timestamp() + index * 0.00001,
                "telegram_text_import",
            )
        )
    return messages
