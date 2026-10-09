"""Оформление только заголовков справки, без общего Markdown-парсера."""

import re

from aiogram.types import MessageEntity

_HEADING = re.compile(r"(?m)^(\s*(?:[-•]\s+|\d+\.\s+)?)(\*\*([^*\n]+)\*\*)")


def format_capability_headings(text: str) -> tuple[str, list[MessageEntity]]:
    """Снять ** с заголовков и задать корректные UTF-16 entities Telegram."""
    pieces: list[str] = []
    entities: list[MessageEntity] = []
    previous = 0
    offset = 0
    fences = [match.start() for match in re.finditer(r"(?m)^[ \t]*```", text)]
    for match in _HEADING.finditer(text):
        start = match.start()
        if sum(position < start for position in fences) % 2:
            continue
        prefix = text[previous:start] + match.group(1)
        heading = match.group(3)
        pieces.extend((prefix, heading))
        offset += len(prefix.encode("utf-16-le")) // 2
        length = len(heading.encode("utf-16-le")) // 2
        entities.append(MessageEntity(type="bold", offset=offset, length=length))
        offset += length
        previous = match.end()
    pieces.append(text[previous:])
    return "".join(pieces), entities
