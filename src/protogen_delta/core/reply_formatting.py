"""Небольшой безопасный набор оформления через entities Telegram."""

import re
from collections.abc import Sequence

from aiogram.types import MessageEntity

from protogen_delta.core.roleplay import roleplay_actions

_FENCE = re.compile(r"(?m)^[ \t]*```")
_TOKEN = re.compile(
    r"(?m:^[ \t]*```(?P<language>[^\n`]*)\n(?P<pre>[\s\S]*?)^[ \t]*```[^\S\n]*(?=\n|$))"
    r"|(?<!`)`(?P<code>[^`\n]+)`(?!`)"
    r"|(?<![\w*])\*\*(?P<bold>[^*\n]+)\*\*(?![\w*])"
    r"|(?<![\w*])\*(?P<italic>[^*\n]+)\*(?![\w*])"
)


def format_reply_text(text: str) -> tuple[str, list[MessageEntity]]:
    """Оформить закрытые конструкции, не исполняя HTML или произвольный Markdown."""
    if len(_FENCE.findall(text)) % 2:
        return text, []
    # Неполные или неподдерживаемые code-маркеры не должны стать RP-ремаркой.
    if text.count("`") % 2 or re.search(r"(?m)^\s*~~~", text):
        return text, []
    if len(re.findall(r"```", text)) != len(_FENCE.findall(text)):
        return text, []
    tokens = list(_TOKEN.finditer(text))
    if sum(match["pre"] is not None for match in tokens) * 2 != len(
        _FENCE.findall(text)
    ):
        return text, []
    pieces: list[str] = []
    entities: list[MessageEntity] = []
    previous = offset = 0
    for match in tokens:
        start, end = match.span()
        kind = next(
            name
            for name in ("pre", "code", "bold", "italic")
            if match[name] is not None
        )
        if kind == "italic" and not roleplay_actions(match.group()):
            continue
        if (
            kind == "italic"
            and match[kind].startswith(" ")
            and match[kind].endswith(" ")
            and re.search(r"\w\s+$", text[:start])
            and re.match(r"\s+\w", text[end:])
        ):
            continue
        prefix, content = text[previous:start], match[kind]
        if not content.strip():
            continue
        pieces.extend((prefix, content))
        offset += len(prefix.encode("utf-16-le")) // 2
        length = len(content.encode("utf-16-le")) // 2
        entities.append(
            MessageEntity(
                type=kind,
                offset=offset,
                length=length,
                language=(match["language"].strip() or None) if kind == "pre" else None,
            )
        )
        offset += length
        previous = match.end()
    pieces.append(text[previous:])
    return "".join(pieces), entities


def format_reply_parts(parts: Sequence[str]) -> list[tuple[str, list[MessageEntity]]]:
    """При разрезанном длинном code block оставить затронутые части дословными."""
    result: list[tuple[str, list[MessageEntity]]] = []
    in_code = False
    for part in parts:
        odd_fences = bool(len(_FENCE.findall(part)) % 2)
        result.append((part, []) if in_code or odd_fences else format_reply_text(part))
        in_code ^= odd_fences
    return result
