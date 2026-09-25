"""Явные команды сцены без дополнительных запросов к модели."""

import re

STOP_PATTERN = re.compile(
    r"^\s*(?:стоп|хватит|выйди из|закончим|закончи)\s+(?:rp|рп)"
    r"(?=$|[\s,.!;:—–-])\s*[,!.;:—–-]*\s*",
    re.IGNORECASE,
)
CONFIG_PATTERN = re.compile(
    r"^\s*(?:давай\s+)?(?:в этой сцене\s+)?ты\s+"
    r"в\s+(женской|мужской)\s+конфигурации(?=$|[\s.!;,])",
    re.IGNORECASE,
)
CHARACTER_PATTERN = re.compile(
    r"^\s*мой персонаж\s*:\s*([^\n]{1,500})\s*$", re.IGNORECASE
)


def split_roleplay_stop(text: str) -> str | None:
    """Вернуть оставшийся вопрос; None означает отсутствие команды остановки."""
    match = STOP_PATTERN.match(text)
    if match is None:
        return None
    start = match.end()
    return text[start:].strip()


def scene_configuration(text: str) -> str | None:
    """Распознавать только явное назначение конфигурации Дельты."""
    match = CONFIG_PATTERN.match(text)
    if match is None:
        return None
    return "female" if match[1].lower() == "женской" else "male"


def scene_character(text: str) -> str | None:
    """Сохранить только описание, явно заданное пользователем."""
    match = CHARACTER_PATTERN.match(text)
    return match[1].strip() if match else None
