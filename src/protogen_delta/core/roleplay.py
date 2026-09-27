"""Явные команды сцены без дополнительных запросов к модели."""

import re

STOP_PATTERN = re.compile(
    r"^\s*(?:стоп|хватит|выйди из|закончим|закончи)\s+(?:rp|рп)"
    r"(?=$|[\s,.!?…;:—–-])\s*[,!.?…;:—–-]*\s*",
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
INTENT_PATTERN = re.compile(
    r"(?:\b(?:давай|хочу|начн[её]м|начать)\b.{0,40}\b(?:rp|рп|рол(?:ку|евую))\b|"
    r"\b(?:сыграем|поиграем|отыграем)\b.{0,40}\b(?:сцену|сюжет|роли?)\b)",
    re.IGNORECASE | re.DOTALL,
)


def has_roleplay_action(text: str) -> bool:
    """Распознать текст в одиночных звёздочках вне кода и формул."""
    without_code = re.sub(r"```.*?(?:```|$)|`[^`\n]*(?:`|$)", "", text, flags=re.DOTALL)
    for match in re.finditer(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", without_code):
        action = match[1].strip()
        if re.search(r"[^\W\d_]{2,}", action) and not re.search(r"[=+^<>/\\|]", action):
            return True
    return False


def has_roleplay_intent(text: str) -> bool:
    """Распознать явное предложение начать RP без действия в звёздочках."""
    return INTENT_PATTERN.search(text) is not None


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
