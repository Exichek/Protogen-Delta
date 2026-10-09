"""Явный возраст и сигналы остановки обрабатываются до запроса к модели."""

import re


def declares_minor(text: str) -> bool:
    """Распознать прямое самоописание; цитаты и возраст персонажа не равны возрасту пользователя."""
    text = re.sub(r"```.*?(?:```|$)|`[^`\n]*`|[«\"].*?[»\"]", "", text, flags=re.S)
    start = r"(?:^|[.!?\n]\s*)\s*"
    if re.search(
        start + r"(?:я\s+несовершеннолетн\w*|мне\s+(?:ещ[её]\s+)?нет\s+18)\b",
        text,
        re.I,
    ):
        return True
    for match in re.finditer(
        start
        + r"(?:мне\s+(?:сейчас\s+)?|мой\s+возраст\s*[:—-]?\s*)(\d{1,2})(?=\s*(?:лет\b|год\w*\b|[.!?,…]|$))",
        text,
        re.I,
    ):
        if 0 < int(match[1]) < 18:
            return True
    return False


def reference_is_child(text: str) -> bool:
    """Явное описание детского персонажа в тексте референса."""
    if re.search(
        r"\b(?:реб[её]нок|детский\s+персонаж|несовершеннолетн\w*|малолетн\w*)\b",
        text,
        re.I,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:персонаж\w*|ему|ей)\s+(?:[1-9]|1[0-7])\s+(?:лет|год\w*)\b",
            text,
            re.I,
        )
    )


def is_scene_stop(text: str, stopword: str) -> bool:
    """Отдельная реплика стоп-слова, без поиска в цитатах и обычном обсуждении."""
    normalized = text.strip().strip(".!?… ").casefold()
    return normalized in {"стоп", "красный", "stop", "red", stopword.casefold()}


def validate_stopword(text: str) -> str:
    value = text.strip()
    if not re.fullmatch(r"[\w -]{1,40}", value, flags=re.UNICODE):
        raise ValueError("Стоп-слово: от 1 до 40 букв, цифр, пробелов или дефисов.")
    return value
