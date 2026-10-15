"""Распознавать служебную разметку, выданную вместо итогового ответа."""

import re

_DSML = re.compile(r"<\s*/?\s*(?:[|｜]\s*)+DSML\s*(?:[|｜]\s*)+", re.I)
_CODE_QUOTE = re.compile(r"```.*?(?:```|\Z)|`[^`\n]*(?:`|$)", re.S)


def contains_tool_markup(text: str) -> bool:
    """Не интерпретировать текстовые вызовы как разрешение исполнить инструмент."""
    return _DSML.search(text) is not None


def contains_unquoted_tool_markup(text: str) -> bool:
    """Позволять объяснение разметки в блоке кода, но не неперехваченный вызов."""
    return contains_tool_markup(_CODE_QUOTE.sub(" ", text))
