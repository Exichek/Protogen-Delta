"""Выбор размышлений для одного ответа без сохранения режима в профиле."""

import re

_QUOTED = re.compile(
    r'```.*?(?:```|\Z)|~~~.*?(?:~~~|\Z)|`[^`\n]*`|«[^»]*»|“[^”]*”|"[^"]*"|^>[^\n]*',
    re.S | re.M,
)
_OFF = re.compile(
    r"\b(?:без\s+(?:режима\s+)?размышлений|не\s+включай\s+размышления|"
    r"без\s+thinking|no\s+reasoning)\b",
    re.I,
)
_EXPLICIT = re.compile(
    r"\b(?:подумай|поразмышляй|продумай|think\s+carefully|"
    r"включи\s+(?:режим\s+)?размышлен\w*)\b",
    re.I,
)
_COMPLEX = re.compile(
    r"\b(?:докажи|доказательство|пошагов\w*|отлад\w*|debug|"
    r"разбер\w*\s+(?:ошиб\w*|баг\w*|код)|проверь\s+(?:код|расч[её]т\w*)|"
    r"сравни\s+.{0,80}(?:решени\w*|архитектур\w*|ограничени\w*))\b",
    re.I,
)
_SMALL_TALK = re.compile(
    r"\s*(?:привет|ку|спасибо|пока|ага|ок(?:ей)?|как\s+дела|соскучился)\W*",
    re.I,
)


def needs_reasoning(message: str, classified: bool = False) -> bool:
    """Команды внутри цитат/кода не переключают режим ответа."""
    text = _QUOTED.sub(" ", message)
    if _OFF.search(text) or _SMALL_TALK.fullmatch(text) or not text.strip():
        return False
    return bool(_EXPLICIT.search(text) or _COMPLEX.search(text) or classified is True)
