"""Явный возраст и сигналы остановки обрабатываются до запроса к модели."""

import re
import unicodedata

_AGE_WORDS = {
    word: age
    for words in (
        "один два три четыре пять шесть семь восемь девять десять "
        "одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать",
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen",
    )
    for age, word in enumerate(words.split(), start=1)
}
_AGE_WORDS.update({"одна": 1, "две": 2})
_QUOTED = re.compile(
    r"```.*?(?:```|$)|~~~.*?(?:~~~|$)|`[^`\n]*`|«[^»]*»|“[^”]*”|„[^“]*“"
    r'|"[^"]*"|(?<!\w)\'(?:[^\'\n]|(?<=\w)\'(?=\w))*\'(?!\w)',
    re.S,
)
_SELF_AGE = re.compile(
    r"\b(?:мне\s+(?:(?:сейчас|щас|счас|уже|всего|только|ещё|еще)\s+){0,3}"
    r"|мой\s+возраст\s*[:—=-]?\s*"
    r"|i(?:'m|\s+am)\s+(?:(?:currently|only|just|now|still)\s+){0,3}"
    r"|my\s+age\s*(?:is\s+|[:=-]\s*))"
    r"(?P<age>\d{1,3}|" + "|".join(_AGE_WORDS) + r")\b"
    r"(?=\s*(?:лет\b|года?\b|годик(?:а|ов)?\b|years?\b|yrs?\b|y/o\b"
    r"|[.!?,;:…)]|$|(?:и|а|но|and|but)\b))"
)
_UNDERAGE = re.compile(
    r"\b(?:я\s+(?:(?:ещё|еще|пока)\s+)?несовершеннолетн\w*"
    r"|мне\s+(?:(?:ещё|еще|сейчас|щас)\s+)?"
    r"(?:(?:нет|меньше)\s+(?:18|восемнадцати)"
    r"|(?:почти|скоро)\s+(?:18|восемнадцать)"
    r"|(?:ещё\s+|еще\s+)?не\s+исполнилось\s+(?:18|восемнадцать))"
    r"|i(?:'m|\s+am)\s+(?:a\s+minor|under\s+(?:18|eighteen)"
    r"|not\s+(?:18|eighteen)\s+yet))\b"
)
_GRADE = (
    r"(?:(?:[1-9]|1[0-2])(?:\s*-?\s*(?:ого|ый|ой|ом|й|м))?"
    r"|(?:перв|втор|треть|четвёрт|четверт|пят|шест|седьм|восьм|девят|десят"
    r"|одиннадцат|двенадцат)(?:ый|ой|ий|ом|ем|ого|его))"
)
_SCHOOL = re.compile(
    r"\b(?:я\s+(?:(?:ещё|еще|сейчас|пока)\s+)?"
    r"(?:школьник(?:ом)?|школьниц(?:а|ей)|ученик(?:ом)?\s+"
    + _GRADE
    + r"\s*класса|учениц(?:а|ей)\s+"
    + _GRADE
    + r"\s*класса|в\s+"
    + _GRADE
    + r"\s*классе)"
    r"|(?:учусь|хожу|перешёл|перешел|перешла)\s+в\s+" + _GRADE + r"\s*класс(?:е)?"
    r"|i(?:'m|\s+am)\s+(?:a\s+school(?:boy|girl)|in\s+(?:the\s+)?"
    r"(?:[1-9]|1[0-2])(?:st|nd|rd|th)?\s+grade))\b"
)
_NOT_CURRENT_PREFIX = re.compile(
    r"(?:\b(?:если(?:\s+бы)?|когда|раньше|тогда|when|if)\s*"
    r"|\b(?:допустим|представь(?:те)?|представим)\s*[, :—-]*(?:что\s+)?"
    r"|\b(?:в\s+(?:детстве|прошлом|игре|роли|рассказе)|по\s+(?:сюжету|роли))\s*"
    r"|\b(?:пример|фраза|текст|цитата)\s*[:—]\s*)$"
)
_NOT_CURRENT_SUFFIX = re.compile(
    r"^\s*(?:(?:лет|года?|years?(?:\s+old)?)\s+)?"
    r"(?:было|будет|опыта|стажа|назад|ago|of\s+experience|in\s+the\s+game"
    r"|в\s+(?:игре|роли|рассказе)|по\s+(?:сюжету|роли))\b"
)


def _is_current_statement(text: str, match: re.Match[str]) -> bool:
    start, end = match.span()
    return not (
        _NOT_CURRENT_PREFIX.search(text[:start])
        or _NOT_CURRENT_SUFFIX.search(text[end:])
    )


def declares_minor(text: str) -> bool:
    """Распознать прямое самоописание; цитаты и возраст персонажа не равны возрасту пользователя."""
    text = (
        unicodedata.normalize("NFKC", text)
        .casefold()
        .replace("’", "'")
        .replace("‘", "'")
    )
    # Replace quoted spans with whitespace so words on their sides cannot join.
    text = _QUOTED.sub(" ", text)
    for pattern in (_UNDERAGE, _SCHOOL):
        if any(_is_current_statement(text, match) for match in pattern.finditer(text)):
            return True
    for match in _SELF_AGE.finditer(text):
        value = match["age"]
        age = int(value) if value.isdecimal() else _AGE_WORDS[value]
        if 0 < age < 18 and _is_current_statement(text, match):
            return True
    return bool(
        re.fullmatch(
            r"\s*(?:мой\s+класс\s*[:—=-]?\s*)?" + _GRADE + r"\s*класс\s*[.!]?\s*", text
        )
    )


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
            r"\b(?:[1-9]|1[0-7])\s+(?:лет|год\w*)\b",
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
