"""Распознать закрытие технической темы и форму конкретного ответа."""

import re
from collections.abc import Sequence

_QUOTED = re.compile(
    r'```[\s\S]*?```|`[^`]*`|«[^»]*»|“[^”]*”|"[^"\n]*"|https?://\S+|(?m:^\s*>[^\n]*)'
)
_TECHNICAL_CLOSURE = re.compile(
    r"(?:^|[.!?;\n])\s*(?:"
    r"с\s+(?:кодом|репозиторием|разбором\s+кода)\s+(?:пока\s+)?вс[её]"
    r"(?=\s*(?:$|[^\w\s]))|"
    r"(?:код|репозиторий|тему\s+кода)\s+"
    r"(?:(?:сегодня|пока|теперь|больше)\s+){0,3}не\s+(?:обсуждаем|разбираем|трогаем)\b|"
    r"не\s+(?:возвращайся|возвращаемся)\s+к\s+(?:коду|репозиторию)\b|"
    r"(?:давай\s+)?(?:хватит|больше\s+не)\s+(?:про\s+код|о\s+коде)\b|"
    r"(?:давай\s+)?(?:закроем|закрыли|закрываем|закончили)\s+"
    r"(?:тему\s+кода|разбор\s+кода|с\s+кодом)\b|"
    r"(?:давай\s+)?(?:не\s+будем|хватит)\s+(?:обсуждать|разбирать)\s+код\b"
    r")",
    re.I,
)
_ONE_ITEM = re.compile(
    r"(?:\b(?:расскажи|придумай|дай|скажи|напиши|покажи|приведи|ещ[её]|нужна|нужен|хочу)\s+"
    r"(?:(?:мне|нам|только|ещ[её]|пожалуйста)\s+){0,3}|(?:^|[.!?;\n])\s*)"
    r"\b(?:одну|одна|один|одно|1)\s+"
    r"(?:(?:короткую|новую|маленькую|короткий|небольшой)\s+){0,3}"
    r"(?P<kind>шутк[ау]|пример|предложение|фразу)\b",
    re.I,
)
_LINES = re.compile(
    r"\b(?:максимум|не\s+больше|не\s+более)\s+"
    r"(?P<count>одну|две|три|1|2|3)\s+строк[иу]?\b",
    re.I,
)


def _visible(message: str) -> str:
    """Оставить координаты исходного текста, скрыв недоверенные цитаты и код."""
    return _QUOTED.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), message)


def closes_technical_topic(message: str) -> bool:
    return _TECHNICAL_CLOSURE.search(_visible(message)) is not None


def technical_request_text(message: str) -> str:
    """Закрытие кода не включает технический режим, новая просьба после него может."""
    matches = list(_TECHNICAL_CLOSURE.finditer(_visible(message)))
    for match in reversed(matches):
        start, end = match.span()
        message = message[:start] + " " + message[end:]
    return message


def turn_contract_context(messages: Sequence[str]) -> str:
    """Вызывающий код передаёт только собственные управляющие реплики, не вложения."""
    rules: list[str] = []
    if any(closes_technical_topic(message) for message in messages):
        rules.append(
            "Тема кода закрыта пользователем. Не возвращай её сам, даже подколом "
            "или предложением продолжить. Новую явную просьбу о коде можно выполнить."
        )
    # Последняя собственная реплика может изменить форму просьбы внутри пачки.
    for message in messages[-1:]:
        visible = _visible(message)
        for item in _ONE_ITEM.finditer(visible):
            if re.search(r"\bне\s*$", visible[: item.start()], re.I):
                continue
            if item.group("kind").casefold().startswith("шутк"):
                rules.append(
                    "Нужна ровно одна шутка. Без второй попытки, самооценки и "
                    "объяснения шутки; короткую просьбу не превращай в отчёт."
                )
            else:
                rules.append(
                    "Дай ровно один запрошенный пример, фразу или предложение."
                )
        lines = _LINES.search(visible)
        if lines:
            count = {"одну": "1", "две": "2", "три": "3"}.get(
                lines.group("count").casefold(), lines.group("count")
            )
            rules.append(f"Ответ не длиннее {count} смысловых строк, без вступления.")
        if re.search(
            r"\b(?:коротко|кратко|покороче|вкратце)\b", visible, re.I
        ) and not re.search(
            r"\bне\s+(?:коротко|кратко|покороче|вкратце)\b", visible, re.I
        ):
            rules.append("Нужен краткий ответ; оставь только запрошенное по существу.")
        if re.search(r"\bбез\s+(?:ссылок|пояснений|объяснений)\b", visible, re.I):
            rules.append("Соблюдай текущую просьбу без ссылок или пояснений.")
    if not rules:
        return ""
    return "Форма текущего ответа по собственной просьбе пользователя: " + " ".join(
        dict.fromkeys(rules)
    )
