"""Исправление ложного отрицания доступной отправки стикеров в ответе модели."""

import re

STICKER_CAPABILITY_REPLY = (
    "У меня есть настоящие Telegram-стикеры из моего набора. "
    "Могу прислать подходящую реакцию отдельным сообщением."
)
_PROTECTED = re.compile(
    r'```[\s\S]*?```|`[^`]*`|«[^»]*»|“[^”]*”|"[^"\n]*"|https?://\S+'
)
_STICKERS = re.compile(r"\b(?:стикер\w*|stickers?)\b", re.I)
_SELF = re.compile(
    r"\b(?:я|мне|меня|мои\s+стикер\w*|мой\s+набор)\b|"
    r"\b(?:не\s+(?:могу|умею)|(?:это|моё|мое)\s+приложение)\b",
    re.I,
)
_OTHER = re.compile(r"\b(?:у\s+(?:него|неё|нее|них)|друг\w*|он|она|они)\b", re.I)
_DENIAL = re.compile(
    r"\bне\s+(?:могу|умею)\s+(?:\w+\s+){0,3}"
    r"(?:отправ\w*|присыл\w*|присл\w*|скин\w*|скидыв\w*|показ\w*|встав\w*|использ\w*)\b|"
    r"\b(?:отправ\w*|присыл\w*|присл\w*|скин\w*|показ\w*|встав\w*)\b"
    r".{0,40}\bне\s+(?:могу|умею)\b|"
    r"\b(?:нет|не\s+предусмотрено)\s+(?:такой\s+)?(?:функции|возможности)\b|"
    r"\b(?:функции|возможности)\s+нет\b|"
    r"\b(?:только|лишь|максимум)\s+(?:словесн\w*|текст\w*|мысленн\w*)\b|"
    r"\b(?:нет\s+стикер\w*|стикер\w*\s+нет)\b",
    re.I,
)
_LIMITATION = re.compile(
    r"\b(?:нарис\w*|рисова\w*|созда\w*|генер\w*|кажд\w*|подряд|"
    r"чаще|слишком\s+часто|таймаут\w*|ошибк\w*|отключ\w*|"
    r"пауза|интервал\w*|(?:именно\s+)?этот\s+стикер\w*|из\s+чужого)\b|"
    r"\b(?:сейчас|пока)\s+не\s+могу\b",
    re.I,
)
_RETRACTION = re.compile(
    r"\b(?:ошиб\w*|неправд\w*|неверн\w*)\b|"
    r"\b(?:раньше|прежде)\b.*\b(?:теперь|сейчас)\s+могу\b",
    re.I,
)


def correct_sticker_capability(reply: str) -> str:
    """Заменить только собственное ложное отрицание, сохраняя остальную реплику.

    Вызывающий сервис обязан проверить реальную доступность пака. Цитаты, код,
    ограничения частоты и сообщения об ошибке доставки остаются без изменений.
    """
    visible = _PROTECTED.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), reply)
    edits: list[tuple[int, int, str]] = []
    for sentence in re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", visible):
        text = sentence.group()
        start, end = sentence.span()
        original = reply[start:end]
        if (
            not _STICKERS.search(original)
            or not _SELF.search(text)
            or not _DENIAL.search(text)
            or _OTHER.search(text)
            or _LIMITATION.search(text)
            or _RETRACTION.search(text)
        ):
            continue
        leading = original[: len(original) - len(original.lstrip())]
        stripped_end = len(original.rstrip())
        trailing = original[stripped_end:]
        replacement = STICKER_CAPABILITY_REPLY if not edits else ""
        edits.append(
            (sentence.start(), sentence.end(), leading + replacement + trailing)
        )
    for start, end, replacement in reversed(edits):
        reply = reply[:start] + replacement + reply[end:]
    return reply
