"""Явная просьба прислать реальное изображение, а не описание или генерацию."""

import re

_REQUEST = re.compile(
    r"\b(?:скинь|пришли|отправь|покажи|найди|сбрось|кинь|"
    r"(?:можешь|можно).{0,25}(?:прислать|скинуть|отправить|показать))\b.{0,100}"
    r"\b(?:фото\w*|фотк\w*|картинк\w*|изображени\w*|скрин\w*|обои|арт\w*)\b",
    re.I | re.S,
)
_NEGATIVE = re.compile(
    r"\b(?:не\s+(?:присылай|скидывай|отправляй|показывай|ищи)|"
    r"как\s+(?:отправить|прислать|скинуть)|(?:сгенерируй|нарисуй|создай)\b)|"
    r"\b(?:скинь|пришли|найди|отправь|покажи)\s+(?:мне\s+)?(?:только\s+)?ссылк\w*",
    re.I,
)


def _own_text(text: str) -> str:
    return re.sub(
        r'```.*?(?:```|\Z)|`[^`]*`|«[^»]*»|"[^"\n]*"|^\s*>[^\n]*',
        " ",
        text,
        flags=re.S | re.M,
    )


def is_web_image_request(text: str) -> bool:
    text = _own_text(text)
    return bool(_REQUEST.search(text) and not _NEGATIVE.search(text))


def web_image_limit(text: str) -> int:
    """Одна картинка по умолчанию; несколько — только по явной просьбе."""
    text = _own_text(text)
    count = re.search(
        r"\b(\d+|одну|один|одно|две|два|три|несколько|пару)\s+"
        r"(?:разных\s+)?(?:фото\w*|фотк\w*|картинк\w*|изображени\w*|скрин\w*|арт\w*)\b",
        text,
        re.I,
    )
    if count is None:
        return 1
    value = count[1].lower()
    if value.isdigit():
        return min(3, max(1, int(value)))
    return 1 if value in {"одну", "один", "одно"} else 3 if value == "три" else 2
