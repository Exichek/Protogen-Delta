"""Проверка описаний внешности без генерации или исполнения инструкций."""

import json

APPEARANCE_LIMIT = 2000
MAX_DESCRIPTION_BYTES = 64 * 1024
_FIELDS = {
    "species": "Вид",
    "build": "Телосложение",
    "colors": "Окрас",
    "head": "Голова",
    "face": "Лицо",
    "eyes": "Глаза",
    "hair": "Волосы",
    "fur": "Шерсть",
    "scales": "Чешуя",
    "horns": "Рога",
    "ears": "Уши",
    "limbs": "Конечности",
    "tail": "Хвост",
    "wings": "Крылья",
    "markings": "Узоры",
    "clothing": "Одежда",
    "accessories": "Аксессуары",
    "details": "Детали",
    "uncertain": "Неясные детали",
}


def validate_description(text: str) -> str:
    """Сохранять текст целиком или сообщать ошибку, не обрезать признаки молча."""
    if not isinstance(text, str):
        raise ValueError("Описание должно быть текстом.")
    text = text.strip()
    if not text:
        raise ValueError("Добавь описание внешности.")
    if len(text) > APPEARANCE_LIMIT:
        raise ValueError(f"Описание должно быть до {APPEARANCE_LIMIT} знаков.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        raise ValueError("В описании есть недопустимые управляющие символы.")
    return text


def parse_description_file(data: bytes, *, json_file: bool) -> str:
    """Принять UTF-8/UTF-16 TXT или JSON с описанием либо визуальными полями."""
    if len(data) > MAX_DESCRIPTION_BYTES:
        raise ValueError("TXT/JSON должен быть до 64 КБ.")
    try:
        encoding = (
            "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        )
        text = data.decode(encoding)
    except UnicodeError as error:
        raise ValueError("Сохрани файл в UTF-8 или UTF-16.") from error
    if not json_file:
        return validate_description(text)
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ValueError("Некорректный JSON.") from error
    if isinstance(value, str):
        return validate_description(value)
    if not isinstance(value, dict) or not value:
        raise ValueError(
            'Нужен JSON-объект, например {"appearance":"Описание внешности"}.'
        )
    if set(value) in ({"appearance"}, {"description"}):
        return validate_description(next(iter(value.values())))
    if set(value) - _FIELDS.keys():
        raise ValueError(
            "Неизвестные поля JSON. Используй appearance или поля species, colors, head, eyes, tail, clothing, details."
        )
    lines = []
    for key, item in value.items():
        if isinstance(item, list) and all(isinstance(part, str) for part in item):
            item = ", ".join(item)
        if not isinstance(item, str):
            raise ValueError("Поля внешности должны содержать текст или список строк.")
        if item.strip():
            lines.append(f"{_FIELDS[key]}: {item.strip()}")
    return validate_description("\n".join(lines))
