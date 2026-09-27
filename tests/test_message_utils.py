"""Тесты утилит для работы с сообщениями."""

import pytest

from protogen_delta.core.message_utils import (
    reply_delay_seconds,
    split_message,
    split_reply,
)


def test_split_message_returns_short_message_unchanged() -> None:
    """Короткое сообщение должно возвращаться одной частью."""
    result = split_message("Привет", limit=10)

    assert result == ["Привет"]


def test_split_message_splits_long_message_by_limit() -> None:
    """Длинное сообщение без переносов должно делиться по лимиту."""
    result = split_message("123456789012345", limit=10)

    assert result == ["1234567890", "12345"]


def test_split_message_prefers_newline() -> None:
    """При наличии переноса строк сообщение должно делиться по нему."""
    result = split_message(
        "12345\n6789012345",
        limit=10,
    )

    assert result == ["12345", "6789012345"]


def test_split_message_does_not_create_empty_part() -> None:
    """Перенос в начале длинного текста не должен создавать пустую часть."""
    result = split_message(
        "\n123456789012345",
        limit=10,
    )

    assert result

    assert all(part for part in result)


def test_split_reply_keeps_short_and_single_paragraph_answers_whole() -> None:
    """Короткая реплика и сплошной текст не должны дробиться искусственно."""
    assert split_reply("Короткий ответ.\n\nИ ещё слово.") == [
        "Короткий ответ.\n\nИ ещё слово."
    ]
    one_block = "Одно длинное предложение. " * 12
    assert split_reply(one_block) == [one_block.strip()]

    short_paragraphs = "Первый короткий абзац. " * 6 + "\n\n" + "Второй короткий. " * 5
    assert split_reply(short_paragraphs) == [short_paragraphs.strip()]


def test_split_reply_uses_natural_paragraphs_and_caps_message_count() -> None:
    """Большой ответ должен стать максимум четырьмя репликами."""
    blocks = [f"Абзац {number}: " + "текст " * 80 for number in range(1, 7)]
    result = split_reply("\n\n".join(blocks))

    assert len(result) == 4
    assert result[0] == blocks[0].strip()
    assert result[1] == blocks[1].strip()
    assert result[2] == blocks[2].strip()
    assert result[3] == "\n\n".join(block.strip() for block in blocks[3:])


def test_split_reply_balances_text_longer_than_telegram_limit() -> None:
    """Очень длинный ответ должен делиться разговорно, а не на два гигантских куска."""
    blocks = [
        f"Раздел {number}: " + " ".join(["текст"] * 145) for number in range(1, 7)
    ]
    text = "\n\n".join(blocks)

    result = split_reply(text)

    assert len(text) > 4096
    assert len(result) == 4
    assert all(len(part) <= 4096 for part in result)
    assert "\n\n".join(result) == text.strip()


def test_split_reply_keeps_tiny_paragraph_with_previous_message() -> None:
    """Короткий хвост не должен отправляться отдельным сообщением."""
    first = "Первый содержательный абзац. " * 10
    second = "Второй содержательный абзац. " * 10
    result = split_reply(f"{first}\n\n{second}\n\nНу как?")

    assert result == [first.strip(), f"{second.strip()}\n\nНу как?"]


def test_split_reply_does_not_split_inside_code_block() -> None:
    """Пустые строки внутри fenced code block должны сохраняться."""
    intro = "Объяснение перед примером. " * 22
    code = "```python\nfirst = 1\n\nsecond = 2\n```"
    result = split_reply(f"{intro}\n\n{code}")

    assert result == [intro.strip(), code]


@pytest.mark.parametrize(
    ("length", "expected"),
    [(100, 1.0), (300, 2.0), (700, 3.0)],
)
def test_reply_delay_depends_on_next_message_length(
    length: int, expected: float
) -> None:
    """Пауза должна ситуативно занимать от одной до трёх секунд."""
    assert reply_delay_seconds("x" * length) == expected
