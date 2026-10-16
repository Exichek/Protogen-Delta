"""Telegram entities сохраняют текст, код, Unicode и отдельные части ответа."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Message
from test_response_engine import _create_engine

from protogen_delta.core.reply_formatting import format_reply_parts, format_reply_text
from protogen_delta.handlers.delivery import create_reply_delivery


def _content(encoded: bytes, offset: int, length: int) -> str:
    start, end = offset * 2, (offset + length) * 2
    return encoded[start:end].decode("utf-16-le")


def test_styles_have_correct_utf16_offsets_after_emoji() -> None:
    text, entities = format_reply_text("🦈 *Принимаю печеньку.* **Спасибо** и `x * 2`.")
    assert text == "🦈 Принимаю печеньку. Спасибо и x * 2."
    encoded = text.encode("utf-16-le")
    assert [(e.type, _content(encoded, e.offset, e.length)) for e in entities] == [
        ("italic", "Принимаю печеньку."),
        ("bold", "Спасибо"),
        ("code", "x * 2"),
    ]


def test_code_block_is_opaque_and_html_remains_literal() -> None:
    text, entities = format_reply_text(
        '**Код**\n```python\nprint("*улыбаюсь* <b>x</b>")\n```\n<b>не HTML</b>'
    )
    assert text == 'Код\nprint("*улыбаюсь* <b>x</b>")\n\n<b>не HTML</b>'
    assert [e.type for e in entities] == ["bold", "pre"]
    assert entities[-1].language == "python"


def test_code_language_with_symbols_still_protects_body() -> None:
    text, entities = format_reply_text("```c#\n**literal code**\n```")
    assert text == "**literal code**\n"
    assert len(entities) == 1 and entities[0].type == "pre"
    assert entities[0].language == "c#"


@pytest.mark.parametrize(
    "text",
    [
        "2 * 3 * 4",
        "a * b * c",
        "foo * bar * baz",
        "*2 + x*",
        "*.py",
        "snake_case",
        "*незакрытая ремарка",
        "```python\n*код*",
        "`*незакрытый код*",
        "~~~python\n*код*\n~~~",
        "```inline *код*```",
        "```python\n*код*\n``` trailing text",
        "Обычный текст <tag> & x",
    ],
)
def test_ambiguous_markup_and_math_remain_literal(text: str) -> None:
    assert format_reply_text(text) == (text, [])


def test_long_split_code_is_not_interpreted_as_roleplay() -> None:
    parts = [
        "**Введение**\n```python\n*код*",
        "*ещё код*",
        "*последний код*\n```",
        "*Киваю.*",
    ]
    formatted = format_reply_parts(parts)
    assert formatted[:3] == [(part, []) for part in parts[:3]]
    assert formatted[-1][0] == "Киваю." and formatted[-1][1][0].type == "italic"


def test_delivery_sends_entities_but_retains_original_reply_for_caller() -> None:
    async def run() -> None:
        message = AsyncMock(spec=Message)
        message.answer = AsyncMock()
        delivery = create_reply_delivery(message, None)
        original = "*Улыбаюсь.* Спасибо."
        engine, _, model, *_ = _create_engine()
        model.chat.return_value = original
        await engine.respond_and_deliver(42, "Привет", delivery)
        args = message.answer.await_args
        assert args.args == ("Улыбаюсь. Спасибо.",)
        assert args.kwargs["parse_mode"] is None
        assert args.kwargs["entities"][0].type == "italic"
        assert engine._user_states.get(42).history[-1].assistant_message == original

    asyncio.run(run())
