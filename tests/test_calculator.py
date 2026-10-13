"""Арифметика, ограничения сложности и реальный обмен инструментом модели."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from test_tool_loop import response

import protogen_delta.services.deepseek as deepseek_module
from protogen_delta.core.calculation import calculate
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.prompt_composer import PromptComposer
from protogen_delta.services.tools import ToolExecutor, default_registry


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(25+17)*3", "126"),
        ("0.1+0.2", "0.3"),
        ("0,1 + 0,2", "0.3"),
        ("-5//3", "-2"),
        ("-5%3", "1"),
        ("5%-3", "-1"),
        ("2^10", "1024"),
        ("2**-3", "0.125"),
        ("1e3/4", "250"),
        ("+5", "5"),
        ("10−3×2", "4"),
        ("10÷4", "2.5"),
        ("-0", "0"),
        ("(1.269+0.931)/1.269", "1.733648542159180457052797478"),
        ("1/3", "0.3333333333333333333333333333"),
    ],
)
def test_decimal_arithmetic(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        " " * 201,
        "1/0",
        "1%0",
        "1//0",
        "0**-1",
        "2**21",
        "2**0.5",
        "1e101",
        "1e100*10",
        "1e-101",
        "1e100**20",
        "9" * 25,
        "__import__('os').system('x')",
        "open('secret')",
        "x+1",
        "[1,2]",
        "True",
        "(1).__class__",
        "1<<2",
        "1 and 2",
        "lambda: 1",
        "[x for x in range(2)]",
        "1;2",
        "1+",
        "9**9**9",
        "1+" * 40 + "1",
        "-" * 20 + "1",
        "1_000",
        "1j",
        "NaN",
        "inf",
        "0x10",
        "1\x00+2",
    ],
)
def test_invalid_or_unbounded_expression(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


@pytest.mark.parametrize(
    "message",
    [
        "Посчитай 2+3",
        "2-1",
        "Сколько будет 0.1 + 0.2?",
        "10÷4",
        "Во сколько раз 0.931 больше 0.803?",
        "Вычисли долю от суммы",
    ],
)
def test_calculator_is_selected_for_arithmetic(message: str) -> None:
    assert "calculate" in PromptComposer.select_tools(message)


def test_model_uses_calculator_result(monkeypatch: pytest.MonkeyPatch) -> None:
    first, final = response(1, json.dumps({"expression": "0.1+0.2"})), response()
    first.choices[0].message.tool_calls[0].function.name = "calculate"
    final.choices[0].message.content = "Получилось 0.3"
    client = Mock()
    client.chat.completions.create = AsyncMock(side_effect=[first, final])
    monkeypatch.setattr(deepseek_module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "test", "https://unused.invalid", "test", tools=ToolExecutor(default_registry())
    )
    assert (
        asyncio.run(
            service.chat("system", "Посчитай 0.1+0.2", tool_names={"calculate"})
        )
        == "Получилось 0.3"
    )
    messages = client.chat.completions.create.await_args.kwargs["messages"]
    result = next(json.loads(m["content"]) for m in messages if m["role"] == "tool")
    assert result["data"]["result"] == "0.3"
    assert [
        s["function"]["name"]
        for s in client.chat.completions.create.await_args.kwargs["tools"]
    ] == ["calculate"]
