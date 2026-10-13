"""Арифметический инструмент модели, не имеющий доступа к файлам и сети."""

from protogen_delta.core.calculation import calculate


async def calculate_expression(args: dict[str, str]) -> dict[str, str]:
    expression = args["expression"]
    return {
        "expression": expression,
        "result": calculate(expression),
        "precision": "28",
    }
