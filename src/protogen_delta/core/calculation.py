"""Ограниченная арифметика Decimal без выполнения пользовательского кода."""

import ast
import re
from decimal import ROUND_FLOOR, Decimal, DecimalException, localcontext

_MAX_VALUE = Decimal("1e100")


def calculate(expression: str) -> str:
    """Вычислить арифметическое выражение с ограничением сложности и величин."""
    if not isinstance(expression, str) or not 1 <= len(expression) <= 200:
        raise ValueError("Выражение должно содержать от 1 до 200 символов")
    source = expression.strip().replace(",", ".").replace("^", "**")
    source = source.replace("×", "*").replace("÷", "/").replace("−", "-")
    if not re.fullmatch(r"[0-9eE.+*/%()\s-]+", source, re.ASCII):
        raise ValueError("Допустимы числа, скобки и операции + − × ÷ // % ^")
    try:
        tree = ast.parse(source, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("Слишком сложное выражение")

        def evaluate(node: ast.AST, depth: int = 0) -> Decimal:
            if depth > 12:
                raise ValueError("Слишком глубокое выражение")
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                raw = ast.get_source_segment(source, node)
                if raw is None or len(raw) > 24:
                    raise ValueError("Слишком длинное число")
                result = Decimal(raw)
            elif isinstance(node, ast.UnaryOp) and isinstance(
                node.op, ast.UAdd | ast.USub
            ):
                result = evaluate(node.operand, depth + 1)
                if isinstance(node.op, ast.USub):
                    result = -result
            elif isinstance(node, ast.BinOp):
                left, right = evaluate(node.left, depth + 1), evaluate(
                    node.right, depth + 1
                )
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    result = left / right
                elif isinstance(node.op, ast.FloorDiv):
                    result = (left / right).to_integral_value(rounding=ROUND_FLOOR)
                elif isinstance(node.op, ast.Mod):
                    result = left - right * (left / right).to_integral_value(
                        rounding=ROUND_FLOOR
                    )
                elif isinstance(node.op, ast.Pow):
                    if abs(right) > 20 or right != right.to_integral_value():
                        raise ValueError(
                            "Степень должна быть целым числом от −20 до 20"
                        )
                    result = left ** int(right)
                else:
                    raise ValueError("Операция не поддерживается")
            else:
                raise ValueError("Ожидалось арифметическое выражение")
            if not result.is_finite() or abs(result) > _MAX_VALUE:
                raise ValueError("Число превышает допустимую величину")
            if result and abs(result) < Decimal("1e-100"):
                raise ValueError("Число слишком мало")
            return result

        with localcontext() as context:
            context.prec = 28
            context.Emax = 200
            context.Emin = -200
            result = evaluate(tree.body)
    except (SyntaxError, DecimalException, OverflowError, RecursionError) as error:
        raise ValueError(
            "Не удалось вычислить выражение: проверь числа и деление на ноль"
        ) from error
    if not result:
        return "0"
    formatted = format(result, "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
