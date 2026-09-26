"""Контракты, реестр и исполнитель внешних инструментов."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)
ToolHandler = Callable[[dict[str, str]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    """Описывать один разрешённый модели внешний инструмент."""

    name: str
    description: str
    parameters: dict[str, str]
    required: tuple[str, ...]
    handler: ToolHandler


class ToolRegistry:
    """Хранить только явно зарегистрированные функции, без eval или shell."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Повтор имени инструмента")

    def schemas(self) -> list[dict[str, Any]]:
        """Преобразовать инструменты в JSON Schema для DeepSeek."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string", "description": description}
                            for name, description in tool.parameters.items()
                        },
                        "required": list(tool.required),
                        "additionalProperties": False,
                    },
                },
            }
            for tool in self.tools.values()
        ]


class ToolExecutor:
    """Проверять аргументы, ограничивать время и размер результата."""

    def __init__(self, registry: ToolRegistry, timeout: float = 12.0) -> None:
        self.registry = registry
        self.timeout = timeout

    async def execute(self, name: str, raw_arguments: str) -> str:
        """Безопасно выполнить зарегистрированный инструмент."""
        started = perf_counter()
        outcome = "error"
        try:
            if name not in self.registry.tools or len(raw_arguments) > 4096:
                raise ValueError("Неизвестный инструмент или слишком большие аргументы")
            tool = self.registry.tools[name]
            args = json.loads(raw_arguments)
            if (
                not isinstance(args, dict)
                or set(args) - set(tool.parameters)
                or set(tool.required) - set(args)
                or any(not isinstance(v, str) or len(v) > 200 for v in args.values())
            ):
                raise ValueError("Некорректные аргументы инструмента")
            result = await asyncio.wait_for(tool.handler(args), self.timeout)
            output = json.dumps({"ok": True, "data": result}, ensure_ascii=False)
            if len(output) > 16000:
                raise ValueError("Ответ источника слишком большой")
            outcome = "ok"
            return output
        except ValueError, KeyError, TypeError:
            return json.dumps({"ok": False, "error": "invalid_arguments_or_data"})
        except Exception:
            # Не передаём модели исключения с URL, ключами или содержимым запросов.
            return json.dumps({"ok": False, "error": "source_unavailable"})
        finally:
            logger.info(
                "Tool name=%s outcome=%s duration=%.3fs",
                name if name in self.registry.tools else "unknown",
                outcome,
                perf_counter() - started,
            )
