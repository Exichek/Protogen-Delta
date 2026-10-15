"""Именованные блоки сборки без изменения отправляемого текста."""

from dataclasses import dataclass, field
from typing import TypedDict


class BlockMeasure(TypedDict):
    """Только положение и размер блока, без текста и отпечатка содержимого."""

    name: str
    start: int
    end: int
    chars: int


@dataclass(frozen=True, slots=True)
class PromptBlock:
    """Часть готовой строки; разделители принадлежат самому блоку."""

    name: str
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class PromptAssembly:
    """Неизменяемая сборка одного запроса, без общего состояния между чатами."""

    blocks: tuple[PromptBlock, ...] = ()

    def append(self, name: str, text: str) -> "PromptAssembly":
        """Сохранить строку дословно; пустой блок не создаёт разделителя."""
        if not text:
            return self
        return PromptAssembly((*self.blocks, PromptBlock(name, text)))

    def render(self) -> str:
        """Вернуть ровно соединение исходных фрагментов, без нормализации."""
        return "".join(block.text for block in self.blocks)

    def describe(self) -> list[BlockMeasure]:
        """Показать карту символов; не публиковать текст, его хеш или оценку токенов."""
        result: list[BlockMeasure] = []
        position = 0
        for block in self.blocks:
            end = position + len(block.text)
            result.append(
                {
                    "name": block.name,
                    "start": position,
                    "end": end,
                    "chars": end - position,
                }
            )
            position = end
        return result
