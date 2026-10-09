"""Минимальный контракт модели для двух проходов анализа внешности."""

from collections.abc import Collection, Sequence
from typing import Protocol

from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.deepseek import ImageInput


class VisualModel(Protocol):
    async def analyze_visual_features(
        self,
        system_prompt: str,
        user_message: str,
        images: Sequence[ImageInput],
    ) -> str: ...

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[ConversationTurn] = (),
        images: Sequence[ImageInput] = (),
        tool_names: Collection[str] | None = None,
        *,
        json_response: bool = False,
    ) -> str: ...
