"""Классификация краткосрочной эмоциональной реакции Дельты."""

import logging
from typing import Literal

from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService

logger = logging.getLogger(__name__)

MoodType = Literal[
    "sweet",
    "horny",
    "angry",
    "playful",
    "neutral",
]


class MoodClassifier:
    """Определять краткосрочную эмоциональную реакцию через DeepSeek."""

    def __init__(
        self,
        deepseek: DeepSeekService,
        prompt: str,
    ) -> None:
        """Инициализировать классификатор эмоциональной реакции."""
        if not prompt.strip():
            raise ValueError("Промпт классификации настроения не может быть пустым")

        self._deepseek = deepseek
        self._prompt = prompt

    async def classify(self, user_message: str) -> MoodType | None:
        """Определить эмоциональную реакцию на сообщение пользователя."""
        try:
            result = await self._deepseek.classify(
                self._prompt,
                user_message,
            )
        except DeepSeekError:
            logger.exception("Ошибка определения настроения")
            return None

        logger.info("Классификация эмоциональной реакции: %s", result)

        if result == "sweet":
            return "sweet"

        if result == "horny":
            return "horny"

        if result == "angry":
            return "angry"

        if result == "playful":
            return "playful"

        if result == "neutral":
            return "neutral"

        logger.warning(
            "Неизвестная эмоциональная реакция от модели: %s. " "Используется neutral",
            result,
        )
        return "neutral"
