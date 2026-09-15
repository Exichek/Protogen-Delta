"""Классификация настроения бота по сообщению пользователя."""

import logging
from typing import Literal

from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService

logger = logging.getLogger(__name__)

MoodType = Literal["sweet", "horny", "angry", "playful"]


class MoodClassifier:
    """Определять настроение бота через DeepSeek."""

    def __init__(
        self,
        deepseek: DeepSeekService,
        prompt: str,
    ) -> None:
        """Инициализировать классификатор настроения."""
        if not prompt.strip():
            raise ValueError("Промпт классификации настроения не может быть пустым")

        self._deepseek = deepseek
        self._prompt = prompt

    async def classify(self, user_message: str) -> MoodType | None:
        """Определить настроение по сообщению пользователя."""
        try:
            result = await self._deepseek.classify(
                self._prompt,
                user_message,
            )
        except DeepSeekError:
            logger.exception("Ошибка определения настроения")
            return None

        logger.info("Классификация настроения: %s", result)

        if result == "sweet":
            return "sweet"

        if result == "horny":
            return "horny"

        if result == "angry":
            return "angry"

        if result == "playful":
            return "playful"

        logger.warning(
            "Неизвестное настроение от модели: %s. Используется playful",
            result,
        )
        return "playful"
