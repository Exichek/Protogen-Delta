"""Классификация оскорблений в сообщениях пользователя."""

import logging
from typing import Literal

from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService

logger = logging.getLogger(__name__)

InsultType = Literal["general", "direct", "question", "none"]


class InsultClassifier:
    """Определять тип оскорбления через DeepSeek."""

    def __init__(
        self,
        deepseek: DeepSeekService,
        prompt: str,
    ) -> None:
        """Инициализировать классификатор и сохранить системный промпт."""
        if not prompt.strip():
            raise ValueError("Промпт классификации оскорблений не может быть пустым")

        self._deepseek = deepseek
        self._prompt = prompt

    async def classify(self, user_message: str) -> InsultType:
        """Определить тип оскорбления в сообщении пользователя."""
        try:
            result = await self._deepseek.classify(
                self._prompt,
                user_message,
            )
        except DeepSeekError:
            logger.exception("Ошибка определения типа оскорбления")
            return "none"

        logger.info("Классификация оскорбления: %s", result)

        if result == "general":
            return "general"

        if result == "direct":
            return "direct"

        if result == "question":
            return "question"

        return "none"
