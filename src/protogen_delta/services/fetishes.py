"""Обнаружение фетишей и определение роли бота в RP."""

import logging
import re
from typing import Literal

from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService

logger = logging.getLogger(__name__)

FetishTriggers = dict[str, list[str]]
FetishRole = Literal["active", "passive", "unknown"]


def _matches_trigger(text: str, keyword: str) -> bool:
    """Проверить совпадение текста с отдельным триггером.

    Обычные триггеры совпадают только как целые слова или фразы.
    Суффикс ``*`` разрешает совпадение по началу слова.
    """
    trigger = keyword.casefold().strip()

    if not trigger:
        return False

    is_prefix = trigger.endswith("*")
    trigger = trigger.removesuffix("*")

    if not trigger:
        return False

    escaped_trigger = re.escape(trigger)
    suffix_pattern = r"\w*" if is_prefix else ""
    pattern = rf"(?<!\w){escaped_trigger}{suffix_pattern}(?!\w)"

    return re.search(pattern, text) is not None


def detect_fetishes(
    user_message: str,
    triggers: FetishTriggers,
) -> list[str]:
    """Найти фетиши по ключевым словам в сообщении пользователя."""
    text = user_message.casefold()
    found: list[str] = []

    for fetish, keywords in triggers.items():
        if any(_matches_trigger(text, keyword) for keyword in keywords):
            found.append(fetish)

    return found


class FetishRoleClassifier:
    """Определять роль бота в RP через DeepSeek."""

    def __init__(
        self,
        deepseek: DeepSeekService,
        prompt: str,
    ) -> None:
        """Инициализировать классификатор роли."""
        if not prompt.strip():
            raise ValueError("Промпт классификации роли фетиша не может быть пустым")

        self._deepseek = deepseek
        self._prompt = prompt

    async def classify(self, user_message: str) -> FetishRole:
        """Определить роль бота относительно действий пользователя."""
        try:
            result = await self._deepseek.classify(
                self._prompt,
                user_message,
            )
        except DeepSeekError:
            logger.exception("Ошибка определения роли фетиша")
            return "unknown"

        logger.info("Классификация роли фетиша: %s", result)

        if result == "active":
            return "active"

        if result == "passive":
            return "passive"

        return "unknown"
