"""Сборка системного промпта из тематических секций."""

import re
from dataclasses import dataclass

_LORE_PATTERN = re.compile(
    r"\b(?:протоген\w*|киборг\w*|нанит\w*|визор\w*|кто\s+ты|"
    r"что\s+ты\s+такое|расскажи\s+о\s+себе|ты\s+робот|твой\s+вид|"
    r"твоя\s+история|твой\s+создатель)\b",
    re.IGNORECASE,
)
_BODY_PATTERN = re.compile(
    r"\b(?:тво[еёийю]|тебя|у\s+тебя|дельт\w*)\b.{0,55}"
    r"\b(?:тел\w*|анатоми\w*|внешност\w*|облик\w*|выгляди\w*|"
    r"визор\w*|хвост\w*|уш\w*|лап\w*|когт\w*|крыл\w*|"
    r"член\w*|ху[ейя]\w*|вагин\w*|жоп\w*|груд\w*|сись\w*)\b|"
    r"\b(?:как\s+ты\s+выглядишь|опиши\s+себя)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class PromptSections:
    """Хранить независимые секции личности и RP."""

    core: str
    lore: str = ""
    body: str = ""
    roleplay: str = ""


class PromptComposer:
    """Подключать тяжёлые секции промпта только по контексту запроса."""

    def __init__(self, sections: PromptSections) -> None:
        """Сохранить предварительно загруженные секции промпта."""
        if not sections.core.strip():
            raise ValueError("Основная секция промпта не может быть пустой")
        self._sections = sections

    def compose(
        self,
        user_message: str,
        *,
        is_roleplay: bool,
        has_images: bool = False,
    ) -> str:
        """Собрать минимальный набор секций для текущего сообщения."""
        parts = [self._sections.core]
        if is_roleplay:
            parts.extend(
                (
                    self._sections.lore,
                    self._sections.body,
                    self._sections.roleplay,
                )
            )
            return self._join(parts)

        if _LORE_PATTERN.search(user_message):
            parts.append(self._sections.lore)
        if _BODY_PATTERN.search(user_message) and not has_images:
            parts.append(self._sections.body)
        return self._join(parts)

    @staticmethod
    def _join(parts: list[str]) -> str:
        """Соединить непустые секции без лишних разделителей."""
        return "\n\n".join(part.strip() for part in parts if part.strip())
