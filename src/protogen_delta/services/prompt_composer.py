"""Сборка системного промпта из тематических секций."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from protogen_delta.core.user_state import ConversationTurn

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
_URL_PATTERN = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_WEB_PATTERN = re.compile(
    r"\b(?:интернет\w*|онлайн|загугл\w*|погугл\w*|web|internet|search)\b|"
    r"\b(?:найди|поищи|проверь|посмотри|глянь|глянуть)\b.{0,80}\b(?:сет[ьи]|сайт)|"
    r"\b(?:сколько|какие)\b.{0,80}\b(?:марки|бренд\w*|энергетик\w*|вкус\w*)\b|"
    r"\b(?:актуальн\w*|последн\w*|свеж\w*|новост\w*|релиз\w*|цена\w*)\b",
    re.IGNORECASE | re.DOTALL,
)
_CLARIFICATION_PATTERN = re.compile(
    r"уточн|подтверди|какой|какая|выбери|вариант|скажи|дай ответ", re.I
)
_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(?:\d+[\s,.!]*$|да\b|нет\b|перв\w*\b|втор\w*\b|"
    r"трет\w*\b|именно\b|верно\b|город\b|москва\b|springfield\b)",
    re.I,
)
_WEATHER_PATTERN = re.compile(r"\b(?:погод\w*|температур\w*|прогноз\w*)\b", re.I)
_RATE_PATTERN = re.compile(
    r"\b(?:курс\w*|валют\w*|доллар\w*|евро|рубл\w*|иен\w*|юан\w*|"
    r"usd|eur|rub|jpy|cny)\b",
    re.I,
)
_TIME_PATTERN = re.compile(
    r"\b(?:который\s+час|сколько\s+времени|текущее\s+время|"
    r"сегодняшн(?:яя|юю)\s+дат\w*|какая\s+(?:сегодня\s+)?дата|"
    r"время\s+(?:в|по)\s+[\w-]+)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class PromptSections:
    """Хранить независимые секции личности и RP."""

    core: str
    lore: str = ""
    body: str = ""
    roleplay: str = ""
    species: str = ""


@dataclass(frozen=True, slots=True)
class HistorySelection:
    """Краткое резюме старых ходов и последние живые ходы."""

    summary: str
    recent: tuple[ConversationTurn, ...]


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
        has_custom_appearance: bool = False,
    ) -> str:
        """Собрать минимальный набор секций для текущего сообщения."""
        parts = [self._sections.core]
        if (
            has_images
            or (is_roleplay and has_custom_appearance)
            or re.search(
                r"\b(?:фурр\w*|фурсон\w*|сергал\w*|sergal\w*|протоген\w*|protogen\w*|"
                r"примаген\w*|primagen\w*|синт(?:ы|а|ов)?|synth\w*|акул\w*|shark\w*|дракон\w*|dragon\w*)\b",
                user_message,
                re.I,
            )
        ):
            parts.append(self._sections.species)
        if is_roleplay:
            if not has_custom_appearance:
                parts.extend((self._sections.lore, self._sections.body))
            parts.append(self._sections.roleplay)
            return self._join(parts)

        if _LORE_PATTERN.search(user_message):
            parts.append(self._sections.lore)
        if (
            _BODY_PATTERN.search(user_message)
            and not has_images
            and not has_custom_appearance
        ):
            parts.append(self._sections.body)
        return self._join(parts)

    @staticmethod
    def select_tools(
        user_message: str, history: Sequence[ConversationTurn] = ()
    ) -> frozenset[str]:
        """Выбрать только инструменты, нужные текущему запросу."""
        selected: set[str] = set()
        if _URL_PATTERN.search(user_message):
            selected.add("fetch_web_page")
        if _WEB_PATTERN.search(user_message):
            selected.add("web_search")
        if _WEATHER_PATTERN.search(user_message):
            selected.add("get_weather")
        if _RATE_PATTERN.search(user_message):
            selected.add("get_exchange_rate")
        if _TIME_PATTERN.search(user_message):
            selected.add("get_current_time")
        if (
            not selected
            and history
            and len(user_message) <= 140
            and not re.search(
                r"\b(?:объясни|расскажи|покажи|давай)\b", user_message, re.I
            )
            and _CLARIFICATION_PATTERN.search(history[-1].assistant_message)
            and (
                _FOLLOWUP_PATTERN.search(user_message)
                or (
                    re.fullmatch(r"[A-ZА-ЯЁ][\w ,.-]{1,60}", user_message)
                    and not re.search(
                        r"\b(?:привет|спасибо|здравствуй|стоп|как|объясни|расскажи|покажи)\b",
                        user_message,
                        re.I,
                    )
                )
            )
        ):
            # Уточнение города/региона не содержит слова «погода», но всё ещё
            # должно иметь инструмент. Старые ответы не задают новые полномочия.
            for turn in reversed(history[-2:]):
                selected.update(PromptComposer.select_tools(turn.user_message))
                if selected:
                    break
        return frozenset(selected)

    @staticmethod
    def compact_history(
        history: Sequence[ConversationTurn],
        *,
        live_turns: int = 4,
        summary_chars: int = 2000,
        history_chars: int = 8000,
    ) -> HistorySelection:
        """Сжать старые ходы без отдельного запроса к модели."""
        if live_turns <= 0 or summary_chars <= 0 or history_chars <= 0:
            raise ValueError("Лимиты истории должны быть больше нуля")
        turns = list(history)
        older = turns[:-live_turns]
        recent = turns[-live_turns:]

        summary = ""
        if older:
            lines = ["Краткое содержание более ранней части текущего диалога:"]
            for turn in older:
                user = " ".join(turn.user_message.split())[:350]
                assistant = " ".join(turn.assistant_message.split())[:350]
                lines.append(f"- Пользователь: {user!r}; Дельта: {assistant!r}")
            summary = "\n".join(lines)[:summary_chars].rstrip()

        kept: list[ConversationTurn] = []
        used = 0
        for turn in reversed(recent):
            size = len(turn.user_message) + len(turn.assistant_message)
            if kept and used + size > history_chars:
                break
            kept.append(turn)
            used += size
        kept.reverse()
        return HistorySelection(summary=summary, recent=tuple(kept))

    @staticmethod
    def _join(parts: list[str]) -> str:
        """Соединить непустые секции без лишних разделителей."""
        return "\n\n".join(part.strip() for part in parts if part.strip())
