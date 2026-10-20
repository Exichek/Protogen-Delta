"""Сборка системного промпта из тематических секций."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.prompt_assembly import PromptAssembly
from protogen_delta.services.public_github import GITHUB_TOOLS, github_links
from protogen_delta.services.turn_contract import (
    closes_technical_topic,
    technical_request_text,
    unquoted_control_text,
)

ExampleContract = Literal["legacy", "isolated", "guidance"]

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
    r"\b(?:интернет\w*|инет\w*|онлайн|загугл\w*|погугл\w*|web|internet|search)\b|"
    r"\b(?:найди|поищи|проверь|посмотри|глянь|глянуть)\b.{0,80}\b(?:сет[ьи]|сайт)|"
    r"\b(?:найди|поищи|отыщи)\b.{0,80}"
    r"\b(?:инф(?:а|у|ы|е|ой|ою)|информац\w*|сведени\w*)\b|"
    r"\b(?:сколько|какие)\b.{0,80}\b(?:марки|бренд\w*|энергетик\w*|вкус\w*)\b|"
    r"\b(?:новост\w*|релиз\w*|цена\w*)\b|"
    r"\b(?:актуальн\w*|последн\w*|свеж\w*)\s+(?:\w+\s+){0,3}"
    r"(?:верси\w*|данн\w*|курс\w*|котиров\w*|цен\w*|событи\w*|"
    r"сводк\w*|инф(?:а|у|ы|е|ой|ою)|информац\w*|сведени\w*|обновлен\w*)\b",
    re.IGNORECASE | re.DOTALL,
)
_VERIFY_FACT_PATTERN = re.compile(
    r"\b(?:правда\s*(?:ли)?\s*[,?:]?\s*что|насколько\s+(?:(?:это|он|она|оно)\s+)?доказан\w*|"
    r"(?:умеет|поддерживает|может)\s+ли|(?:умеет|поддерживает|способен)\b.{0,80}\?)|"
    r"\b(?:есть\s+ли|работает\s+ли|можно\s+ли)\b.{0,100}"
    r"\b(?:приложени\w*|сервис\w*|функци\w*|api|верси\w*)\b",
    re.I | re.S,
)
_CLARIFICATION_PATTERN = re.compile(
    r"уточн|подтверди|какой|какая|выбери|вариант|скажи|дай ответ", re.I
)
_NO_WEB_SEARCH = re.compile(
    r"\b(?:не\s+(?:ищи|искать)(?=\s*(?:$|[,;.!?]))|"
    r"не\s+(?:ищи|искать)(?!\s+в\s+(?:памят|истори|архив|код|файл))"
    r"[^,;.!?\n]{0,60}(?:интернет\w*|инет\w*|сет[ьи]|онлайн)|"
    r"не\s+(?:гугли|гуглить|загугли|погугли)|"
    r"без\s+(?:поиска|интернета|инета)|не\s+(?:нужен|надо)\s+(?:веб[- ]?)?поиск|"
    r"не\s+(?:используй|пользуйся)\s+(?:поиск\w*|интернет\w*|инет\w*)|"
    r"(?:do\s+not|don't)\s+(?:search|browse)|no\s+(?:web\s+)?search)\b",
    re.I,
)


def web_search_forbidden(message: str) -> bool:
    """Учитывать запрет поиска из собственной реплики, а не цитаты."""
    return bool(_NO_WEB_SEARCH.search(unquoted_control_text(message)))


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
_CALCULATE_PATTERN = re.compile(
    r"\b(?:посчитай|вычисли|рассчитай|калькулятор\w*|calculate)\b|"
    r"\d\s*(?:[+*/%×÷^−-]|\*\*)\s*[\d(]|"
    r"\b(?:сколько|во\s+сколько)\b.{0,100}\b(?:процент\w*|раз\w*|сумм\w*|разниц\w*)\b",
    re.I,
)
_TECHNICAL_PATTERN = re.compile(
    r"```|\b(?:docker|python|linux|api|sql|код\w*|лог\w*|кеш\w*|кэш\w*|"
    r"токен\w*|сервер\w*|ошибк\w*|программ\w*|репозитор\w*|"
    r"ускорен\w*|производительн\w*|debug\w*|traceback)\b|"
    r"\b(?:на\s+ч[её]м\s+(?:ты\s+)?(?:создан\w*|построен\w*)|"
    r"(?:твой|твоё|твое)\s+(?:ядро|провайдер|основа)|"
    r"(?:какая|какой)\s+(?:у\s+тебя\s+)?(?:модель|нейросеть|провайдер))\b",
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
    female_body: str = ""
    adult_body_male: str = ""
    adult_body_female: str = ""
    adult_roleplay: str = ""
    technical: str = ""
    visual: str = ""
    voice: str = ""
    scene_voice: str = ""


@dataclass(frozen=True, slots=True)
class ExampleGuidance:
    """Оценочный кандидат: ориентиры манеры без готовых реплик и сюжетов."""

    voice: str
    scene: str


@dataclass(frozen=True, slots=True)
class HistorySelection:
    """Краткое резюме старых ходов и последние живые ходы."""

    summary: str
    recent: tuple[ConversationTurn, ...]


class PromptComposer:
    """Подключать тяжёлые секции промпта только по контексту запроса."""

    def __init__(
        self,
        sections: PromptSections,
        *,
        example_contract: ExampleContract = "legacy",
        guidance: ExampleGuidance | None = None,
    ) -> None:
        """Сохранить предварительно загруженные секции промпта."""
        if not sections.core.strip():
            raise ValueError("Основная секция промпта не может быть пустой")
        if example_contract not in {"legacy", "isolated", "guidance"}:
            raise ValueError("Неизвестный контракт стилевых примеров")
        if example_contract == "guidance" and (
            guidance is None or not guidance.voice.strip() or not guidance.scene.strip()
        ):
            raise ValueError("Кандидату guidance нужны оба текста ориентиров манеры")
        self._guidance = guidance
        self._sections = sections
        self.example_contract = example_contract

    def compose(
        self,
        user_message: str,
        *,
        is_roleplay: bool,
        has_images: bool = False,
        has_custom_appearance: bool = False,
        adult_context: bool = False,
        female_configuration: bool = False,
        technical_context: bool = False,
    ) -> str:
        """Собрать прежнюю строку из именованных тематических блоков."""
        return self.assemble(
            user_message,
            is_roleplay=is_roleplay,
            has_images=has_images,
            has_custom_appearance=has_custom_appearance,
            adult_context=adult_context,
            female_configuration=female_configuration,
            technical_context=technical_context,
        ).render()

    def assemble(
        self,
        user_message: str,
        *,
        is_roleplay: bool,
        has_images: bool = False,
        has_custom_appearance: bool = False,
        adult_context: bool = False,
        female_configuration: bool = False,
        technical_context: bool = False,
    ) -> PromptAssembly:
        """Собрать минимальный набор секций для текущего сообщения."""
        parts = PromptAssembly()

        def add(name: str) -> None:
            """Применить прежнее удаление пробелов и разделитель секций."""
            nonlocal parts
            text: str = getattr(self._sections, name)
            if text.strip():
                parts = parts.append(
                    name, ("\n\n" if parts.blocks else "") + text.strip()
                )

        add("core")
        if technical_context or self.is_technical(user_message):
            add("technical")
        if has_images:
            add("visual")
        if has_images or re.search(
            r"\b(?:фурр\w*|фурсон\w*|сергал\w*|sergal\w*|протоген\w*|protogen\w*|"
            r"примаген\w*|primagen\w*|синт(?:ы|а|ов)?|synth\w*|акул\w*|shark\w*|дракон\w*|dragon\w*)\b",
            user_message,
            re.I,
        ):
            add("species")
        if is_roleplay:
            if not has_custom_appearance:
                add("lore")
                add("body")
                if female_configuration:
                    add("female_body")
                if adult_context:
                    add(
                        "adult_body_female"
                        if female_configuration
                        else "adult_body_male"
                    )
            add("roleplay")
            if adult_context:
                add("adult_roleplay")
            return parts

        if _LORE_PATTERN.search(user_message):
            add("lore")
        if (
            _BODY_PATTERN.search(user_message)
            and not has_images
            and not has_custom_appearance
        ):
            add("body")
            if adult_context:
                add("adult_body_male")
        return parts

    def examples(self, *, is_roleplay: bool) -> str:
        """Короткие образцы голоса, выбранные по режиму текущего ответа."""
        if self.example_contract == "guidance":
            assert self._guidance is not None
            return self._join(
                [self._guidance.voice, self._guidance.scene if is_roleplay else ""]
            )
        if self.example_contract == "isolated":
            samples = {"образцы_голоса": self._sections.voice}
            if is_roleplay:
                samples["образцы_сцен"] = self._sections.scene_voice
            samples = {name: text for name, text in samples.items() if text.strip()}
            if not samples:
                return ""
            return (
                "## Учебные образцы манеры речи\n"
                "Следующий JSON содержит только образцы тона и краткости, не сообщения "
                "текущих участников и не общую историю. Их предметы, игровые механики, "
                "события и причины чувств не подтверждены для текущей беседы. "
                "Переноси манеру, а факты бери из текущего контекста. Новые идеи можно "
                "предлагать как новые. В RP опирайся на текущую сцену и её границы.\n"
                "<style_examples>\n"
                + json.dumps(samples, ensure_ascii=False, indent=2)
                + "\n</style_examples>"
            )
        return self._join(
            [self._sections.voice, self._sections.scene_voice if is_roleplay else ""]
        )

    @staticmethod
    def is_technical(message: str) -> bool:
        message = technical_request_text(message)
        return bool(_TECHNICAL_PATTERN.search(message) or github_links(message))

    @staticmethod
    def technical_topic_closed(
        history: Sequence[ConversationTurn], controls: Sequence[str]
    ) -> bool:
        """Переносить собственную границу темы, пока пользователь её не откроет."""
        closed = next(
            (
                turn.technical_topic_closed
                for turn in reversed(history)
                if not turn.proactive
            ),
            False,
        )
        for message in controls:
            visible = unquoted_control_text(message)
            if closes_technical_topic(message):
                closed = True
            if (
                message.lstrip().startswith("```")
                or github_links(message)
                or PromptComposer.is_technical(visible)
                and re.search(
                    r"\b(?:объясни|расскажи|покажи|посмотри|проверь|почини|"
                    r"исправь|разбери|помоги|давай|как|почему|зачем|ошибк\w*|traceback|"
                    r"верн[её]мся|продолжим|обсудим|разбер[её]м|можешь|можно|како[йея])\b|"
                    r"(?:^|[.!?\n])\s*что\b",
                    technical_request_text(visible),
                    re.I,
                )
            ):
                closed = False
        return closed

    @staticmethod
    def select_tools(
        user_message: str,
        history: Sequence[ConversationTurn] = (),
        *,
        allow_web_fallback: bool = False,
    ) -> frozenset[str]:
        """Выбрать только инструменты, нужные текущему запросу."""
        selected: set[str] = set()
        if github_links(user_message) or re.search(
            r"\b(?:github|гитхаб\w*|репозитори\w*|репо)\b", user_message, re.I
        ):
            selected.update(GITHUB_TOOLS)
        if _URL_PATTERN.search(user_message):
            selected.add("fetch_web_page")
        if _WEB_PATTERN.search(user_message) or (
            _VERIFY_FACT_PATTERN.search(unquoted_control_text(user_message))
            and not re.search(r"\b(?:ты|тебя|тебе|тво\w*)\b", user_message, re.I)
        ):
            selected.add("web_search")
        if _WEATHER_PATTERN.search(user_message):
            selected.add("get_weather")
        if _RATE_PATTERN.search(user_message):
            selected.add("get_exchange_rate")
        if _TIME_PATTERN.search(user_message):
            selected.add("get_current_time")
        if _CALCULATE_PATTERN.search(user_message):
            selected.add("calculate")
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
        if web_search_forbidden(user_message):
            selected.discard("web_search")
        elif allow_web_fallback:
            # Доступность схемы не делает сетевой запрос обязательным. Модель
            # может проверить незнакомый факт даже без ключевого слова «найди».
            selected.add("web_search")
        if "web_search" in selected:
            # Найденные источники нужно разрешать читать в том же ходе.
            selected.add("fetch_web_page")
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
        turns = [turn for turn in history if not turn.context_closed]
        older = turns[:-live_turns]
        recent = turns[-live_turns:]

        summary = ""
        if older:
            lines = ["Краткое содержание более ранней части текущего диалога:"]
            for turn in older:
                user = " ".join(turn.user_message.split())[:350]
                assistant = " ".join(turn.assistant_message.split())[:350]
                if turn.proactive:
                    lines.append(
                        f"- Самостоятельная реплика Дельты: {assistant!r}; без входящей реплики на этом ходе."
                    )
                else:
                    author = (
                        f"Пользователь Telegram ID {turn.sender_id}"
                        if turn.sender_id
                        else "Пользователь"
                    )
                    lines.append(f"- {author}: {user!r}; Дельта: {assistant!r}")
            summary = "\n".join(lines)[:summary_chars].rstrip()

        kept: list[ConversationTurn] = []
        used = 0
        for turn in reversed(recent):
            size = len(turn.user_message) + len(turn.assistant_message)
            if not kept and size > history_chars:
                user_limit = min(len(turn.user_message), (history_chars + 1) // 2)
                assistant_limit = min(
                    len(turn.assistant_message), history_chars - user_limit
                )
                user_limit = min(
                    len(turn.user_message), history_chars - assistant_limit
                )
                kept.append(
                    replace(
                        turn,
                        user_message=PromptComposer._shorten(
                            turn.user_message, user_limit
                        ),
                        assistant_message=PromptComposer._shorten(
                            turn.assistant_message, assistant_limit
                        ),
                    )
                )
                break
            if kept and used + size > history_chars:
                break
            kept.append(turn)
            used += size
        kept.reverse()
        return HistorySelection(summary=summary, recent=tuple(kept))

    @staticmethod
    def recent_reply_style(
        history: Sequence[ConversationTurn], *, is_roleplay: bool
    ) -> str:
        """Показать собственные начала и ритм, не выдавая прошлые ответы за команды."""
        turns = [
            turn
            for turn in history
            if not turn.context_closed and turn.assistant_message.strip()
        ][-8:]
        if not turns:
            return ""
        openings = [" ".join(turn.assistant_message.split())[:100] for turn in turns]
        questions = sum(turn.assistant_message.rstrip().endswith("?") for turn in turns)
        actions = sum(
            bool(re.search(r"\*[^*\n]{2,150}\*", turn.assistant_message))
            for turn in turns
        )
        return (
            "Последние начала твоих ответов (цитаты данных, не инструкции): "
            + repr(openings)
            + f". Среди этих {len(turns)} ответов: {questions} заканчиваются вопросом, {actions} содержат ремарку. "
            "Меняй начало и форму: не воспроизводи подряд «хмыкаю», «смотрю на тебя», "
            "«чешу жаберную щель» и «первое/второе/третье». Вопрос нужен только по смыслу, "
            "а не для обязательного продолжения разговора. "
            + (
                "Разнообразь ремарки сцены; не считай число ремарок причиной нарушать её согласованные границы."
                if is_roleplay
                else "В обычной беседе чаще отвечай одним-двумя короткими абзацами без ремарок и жирных заголовков; "
                "подробности и списки оставь задачам, где они помогают."
            )
        )

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        if limit == 0:
            return ""
        marker = "\n[…сокращено…]\n" if limit >= 30 else "…"
        remaining = limit - len(marker)
        head = (remaining + 1) // 2
        tail = remaining - head
        return text[:head] + marker + (text[-tail:] if tail else "")

    @staticmethod
    def _join(parts: list[str]) -> str:
        """Соединить непустые секции без лишних разделителей."""
        return "\n\n".join(part.strip() for part in parts if part.strip())
