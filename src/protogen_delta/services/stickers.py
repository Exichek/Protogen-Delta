"""Редкие контекстные реакции из размеченного стикерпака."""

import logging
import random
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from time import monotonic

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.stickers import StickerEntry, StickersRepository

logger = logging.getLogger(__name__)

_CONTEXT_PATTERNS = {
    "greeting": r"^\s*(?:привет\w*|здравствуй\w*|доброе утро|hi|hello|йоу)\b",
    "coffee": r"\b(?:кофе\w*|coffee|капучино|эспрессо)\b",
    "tired": r"\b(?:устал\w*|сонн\w*|спать|выспаться|sleepy|tired)\b",
    "hungry": r"\b(?:голод\w*|проголод\w*|хочу есть|hungry)\b",
    "food": r"\b(?:ед[ау]|поесть|перекус\w*|кушать|food)\b",
    "sushi": r"\b(?:суши|роллы|sushi)\b",
    "toaster": r"\b(?:тостер\w*|toaster|тост[аы]?|хлеб\w*)\b",
    "laugh": r"\b(?:ахах\w*|хаха\w*|лол|ржу\w*|смешн\w*|lol|rofl|[xх][dд]+)\b|[:;=]-?[dDДд]+|[😂🤣]",
    "oops": r"\b(?:упс|ой|неловк\w*|oops)\b",
    "okay": r"^\s*(?:окей|ок|понял\w*|согласен|okay|ok)\b",
    "oral": r"\b(?:минет\w*|отсос\w*|сос[аёе]\w*|oral)\b",
    "rimming": r"\b(?:римминг\w*|анилингус\w*|rimming)\b",
    "dominance": r"\b(?:доминир\w*|подчини\w*|dominance)\b",
    "humiliation": r"\b(?:униж\w*|humiliation)\b",
    "cuddle": r"\b(?:обним\w*|обня\w*|прижм\w*|cuddle)\b",
    "happy": (
        r"\b(?:спасибо|благодарю|круто|отлично|ништяк|ура|радуюсь)\b|[😊😄🥳]|"
        r"[:=]-?\)+|(?:[а-яё]|\s|^)\){2,}(?=\s|$)"
    ),
    "playful": r"[:;]-?[pр3]|;-?\)|\b(?:uwu|owo)\b|[😜😝]",
    "love": r"[❤💜💕]",
}


def sticker_reply_tags(text: str) -> set[str]:
    """Учитывать эмоцию Дельты в ремарках, а не темы справочного ответа."""
    actions = " ".join(re.findall(r"\*([^*\n]+)\*", text))
    patterns = {
        "laugh": r"\b(?:рассмеял\w*|засмеял\w*|хохот\w*|сме[яё]\w*)\b",
        "happy": r"\b(?:улыбнул\w*|заулыбал\w*|обрадовал\w*)\b",
        "oops": r"\b(?:смутил\w*|покраснел\w*)\b",
    }
    return {
        tag for tag, pattern in patterns.items() if re.search(pattern, actions, re.I)
    }


def sticker_context_tags(text: str) -> set[str]:
    """Найти конкретный повод для реакции без нового запроса к модели."""
    text = re.sub(r"```[\s\S]*?```|`[^`]*`|https?://\S+", "", text)
    return {
        tag
        for tag, pattern in _CONTEXT_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE)
    }


def has_sticker_request(text: str) -> bool:
    """Распознать просьбу показать реакцию, исключая код и обсуждение настройки."""
    text = re.sub(r"```[\s\S]*?```|`[^`]*`|https?://\S+", "", text).lower()
    if not re.search(r"\b(?:стикер\w*|stickers?)\b", text):
        return False
    if re.search(
        r"\b(?:не\s+(?:надо|нужно|хочу|присылай|отправляй|кидай)|"
        r"перестань|отключи|убери|опиши|распознай|добавить|настроить)\b",
        text,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:покажи|пришли|скинь|кинь|отправь|дай|send|show)\b|"
            r"\b(?:есть|умеешь|можешь)\b.{0,45}\bстикер\w*\b|"
            r"\bстикер\w*\b.{0,45}\b(?:есть|умеешь|можешь)\b",
            text,
        )
    )


@dataclass(slots=True)
class _ReactionState:
    replies_since_sticker: int = 0
    last_sent_at: float | None = None
    last_file_unique_id: str | None = None


class ContextualStickerService:
    """Выбирать стикер из текущего состояния без дополнительного вызова LLM."""

    def __init__(
        self,
        bot: Bot,
        repository: StickersRepository,
        user_states: UserStateStore,
        *,
        chance: float = 0.35,
        cooldown_seconds: float = 180.0,
        min_replies: int = 3,
        clock: Callable[[], float] = monotonic,
        random_value: Callable[[], float] = random.random,
        choose: Callable[[list[StickerEntry]], StickerEntry] = random.choice,
    ) -> None:
        if not 0 <= chance <= 1:
            raise ValueError("chance должна быть от 0 до 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds не может быть отрицательным")
        if min_replies <= 0:
            raise ValueError("min_replies должен быть больше нуля")
        self._bot = bot
        self._repository = repository
        self._user_states = user_states
        self._chance = chance
        self._cooldown_seconds = cooldown_seconds
        self._min_replies = min_replies
        self._clock = clock
        self._random_value = random_value
        self._choose = choose
        self._reactions: dict[int, _ReactionState] = {}

    def capabilities_context(self, user_id: int) -> str:
        """Передать модели фактическую доступность пака для текущего пользователя."""
        state = self._user_states.get(user_id)
        available = sum(
            state.content_mode == "adult" or entry.rating == "safe"
            for entry in self._repository.get_all()
        )
        if self._chance == 0 or not available:
            return "Отправка стикеров сейчас отключена или нет доступных реакций."
        return (
            f"Приложение умеет отправлять настоящие Telegram-стикеры: доступны "
            f"{available} размеченных реакций. После текстового ответа приложение "
            "само выбирает и отправляет подходящую реакцию; это отдельный механизм. "
            "Не отрицай эту возможность и не заменяй её рассказом о визоре. "
            "По просьбе показать стикер приложение попробует отправить пример. "
            "Не утверждай, что стикер уже отправлен, до фактической отправки."
        )

    def is_available(self, user_id: int) -> bool:
        """Проверить, включены ли реакции и доступны ли стикеры по возрастному режиму."""
        state = self._user_states.get(user_id)
        return self._chance > 0 and any(
            state.content_mode == "adult" or entry.rating == "safe"
            for entry in self._repository.get_all()
        )

    async def maybe_send(
        self,
        *,
        chat_id: int,
        user_id: int,
        context_tags: Collection[str] = (),
        context_text: str = "",
        reply_text: str = "",
    ) -> bool:
        """Иногда отправить подходящий стикер и сообщить об успехе."""
        reaction = self._reactions.setdefault(user_id, _ReactionState())
        reaction.replies_since_sticker += 1
        requested = has_sticker_request(context_text)
        semantic_tags = sticker_context_tags(context_text)
        greeting = "greeting" in semantic_tags
        if (
            not requested
            and not greeting
            and reaction.replies_since_sticker < self._min_replies
        ):
            logger.debug("Sticker reaction outcome=gap")
            return False

        now = self._clock()
        if reaction.last_sent_at is not None and now - reaction.last_sent_at < (
            min(10.0, self._cooldown_seconds) if requested else self._cooldown_seconds
        ):
            logger.debug("Sticker reaction outcome=cooldown")
            return False
        if self._chance == 0:
            return False

        state = self._user_states.get(user_id)
        tags = {tag.strip().lower() for tag in context_tags if tag.strip()}
        semantic_tags.update(sticker_reply_tags(reply_text))
        tags.update(semantic_tags)
        tags.add(state.mood)
        if state.roleplay_active:
            tags.add("rp")

        entries = [
            entry
            for entry in self._repository.get_all()
            if (tags.intersection(entry.tags) or requested and entry.rating == "safe")
            and (state.content_mode == "adult" or entry.rating == "safe")
            and (entry.rating == "safe" or (tags - {"rp"}).intersection(entry.tags))
        ]
        if not entries:
            logger.debug("Sticker reaction outcome=no_match")
            return False

        if greeting and not requested:
            greeting_entries = [entry for entry in entries if "greeting" in entry.tags]
            entries = greeting_entries or entries

        # Явный повод не теряется за случайным фильтром; интервалы остаются общими.
        explicit_match = any(
            semantic_tags.intersection(entry.tags) for entry in entries
        )
        if (
            not requested
            and not explicit_match
            and self._random_value() >= self._chance
        ):
            logger.debug("Sticker reaction outcome=probability")
            return False

        def score(entry: StickerEntry) -> int:
            return 10 * len(semantic_tags.intersection(entry.tags)) + len(
                tags.intersection(entry.tags)
            )

        best_score = max(map(score, entries))
        entries = [entry for entry in entries if score(entry) == best_score]

        without_repeat = [
            entry
            for entry in entries
            if entry.file_unique_id != reaction.last_file_unique_id
        ]
        selected = self._choose(without_repeat or entries)
        try:
            await self._bot.send_sticker(chat_id=chat_id, sticker=selected.file_id)
        except TelegramAPIError:
            logger.warning("Не удалось отправить контекстный стикер", exc_info=True)
            return False

        reaction.replies_since_sticker = 0
        reaction.last_sent_at = now
        reaction.last_file_unique_id = selected.file_unique_id
        logger.info(
            "Sticker reaction outcome=sent explicit=%s rating=%s requested=%s",
            explicit_match,
            selected.rating,
            requested,
        )
        return True
