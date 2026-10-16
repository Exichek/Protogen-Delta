"""Редкие реакции Telegram по поводу текущей реплики и состоянию Дельты."""

import asyncio
import logging
import math
import random
import re
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message, ReactionTypeEmoji

from protogen_delta.core.reply_capture import current_reply_sources
from protogen_delta.core.reply_outcome import reply_error
from protogen_delta.core.user_state import UserState, UserStateStore
from protogen_delta.services.stickers import sticker_context_tags, sticker_reply_tags

logger = logging.getLogger(__name__)
_PROTECTED = re.compile(
    r'```[\s\S]*?```|`[^`]*`|https?://\S+|«[^»]*»|“[^”]*”|"[^"\n]*"|(?m:^\s*>[^\n]*)'
)
_NEGATIVE = re.compile(
    r"\b(?:груст\w*|больно|плохо|страшно|тревож\w*|пережива\w*|умер\w*|"
    r"смерт\w*|увол\w*|болею|заболел\w*|потерял\w*|ненавижу|"
    r"не\s+(?:получилось|заработало|смешно|весело|рад\w*|люблю|нравится))\b",
    re.I,
)
_DECLINE = re.compile(
    r"\b(?:без\s+реакци\w*|не\s+(?:ставь|ставить|нужны|надо)\s+реакци\w*|"
    r"реакци\w*\s+не\s+(?:нужны|надо)|перестань\s+реагировать)\b",
    re.I,
)


def choose_message_reaction(text: str, reply: str, state: UserState) -> str | None:
    """Одного накопленного настроения недостаточно: нужен повод в этой реплике."""
    visible = _PROTECTED.sub(" ", text).strip()
    answer = _PROTECTED.sub(" ", reply).strip()
    if not visible or not answer or _DECLINE.search(visible) or state.mood == "horny":
        return None
    memory = state.emotions.memory
    warmth = state.emotions.warmth
    if _NEGATIVE.search(visible):
        if re.search(r"\b(?:сочувствую|жаль|поддерж\w*|рядом|обнимаю)\b", answer, re.I):
            return "😢" if warmth >= 0.1 or memory.sadness >= 0.1 else None
        return None
    if state.emotions.irritation >= 0.6 or state.mood == "angry":
        return None
    if re.match(r"(?:нет\b|не\s+(?:хочу|буду|согласен)\b|отказываюсь\b)", answer, re.I):
        return None
    tags = sticker_context_tags(visible)
    answer_tags = sticker_context_tags(answer) | sticker_reply_tags(answer)
    if re.search(r"\b(?:не люблю|не обнимай|не целуй|не смешно)\b", visible, re.I):
        return None
    if tags.intersection({"love", "cuddle", "kiss"}) and (
        warmth >= 0.1 or state.mood == "sweet"
    ):
        return "❤"
    if "laugh" in tags and (
        "laugh" in answer_tags or state.emotions.playfulness >= 0.2
    ):
        return "🤣"
    if re.search(r"\b(?:получилось|заработало|победил\w*|сдал\w*)\b", visible, re.I):
        if "happy" in answer_tags and (memory.joy >= 0.2 or memory.confidence >= 0.15):
            return "🔥"
    if (
        "happy" in tags
        and "happy" in answer_tags
        and (memory.joy >= 0.2 or warmth >= 0.1)
    ):
        return "👍"
    if (
        memory.curiosity >= 0.2
        and re.search(
            r"\b(?:интересн\w*|что думаешь|как считаешь|загадк\w*)\b", visible, re.I
        )
        and re.search(
            r"\b(?:интересн\w*|сомневаюсь|не уверен|надо подумать)\b", answer, re.I
        )
    ):
        return "🤔"
    return None


@dataclass(slots=True)
class _ReactionState:
    last_attempt: float = -math.inf
    messages: int = 3
    seen: deque[int] = field(default_factory=lambda: deque(maxlen=32))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MessageReactionService:
    """Побочное действие не меняет ответ и не запускает дополнительную генерацию."""

    def __init__(
        self,
        bot: Bot,
        states: UserStateStore,
        *,
        chance: float = 0.35,
        cooldown_seconds: float = 120,
        clock: Callable[[], float] = monotonic,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if not math.isfinite(chance) or not 0 <= chance <= 1:
            raise ValueError("Вероятность реакции должна быть от 0 до 1")
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
            raise ValueError("Пауза между реакциями должна быть неотрицательной")
        self._bot, self._states = bot, states
        self._chance, self._cooldown = chance, cooldown_seconds
        self._clock, self._random = clock, random_value
        self._recent: OrderedDict[int, _ReactionState] = OrderedDict()

    async def maybe_react(
        self, message: Message, reply: str, user_id: int
    ) -> str | None:
        """Вернуть эмодзи только после подтверждения Telegram; пачку не оценивать целиком."""
        sources = current_reply_sources.get()
        if (
            self._chance == 0
            or reply_error.get()
            or message.chat.type != "private"
            or message.chat.id != user_id
            or message.from_user is None
            or message.from_user.id != user_id
            or message.from_user.is_bot
            or message.forward_origin is not None
            or message.message_id <= 0
            or sources is not None
            and len(sources) != 1
        ):
            return None
        text = message.text or message.caption or ""
        if message.sticker is not None and message.sticker.emoji:
            text += " " + message.sticker.emoji
        if text.startswith("/"):
            return None
        recent = self._recent.setdefault(message.chat.id, _ReactionState())
        self._recent.move_to_end(message.chat.id)
        if len(self._recent) > 2048:
            self._recent.popitem(last=False)
        async with recent.lock:
            if message.message_id in recent.seen:
                return None
            recent.seen.append(message.message_id)
            recent.messages += 1
            now = self._clock()
            if now - recent.last_attempt < self._cooldown or recent.messages < 3:
                return None
            emoji = choose_message_reaction(text, reply, self._states.get(user_id))
            if emoji is None or self._random() >= self._chance:
                return None
            # Не повторять попытку при неизвестном сетевом результате.
            recent.last_attempt, recent.messages = now, 0
            try:
                async with asyncio.timeout(3):
                    accepted = await self._bot.set_message_reaction(
                        chat_id=message.chat.id,
                        message_id=message.message_id,
                        reaction=[ReactionTypeEmoji(emoji=emoji)],
                        is_big=False,
                    )
            except TelegramAPIError, TimeoutError:
                logger.info("Telegram reaction outcome=unavailable")
                return None
            if accepted is not True:
                return None
            logger.info("Telegram reaction outcome=sent emoji=%s", emoji)
            return emoji
