"""Сборка и обработка ответов Telegram-бота."""

import logging
import random
import re
from dataclasses import dataclass

from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import (
    ConversationTurn,
    UserState,
    UserStateStore,
)
from protogen_delta.services.deepseek import (
    DeepSeekAPIError,
    DeepSeekAuthError,
    DeepSeekConnectionError,
    DeepSeekError,
    DeepSeekRateLimitError,
    DeepSeekService,
    DeepSeekTimeoutError,
)
from protogen_delta.services.emotes import EmoteCategories, ends_with_emote, pick_emote
from protogen_delta.services.fetishes import (
    FetishRole,
    FetishRoleClassifier,
    FetishTriggers,
    detect_fetishes,
)
from protogen_delta.services.greetings import is_greeting
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier

logger = logging.getLogger(__name__)

MoodReplies = dict[str, list[str]]
FetishNames = dict[str, str]


@dataclass(frozen=True, slots=True)
class ResponseEngineConfig:
    """Статические данные, необходимые движку ответов."""

    greetings: list[str]
    insults: list[str]
    question_insult_replies: list[str]
    horny_replies: list[str]
    moods: MoodReplies
    fetish_triggers: FetishTriggers
    fetish_names: FetishNames
    emote_categories: EmoteCategories
    system_prompt: str
    rp_prompt: str


class ResponseEngine:
    """Координировать обработку сообщений и формирование ответов."""

    def __init__(
        self,
        deepseek: DeepSeekService,
        insult_classifier: InsultClassifier,
        mood_classifier: MoodClassifier,
        fetish_role_classifier: FetishRoleClassifier,
        bot_state: BotState,
        user_states: UserStateStore,
        config: ResponseEngineConfig,
    ) -> None:
        """Сохранить сервисы и статические данные движка."""
        if not config.system_prompt.strip():
            raise ValueError("Системный промпт не может быть пустым")

        if not config.rp_prompt.strip():
            raise ValueError("RP-промпт не может быть пустым")

        self._deepseek = deepseek
        self._insult_classifier = insult_classifier
        self._mood_classifier = mood_classifier
        self._fetish_role_classifier = fetish_role_classifier
        self._bot_state = bot_state
        self._user_states = user_states
        self._config = config

    async def respond(
        self,
        user_id: int,
        user_message: str,
    ) -> str:
        """Сформировать готовый ответ на сообщение пользователя."""
        user_state = self._user_states.get(user_id)

        async with user_state.lock:
            return await self._respond_for_user(
                user_message=user_message,
                user_state=user_state,
            )

    async def reset_user_context(
        self,
        user_id: int,
    ) -> None:
        """Безопасно сбросить контекст конкретного пользователя."""
        user_state = self._user_states.get(user_id)

        async with user_state.lock:
            user_state.reset_context()

    async def _respond_for_user(
        self,
        user_message: str,
        user_state: UserState,
    ) -> str:
        """Обработать сообщение внутри блокировки состояния пользователя."""
        greeting_reply = self._handle_greeting(
            user_message,
            user_state,
        )

        if greeting_reply is not None:
            self._remember_turn(
                user_state=user_state,
                user_message=user_message,
                assistant_message=greeting_reply,
            )
            return greeting_reply

        insult_reply = await self._handle_insult(
            user_message,
            user_state,
        )

        if insult_reply is not None:
            self._remember_turn(
                user_state=user_state,
                user_message=user_message,
                assistant_message=insult_reply,
            )
            return insult_reply

        await self._update_mood(
            user_message,
            user_state,
        )

        is_rp = self._is_rp(user_message)

        fetishes = detect_fetishes(
            user_message,
            self._config.fetish_triggers,
        )

        role: FetishRole = "unknown"

        # Определять роль есть смысл только при обнаруженном fetish-контексте.
        if is_rp and fetishes:
            role = await self._fetish_role_classifier.classify(user_message)

            logger.info(
                "Обнаружены фетиши: %s | роль бота: %s",
                ", ".join(fetishes),
                role,
            )

        prompt = self._build_prompt(
            is_rp=is_rp,
            fetishes=fetishes,
            role=role,
        )

        try:
            reply = await self._deepseek.chat(
                system_prompt=prompt,
                user_message=user_message,
                history=tuple(user_state.history),
            )
        except DeepSeekTimeoutError:
            logger.warning("DeepSeek не ответил за установленное время")
            return "Я чёт завис и слишком долго думаю... попробуй ещё раз ≧◡≦"

        except DeepSeekRateLimitError:
            logger.warning("DeepSeek отклонил запрос из-за ограничения частоты")
            return (
                "Меня сейчас слишком сильно дёргают запросами... "
                "дай мне немного времени ≧◡≦"
            )

        except DeepSeekConnectionError:
            logger.warning("Не удалось установить соединение с DeepSeek")
            return "У меня отвалилось соединение... попробуй чуть позже ≧◡≦"

        except DeepSeekAuthError:
            logger.exception("Ошибка доступа к DeepSeek API")
            return "У меня какая-то внутренняя хуйня сломалась... попробуй позже ≧◡≦"

        except DeepSeekAPIError as error:
            logger.exception(
                "DeepSeek вернул ошибку API, HTTP-код: %s",
                error.status_code,
            )
            return "У меня мозги сейчас чудят... попробуй чуть позже ≧◡≦"

        except DeepSeekError:
            logger.exception("Неизвестная ошибка сервиса DeepSeek")
            return "Бля, у тостера что-то сломалось... ≧◡≦"

        except Exception:
            logger.exception("Непредвиденная ошибка при получении ответа от DeepSeek")
            return "Бля, у тостера что-то сломалось... ≧◡≦"

        if not reply:
            reply = "Пустой ответ от DeepSeek"
        elif not reply.strip():
            reply = "DeepSeek промолчал..."

        reply = self._decorate_reply(
            reply=reply,
            is_rp=is_rp,
            fetishes=fetishes,
            user_state=user_state,
        )

        self._register_reply(user_state)

        self._remember_turn(
            user_state=user_state,
            user_message=user_message,
            assistant_message=reply,
        )

        return reply

    def _register_reply(
        self,
        user_state: UserState,
    ) -> None:
        """Учесть ответ глобально и в состоянии пользователя."""
        user_state.register_reply()
        self._bot_state.register_reply()

    @staticmethod
    def _remember_turn(
        user_state: UserState,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Сохранить завершённый ход в истории пользователя."""
        user_state.history.append(
            ConversationTurn(
                user_message=user_message,
                assistant_message=assistant_message,
            )
        )

    def _handle_greeting(
        self,
        user_message: str,
        user_state: UserState,
    ) -> str | None:
        """Вернуть быстрый ответ на одиночное приветствие."""
        if not is_greeting(user_message):
            return None

        if not self._config.greetings:
            return None

        self._register_reply(user_state)

        return random.choice(self._config.greetings)

    async def _handle_insult(
        self,
        user_message: str,
        user_state: UserState,
    ) -> str | None:
        """Вернуть специальный ответ на оскорбление."""
        insult_type = await self._insult_classifier.classify(user_message)

        if insult_type == "question" and self._config.question_insult_replies:
            reply = random.choice(self._config.question_insult_replies)
            emote = pick_emote(
                self._config.emote_categories,
                "BLUSH",
            )

            self._register_reply(user_state)

            return f"{reply} {emote}".rstrip()

        if insult_type == "direct" and self._config.insults:
            reply = random.choice(self._config.insults)
            emote = pick_emote(
                self._config.emote_categories,
                "INSULT",
            )

            self._register_reply(user_state)

            return f"{reply} {emote}".rstrip()

        if insult_type == "general" and self._config.insults:
            self._register_reply(user_state)

            return random.choice(self._config.insults)

        return None

    async def _update_mood(
        self,
        user_message: str,
        user_state: UserState,
    ) -> None:
        """Обновить настроение бота, если классификация успешна."""
        mood = await self._mood_classifier.classify(user_message)

        if mood is None:
            return

        if mood != user_state.mood:
            logger.info(
                "Настроение пользователя сменилось: %s -> %s",
                user_state.mood,
                mood,
            )

            user_state.mood = mood

    def _build_prompt(
        self,
        is_rp: bool,
        fetishes: list[str],
        role: FetishRole,
    ) -> str:
        """Выбрать системный промпт и добавить динамический RP-контекст."""
        if not is_rp:
            return self._config.system_prompt

        prompt = self._config.rp_prompt

        # Сначала добавляем найденный fetish-контекст,
        # как это происходило в старой версии бота.
        if fetishes:
            names = [
                self._config.fetish_names.get(
                    fetish,
                    fetish,
                )
                for fetish in fetishes
            ]

            fetish_text = ", ".join(names)

            if role == "active":
                fetish_text += " (бот актив)"
            elif role == "passive":
                fetish_text += " (бот пассив)"

            prompt += (
                f"\n⚡ Учитывай, что в этом сообщении есть фетиши: "
                f"{fetish_text}. "
                "Отыгрывай их в ролевке откровенно и детально."
            )

        prompt += (
            "\nВАЖНО: Никогда не используй слово 'механический' "
            "для описания тела Дельты. "
            "Его тело синтетическое и почти как живое. "
            "Используй 'синтетические мышцы', "
            "'синтетическая шерсть', "
            "но чаще просто 'руки', 'ноги', 'лапки', 'пальцы'. "
            "Не пиши 'механический член' — просто 'член'. "
            "Не пиши 'снял корпус' — описывай как одежду "
            "или движение тела. "
            "Единственное техно-исключение: визор, "
            "внутренние датчики и вентиляторы охлаждения.\n\n"
            "Базовая одежда Дельты: без одежды. "
            "Протогены не носят штанов, "
            "так что нижняя часть тела открыта. "
            "Дельта может снимать или менять одежду по ходу ролевки, "
            "если инициирует пользователь."
        )

        if role == "active":
            prompt += (
                "\n⚡ В этой ролевке Дельта должен играть роль " "активного партнёра."
            )

        elif role == "passive":
            prompt += (
                "\n⚡ В этой ролевке Дельта должен играть роль " "пассивного партнёра."
            )

        return prompt

    def _decorate_reply(
        self,
        reply: str,
        is_rp: bool,
        fetishes: list[str],
        user_state: UserState,
    ) -> str:
        """Добавить к ответу случайные реплики и эмоуты."""
        mood_lines = self._config.moods.get(
            user_state.mood,
            [],
        )

        if mood_lines and random.random() < 0.3:
            reply += "\n\n" + random.choice(mood_lines)

        if is_rp and fetishes and random.random() < 0.3:
            fetish_names = [
                self._config.fetish_names.get(
                    fetish,
                    fetish,
                )
                for fetish in fetishes
            ]

            names_text = ", ".join(fetish_names)

            tease_lines = [
                (f"Ммм, похоже ты любишь темы: " f"{names_text}… ^w^"),
                (f"Ооо, так вот какие у тебя фетиши — " f"{names_text} >///<"),
                (f"Ты явно возбуждаешься от " f"{names_text}, верно? UwU"),
                (f"Хех, я обожаю играться с " f"{names_text} ;3"),
            ]

            reply += "\n\n" + random.choice(tease_lines)

        if (
            is_rp
            and user_state.reply_count >= 1
            and self._config.horny_replies
            and random.random() < 0.2
        ):
            horny_reply = random.choice(self._config.horny_replies)

            if random.choice([True, False]):
                reply = horny_reply + "\n\n" + reply
            else:
                reply += "\n\n" + horny_reply

        if (
            not is_rp
            and user_state.reply_count >= 3
            and self._config.horny_replies
            and random.random() < 0.15
        ):
            reply += "\n\n" + random.choice(self._config.horny_replies)

        if random.random() < 0.25 and not ends_with_emote(
            reply,
            self._config.emote_categories,
        ):
            emote = pick_emote(
                self._config.emote_categories,
                "NORMAL",
            )

            if emote:
                reply = f"{reply} {emote}"

        return reply

    @staticmethod
    def _is_rp(
        user_message: str,
    ) -> bool:
        """Проверить наличие RP-действия в звёздочках."""
        return bool(
            re.search(
                r"\*[^*]+\*",
                user_message,
            )
        )
