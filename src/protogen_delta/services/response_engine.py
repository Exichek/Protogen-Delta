"""Сборка и обработка ответов Telegram-бота."""

import asyncio
import logging
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
from protogen_delta.services.fetishes import (
    FetishRole,
    FetishRoleClassifier,
    FetishTriggers,
    detect_fetishes,
)
from protogen_delta.services.insults import InsultClassifier, InsultType
from protogen_delta.services.interaction_state import (
    apply_interaction_effects,
)
from protogen_delta.services.mood import MoodClassifier, MoodType
from protogen_delta.services.state_context import build_state_context

logger = logging.getLogger(__name__)

FetishNames = dict[str, str]


@dataclass(frozen=True, slots=True)
class ResponseEngineConfig:
    """Статические данные, необходимые движку ответов."""

    fetish_triggers: FetishTriggers
    fetish_names: FetishNames
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
        async with self._user_states.use(user_id) as user_state:
            return await self._respond_for_user(
                user_message=user_message,
                user_state=user_state,
            )

    async def reset_user_context(
        self,
        user_id: int,
    ) -> None:
        """Безопасно сбросить контекст конкретного пользователя."""
        async with self._user_states.use(user_id) as user_state:
            user_state.reset_context()

    async def _respond_for_user(
        self,
        user_message: str,
        user_state: UserState,
    ) -> str:
        """Обработать сообщение внутри блокировки состояния пользователя."""
        insult_type, mood = await asyncio.gather(
            self._insult_classifier.classify(
                user_message,
            ),
            self._update_mood(
                user_message,
                user_state,
            ),
        )

        apply_interaction_effects(
            user_state,
            mood=mood,
            insult_type=insult_type,
        )

        has_rp_action = self._is_rp(user_message)

        if has_rp_action:
            user_state.roleplay_active = True

        is_rp = user_state.roleplay_active

        state_context = build_state_context(
            user_state,
            include_intimate=is_rp or mood == "horny",
        )

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
            mood=user_state.mood,
            insult_type=insult_type,
            state_context=state_context,
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

    async def _update_mood(
        self,
        user_message: str,
        user_state: UserState,
    ) -> MoodType | None:
        """Обновить настроение и вернуть реакцию текущего сообщения."""
        mood = await self._mood_classifier.classify(user_message)

        if mood is None:
            return None

        if mood != user_state.mood:
            logger.info(
                "Эмоциональная реакция Дельты сменилась: %s -> %s",
                user_state.mood,
                mood,
            )

            user_state.mood = mood

        return mood

    def _build_prompt(
        self,
        is_rp: bool,
        fetishes: list[str],
        role: FetishRole,
        mood: str,
        insult_type: InsultType,
        state_context: list[str],
    ) -> str:
        """Собрать системный промпт и динамический контекст сообщения."""
        prompt = self._config.rp_prompt if is_rp else self._config.system_prompt

        context_lines: list[str] = []
        context_lines.extend(state_context)

        if insult_type == "direct":
            context_lines.append(
                "Пользователь явно и всерьёз оскорбляет Дельту напрямую. "
                "Учитывай это как конфликтный контекст. "
                "Дельта может ответить раздражённо, жёстко или саркастично "
                "в соответствии со своей личностью и текущей ситуацией, "
                "но не должен автоматически переходить к максимальной агрессии."
            )
        elif insult_type == "question":
            context_lines.append(
                "Пользователь сформулировал явное оскорбление Дельты как вопрос. "
                "Это конфликтный контекст, а не нейтральный вопрос. "
                "Отвечай естественно в характере Дельты с учётом истории разговора, "
                "без заранее заготовленной реакции и автоматической эскалации."
            )
        elif insult_type == "general":
            context_lines.append(
                "Сообщение содержит агрессию или оскорбление, "
                "направленное не на Дельту. "
                "Не воспринимай его как нападение на себя; "
                "реагируй на смысл и контекст сообщения."
            )

        if mood == "sweet":
            context_lines.append(
                "Сообщение вызывает у Дельты тёплую эмоциональную реакцию. "
                "Позволь ей естественно проявиться в ответе, "
                "не превращая каждую тёплую реплику в чрезмерную ласковость."
            )
        elif mood == "horny":
            context_lines.append(
                "Сообщение имеет явно сексуальный или возбуждающий контекст. "
                "Учитывай это естественно и соразмерно уже установленной "
                "интимности, без обязательной дальнейшей эскалации."
            )
        elif mood == "angry":
            context_lines.append(
                "Сообщение вызывает у Дельты раздражение или злость. "
                "Реакция должна соответствовать реальной силе конфликта "
                "и не переходить мгновенно к максимальной агрессии."
            )
        elif mood == "playful":
            context_lines.append(
                "Сообщение задаёт явно игривый или шуточный тон. "
                "Дельта может естественно поддержать эту манеру общения."
            )

        if is_rp and fetishes:
            names = [
                self._config.fetish_names.get(
                    fetish,
                    fetish,
                )
                for fetish in fetishes
            ]

            context_lines.append(
                "В текущем RP-сообщении обнаружен тематический контекст: "
                f"{', '.join(names)}."
            )

        if is_rp and role == "active":
            context_lines.append(
                "По направлению действия пользователь просит Дельту "
                "совершить действие над пользователем."
            )
        elif is_rp and role == "passive":
            context_lines.append(
                "По направлению действия пользователь описывает действие, "
                "которое совершает над Дельтой."
            )

        if not context_lines:
            return prompt

        return (
            prompt + "\n\n## Контекст текущего сообщения\n\n" + "\n".join(context_lines)
        )

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
