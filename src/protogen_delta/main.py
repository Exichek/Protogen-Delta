"""Точка входа Telegram-бота."""

import asyncio
import logging
from typing import cast

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from protogen_delta.config.json_loader import load_json
from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.config.settings import Settings, load_settings
from protogen_delta.core.logging_config import setup_logging
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.core.state import BotState
from protogen_delta.core.telegram_commands import set_commands
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.admin import create_admin_router
from protogen_delta.handlers.art import create_art_router
from protogen_delta.handlers.errors import register_error_handler
from protogen_delta.handlers.help import create_help_router
from protogen_delta.handlers.reset import create_reset_router
from protogen_delta.handlers.rp import create_rp_router
from protogen_delta.handlers.start import create_start_router
from protogen_delta.handlers.text import create_text_router
from protogen_delta.handlers.unknown_command import create_unknown_command_router
from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.response_engine import ResponseEngine, ResponseEngineConfig
from protogen_delta.services.tools import ToolExecutor, default_registry

logger = logging.getLogger(__name__)


def _require_string_list(value: object, name: str) -> list[str]:
    """Проверить, что значение является списком строк."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} должен содержать список строк")

    return cast(list[str], value)


def _require_string_lists(
    value: object,
    name: str,
) -> dict[str, list[str]]:
    """Проверить, что значение является словарём списков строк."""
    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and isinstance(items, list)
        and all(isinstance(item, str) for item in items)
        for key, items in value.items()
    ):
        raise TypeError(f"{name} должен содержать словарь списков строк")

    return cast(dict[str, list[str]], value)


def _require_string_dict(
    value: object,
    name: str,
) -> dict[str, str]:
    """Проверить, что значение является словарём строк."""
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise TypeError(f"{name} должен содержать словарь строк")

    return cast(dict[str, str], value)


def _create_bot(settings: Settings) -> Bot:
    """Создать Telegram-бота с необязательным прокси."""
    if settings.telegram_proxy_url is None:
        return Bot(
            token=settings.telegram_token,
        )

    session = AiohttpSession(
        proxy=settings.telegram_proxy_url,
    )

    return Bot(
        token=settings.telegram_token,
        session=session,
    )


async def main() -> None:
    """Создать зависимости приложения и запустить Telegram polling."""
    settings = load_settings()
    setup_logging(settings.log_level)

    bot = _create_bot(settings)
    deepseek: DeepSeekService | None = None

    try:
        dispatcher = Dispatcher()
        register_error_handler(dispatcher)

        images_repository = ImagesRepository(settings.data_dir)
        users_repository = UsersRepository(settings.data_dir)
        user_state_repository = UserStateRepository(settings.data_dir)

        bot_state = BotState()
        user_states = UserStateStore(
            history_limit=settings.conversation_history_limit,
            retention_seconds=settings.user_state_retention_seconds,
            persistence=user_state_repository,
        )

        start_data = load_json("start_messages.json")
        start_messages = _require_string_list(
            start_data.get("START_MESSAGES", []),
            "START_MESSAGES",
        )

        fetish_triggers = _require_string_lists(
            load_json("fetishes_triggers.json"),
            "fetishes_triggers.json",
        )
        fetish_names = _require_string_dict(
            load_json("fetish_names.json"),
            "fetish_names.json",
        )

        insult_prompt = load_prompt(
            "insult_classification.txt",
        )
        mood_prompt = load_prompt(
            "mood_classification.txt",
        )
        fetish_role_prompt = load_prompt(
            "fetish_role_classification.txt",
        )

        first_start_prompt = load_prompt(
            "start_greeting.txt",
        )

        core_prompt = load_prompt(
            "personality/core.txt",
        )
        protogen_lore_prompt = load_prompt(
            "personality/protogen_lore.txt",
        )
        body_prompt = load_prompt(
            "personality/body.txt",
        )
        rp_modifier_prompt = load_prompt(
            "personality/rp.txt",
        )

        system_prompt = "\n\n".join(
            (
                core_prompt,
                protogen_lore_prompt,
                body_prompt,
            )
        )
        rp_prompt = "\n\n".join(
            (
                system_prompt,
                rp_modifier_prompt,
            )
        )

        deepseek = DeepSeekService(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            tools=ToolExecutor(default_registry(proxy_url=settings.telegram_proxy_url)),
        )

        insult_classifier = InsultClassifier(
            deepseek=deepseek,
            prompt=insult_prompt,
        )
        mood_classifier = MoodClassifier(
            deepseek=deepseek,
            prompt=mood_prompt,
        )
        fetish_role_classifier = FetishRoleClassifier(
            deepseek=deepseek,
            prompt=fetish_role_prompt,
        )

        response_engine_config = ResponseEngineConfig(
            fetish_triggers=fetish_triggers,
            fetish_names=fetish_names,
            system_prompt=system_prompt,
            rp_prompt=rp_prompt,
        )

        response_engine = ResponseEngine(
            deepseek=deepseek,
            insult_classifier=insult_classifier,
            mood_classifier=mood_classifier,
            fetish_role_classifier=fetish_role_classifier,
            bot_state=bot_state,
            user_states=user_states,
            config=response_engine_config,
        )

        start_router = create_start_router(
            users_repository=users_repository,
            user_states=user_states,
            start_messages=start_messages,
            deepseek=deepseek,
            first_start_prompt=first_start_prompt,
        )
        help_router = create_help_router()

        art_router = create_art_router(
            images_repository=images_repository,
            art_chat_id=settings.art_chat_id,
            admin_ids=settings.admin_ids,
            sources=ArtSourcesRepository(settings.data_dir, settings.art_chat_id),
        )

        admin_router = create_admin_router(
            images_repository=images_repository,
            users_repository=users_repository,
            bot_state=bot_state,
            admin_ids=settings.admin_ids,
        )

        rate_limiter = UserRateLimiter(
            cooldown_seconds=settings.rate_limit_seconds,
            retention_seconds=settings.rate_limit_retention_seconds,
        )

        reset_router = create_reset_router(
            response_engine,
            users_repository,
        )

        rp_router = create_rp_router(
            response_engine,
        )

        unknown_command_router = create_unknown_command_router()

        text_router = create_text_router(
            response_engine,
            rate_limiter=rate_limiter,
        )

        dispatcher.include_router(start_router)
        dispatcher.include_router(help_router)
        dispatcher.include_router(art_router)
        dispatcher.include_router(admin_router)
        dispatcher.include_router(reset_router)
        dispatcher.include_router(rp_router)
        dispatcher.include_router(unknown_command_router)
        dispatcher.include_router(text_router)

        await bot.delete_webhook(
            drop_pending_updates=True,
        )
        await set_commands(bot)

        logger.info("Бот запущен")

        await dispatcher.start_polling(bot)
    finally:
        try:
            if deepseek is not None:
                await deepseek.close()
        finally:
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
