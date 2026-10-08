"""Точка входа Telegram-бота."""

import asyncio
import logging
from contextlib import suppress
from typing import cast

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramAPIError
from openai import AsyncOpenAI

from protogen_delta.config.json_loader import load_json
from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.config.settings import Settings, load_settings
from protogen_delta.core.input_operations import (
    InputOperations,
    MemoryInputsMiddleware,
    ReceiveInputsMiddleware,
)
from protogen_delta.core.logging_config import setup_logging
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.core.runtime_health import PollingHealthMiddleware, RuntimeHealth
from protogen_delta.core.state import BotState
from protogen_delta.core.telegram_commands import set_commands, set_user_commands
from protogen_delta.core.user_state import ContentMode, UserStateStore
from protogen_delta.core.user_statistics import UserStatisticsMiddleware
from protogen_delta.handlers.admin import create_admin_router
from protogen_delta.handlers.adult import create_adult_router
from protogen_delta.handlers.art import create_art_router
from protogen_delta.handlers.creator import create_creator_router
from protogen_delta.handlers.documents import create_document_router
from protogen_delta.handlers.download import create_download_router
from protogen_delta.handlers.e621 import create_e621_router
from protogen_delta.handlers.errors import register_error_handler
from protogen_delta.handlers.help import create_help_router
from protogen_delta.handlers.image_source import create_image_source_router
from protogen_delta.handlers.media import create_media_router
from protogen_delta.handlers.memory import create_memory_router
from protogen_delta.handlers.menu import create_menu_router
from protogen_delta.handlers.reset import create_reset_router
from protogen_delta.handlers.rp import create_rp_router
from protogen_delta.handlers.start import create_start_router
from protogen_delta.handlers.stickers import create_sticker_admin_router
from protogen_delta.handlers.text import create_text_router
from protogen_delta.handlers.unknown_command import create_unknown_command_router
from protogen_delta.handlers.utilities import create_utilities_router
from protogen_delta.handlers.voice import create_voice_router
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.miniapp.tools import MiniAppTools
from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.creator_messages import CreatorMessagesRepository
from protogen_delta.repositories.e621_history import E621HistoryRepository
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.stickers import StickersRepository
from protogen_delta.repositories.user_facts import UserFactsRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.repositories.user_statistics import UserStatisticsRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.audio_understanding import AudioUnderstandingService
from protogen_delta.services.creator_messages import CreatorMessageService
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.e621 import E621Client
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.image_source import ImageSourceService
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.media_download import MediaDownloader
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.menu_sync import synchronize_menus
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.music import MusicRecognitionService
from protogen_delta.services.native_work import NativeWorkPool
from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger
from protogen_delta.services.response_engine import ResponseEngine, ResponseEngineConfig
from protogen_delta.services.speech import SpeechTranscriber
from protogen_delta.services.sticker_pack import StickerPackImporter
from protogen_delta.services.stickers import ContextualStickerService
from protogen_delta.services.tools import ToolExecutor, default_registry
from protogen_delta.services.user_facts import UserFactsService

logger = logging.getLogger(__name__)


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
    health = RuntimeHealth()
    bot.session.middleware.register(PollingHealthMiddleware(health))
    menu_task: asyncio.Task[None] | None = None
    deepseek: DeepSeekService | None = None
    proactive_messenger: ProactiveMessenger | None = None
    native_work = NativeWorkPool(2)
    health.add_diagnostics("native", native_work.snapshot)
    proactive_task: asyncio.Task[None] | None = None
    mini_app_server: MiniAppServer | None = None
    audio_understanding: AudioUnderstandingService | None = None
    transcriber: SpeechTranscriber | None = None

    try:
        if settings.audio_understanding_enabled:
            audio_understanding = AudioUnderstandingService(
                AsyncOpenAI(
                    api_key=settings.audio_api_key,
                    base_url=settings.audio_base_url,
                    timeout=40,
                    max_retries=0,
                ),
                settings.audio_model or "",
                input_data_url=settings.audio_input_data_url,
                native_work=native_work,
            )
            health.add_diagnostics("audio", audio_understanding.snapshot)
        dispatcher = Dispatcher()
        register_error_handler(dispatcher)

        images_repository = ImagesRepository(settings.data_dir)
        users_repository = UsersRepository(settings.data_dir)
        user_state_repository = UserStateRepository(settings.data_dir)
        memories_repository = MemoriesRepository(settings.data_dir)
        creator_messages_repository = CreatorMessagesRepository(settings.data_dir)
        stickers_repository = StickersRepository(settings.data_dir)
        e621_history = E621HistoryRepository(settings.data_dir)
        facts_repository = UserFactsRepository(settings.data_dir)
        statistics_repository = UserStatisticsRepository(settings.data_dir)

        bot_state = BotState()
        user_states = UserStateStore(
            history_limit=settings.conversation_history_limit,
            history_ttl_seconds=settings.conversation_history_ttl_seconds,
            retention_seconds=settings.user_state_retention_seconds,
            persistence=user_state_repository,
        )
        creator_messages = CreatorMessageService(
            bot, creator_messages_repository, user_states
        )
        input_operations = InputOperations()
        dispatcher.message.outer_middleware(ReceiveInputsMiddleware(input_operations))
        dispatcher.message.outer_middleware(
            UserStatisticsMiddleware(
                statistics_repository, user_states, input_operations
            )
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
        repeat_start_prompt = load_prompt(
            "repeat_start_greeting.txt",
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

        deepseek = DeepSeekService(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            disable_thinking=settings.llm_disable_thinking,
            tools=ToolExecutor(
                default_registry(
                    proxy_url=settings.telegram_proxy_url,
                    brave_search_api_key=settings.brave_search_api_key,
                )
            ),
        )

        health.add_diagnostics("llm", deepseek.snapshot)
        memory = MemoryService(
            memories_repository,
            facts=UserFactsService(facts_repository, deepseek),
            creator_messages=creator_messages_repository,
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
            system_prompt=core_prompt,
            rp_prompt=rp_modifier_prompt,
            protogen_lore_prompt=protogen_lore_prompt,
            body_prompt=body_prompt,
            species_prompt=load_prompt("furry_species_reference"),
            adult_conversation_prompt=load_prompt("adult_conversation_style"),
            capabilities_context=(
                "Читаю PDF, DOCX, XLSX и текстовые файлы/код. Рассматриваю фото, "
                "стикеры, GIF, TGS, MP4 и WebM: приложение извлекает несколько "
                "кадров в порядке времени. Это анализ выбранных кадров, не просмотр "
                "каждого мгновения; звук видео автоматически не анализируется. "
                "Распознаю речь из голосовых и аудиофайлов. "
                "Приложение отправляет готовые Telegram-стикеры из размеченного "
                "пака отдельным механизмом после основного текста; это не "
                "рисование и не мысленная или словесная реакция. "
                "Доступность и интервалы задаются текущими настройками. "
                + (
                    "Для анализа музыки и звуков подключена аудиомодель. "
                    if settings.audio_understanding_enabled
                    else "Анализ музыки/тембра отдельной аудиомоделью не подключён. "
                )
                + (
                    "Для сканированных PDF включено распознавание текста. "
                    if settings.pdf_ocr_enabled
                    else ""
                )
                + "Входящие медиа ограничены 20 МБ. /download скачивает публичные "
                "видео до 10 минут/100 МБ и GIF из X. Крупные файлы сжимаются "
                "для отправки в Telegram. /e6 ищет арты; /id показывает Telegram ID. "
                "При вопросах о форматах описывай эти реальные возможности, "
                "а не ограничения отдельно взятой языковой модели."
            ),
        )

        sticker_service = ContextualStickerService(
            bot,
            stickers_repository,
            user_states,
            chance=settings.sticker_reaction_chance,
            cooldown_seconds=settings.sticker_cooldown_seconds,
            min_replies=settings.sticker_min_replies,
        )
        response_engine = ResponseEngine(
            deepseek=deepseek,
            insult_classifier=insult_classifier,
            mood_classifier=mood_classifier,
            fetish_role_classifier=fetish_role_classifier,
            bot_state=bot_state,
            user_states=user_states,
            config=response_engine_config,
            memory=memory,
            creator_id=settings.creator_id,
            creator_messages=creator_messages,
            reply_transform=sticker_service.correct_reply,
            interaction_classifier=InteractionClassifier(
                deepseek, mood_prompt, insult_prompt
            ),
        )
        sticker_importer = StickerPackImporter(bot, stickers_repository)
        if settings.sticker_pack_enabled:
            try:
                async with asyncio.timeout(15):
                    imported = await sticker_importer.sync()
                logger.info(
                    "Sticker pack matched=%d added=%d unknown=%d",
                    imported.matched,
                    imported.added,
                    imported.unknown,
                )
            except (TelegramAPIError, TimeoutError) as error:
                logger.warning(
                    "Не удалось обновить стикерпак (%s); использую локальную разметку",
                    type(error).__name__,
                )

        start_router = create_start_router(
            users_repository=users_repository,
            user_states=user_states,
            sticker_service=sticker_service,
            deepseek=deepseek,
            first_start_prompt=first_start_prompt,
            repeat_start_prompt=repeat_start_prompt,
        )

        async def on_mode_change(user_id: int, mode: ContentMode) -> None:
            async with user_states.use(user_id) as current_state:
                await set_user_commands(bot, user_id, current_state.content_mode)

        help_router = create_help_router(user_states)
        menu_router = create_menu_router(settings.mini_app_url, user_states)
        utilities_router = create_utilities_router(bot)

        art_router = create_art_router(
            images_repository=images_repository,
            art_chat_id=settings.art_chat_id,
            admin_ids=settings.admin_ids,
            sources=ArtSourcesRepository(settings.data_dir, settings.art_chat_id),
            user_states=user_states,
        )

        effective_admin_ids = settings.admin_ids | (
            frozenset({settings.creator_id})
            if settings.creator_id is not None
            else frozenset()
        )
        admin_router = create_admin_router(
            images_repository=images_repository,
            users_repository=users_repository,
            bot_state=bot_state,
            admin_ids=effective_admin_ids,
            user_statistics=statistics_repository,
            runtime_health=health,
        )
        sticker_admin_router = create_sticker_admin_router(
            stickers_repository,
            effective_admin_ids,
            sticker_importer,
        )

        creator_router = create_creator_router(
            bot=bot,
            users_repository=users_repository,
            creator_id=settings.creator_id,
            creator_messages=creator_messages,
        )

        adult_router = create_adult_router(user_states, on_mode_change)
        e621_client = E621Client(
            settings.e621_user_agent,
            proxy_url=settings.telegram_proxy_url,
            request_interval=settings.e621_request_interval_seconds,
        )
        e621_router = create_e621_router(
            e621_client,
            e621_history,
            user_states,
            bot_id=bot.id,
            images_repository=images_repository,
        )

        rate_limiter = UserRateLimiter(
            cooldown_seconds=settings.rate_limit_seconds,
            retention_seconds=settings.rate_limit_retention_seconds,
        )

        reset_router = create_reset_router(
            response_engine,
            users_repository,
            memory,
            on_mode_change=on_mode_change,
            user_statistics=statistics_repository,
            input_operations=input_operations,
        )

        rp_router = create_rp_router(
            response_engine,
        )

        unknown_command_router = create_unknown_command_router()

        text_router = create_text_router(
            response_engine,
            rate_limiter=rate_limiter,
            bot=bot,
            sticker_service=sticker_service,
        )
        media_router = create_media_router(
            response_engine,
            bot,
            rate_limiter=rate_limiter,
            sticker_service=sticker_service,
            native_work=native_work,
        )
        document_router = create_document_router(
            response_engine,
            bot,
            rate_limiter=rate_limiter,
            sticker_service=sticker_service,
            native_work=native_work,
            ocr_enabled=settings.pdf_ocr_enabled,
        )
        transcriber = SpeechTranscriber(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            native_work=native_work,
        )
        health.add_diagnostics("whisper", transcriber.snapshot)
        voice_router = create_voice_router(
            response_engine,
            bot,
            transcriber,
            rate_limiter=rate_limiter,
            sticker_service=sticker_service,
            native_work=native_work,
            audio_understanding=audio_understanding,
            music_recognition=(
                MusicRecognitionService(
                    settings.music_audd_api_token, native_work=native_work
                )
                if settings.music_audd_api_token
                else None
            ),
        )

        for router in (text_router, media_router, document_router, voice_router):
            router.message.middleware(MemoryInputsMiddleware(input_operations))

        dispatcher.include_router(start_router)
        dispatcher.include_router(menu_router)
        dispatcher.include_router(help_router)
        dispatcher.include_router(utilities_router)
        dispatcher.include_router(art_router)
        dispatcher.include_router(admin_router)
        dispatcher.include_router(sticker_admin_router)
        dispatcher.include_router(creator_router)
        dispatcher.include_router(adult_router)
        dispatcher.include_router(e621_router)
        dispatcher.include_router(reset_router)
        dispatcher.include_router(create_memory_router(memory, user_states))
        dispatcher.include_router(rp_router)
        downloader = MediaDownloader(settings.youtube_cookies_file)
        dispatcher.include_router(create_download_router(downloader))
        dispatcher.include_router(
            create_image_source_router(
                bot, ImageSourceService(settings.saucenao_api_key or "")
            )
        )
        dispatcher.include_router(unknown_command_router)
        dispatcher.include_router(media_router)
        dispatcher.include_router(document_router)
        dispatcher.include_router(voice_router)
        dispatcher.include_router(text_router)

        await bot.delete_webhook(
            drop_pending_updates=False,
        )
        await set_commands(bot, settings.mini_app_url)

        # Команда /proactive временно скрыта: на время этого режима фоновые
        # сообщения включаются всем, включая ранее отключившие их в тестах.
        await memories_repository.enable_proactive_for_all()

        if settings.mini_app_server_enabled:
            mini_app_server = MiniAppServer(
                settings.telegram_token,
                user_states,
                host=settings.mini_app_host,
                port=settings.mini_app_port,
                auth_max_age_seconds=settings.mini_app_auth_max_age_seconds,
                response_engine=response_engine,
                on_mode_change=on_mode_change,
                tools=MiniAppTools(bot, downloader),
                native_work=native_work,
                runtime_health=health,
            )
            await mini_app_server.start()

        logger.info("Бот запущен")
        menu_task = asyncio.create_task(
            synchronize_menus(
                users_repository.get_all(),
                lambda user_id: on_mode_change(user_id, "unselected"),
            )
        )

        proactive_messenger = ProactiveMessenger(
            bot=bot,
            deepseek=deepseek,
            repository=memories_repository,
            system_prompt=core_prompt,
            config=ProactiveConfig(
                check_interval_seconds=settings.proactive_check_seconds,
                idle_seconds=settings.proactive_idle_seconds,
                cooldown_seconds=settings.proactive_cooldown_seconds,
            ),
        )
        proactive_task = asyncio.create_task(proactive_messenger.run_forever())

        await dispatcher.start_polling(bot, tasks_concurrency_limit=32)
    finally:
        health.stop()
        if transcriber is not None:
            await transcriber.close()
        if menu_task is not None:
            menu_task.cancel()
            with suppress(asyncio.CancelledError):
                await menu_task
        if proactive_messenger is not None:
            proactive_messenger.stop()
        if proactive_task is not None:
            try:
                await proactive_task
            except Exception:
                logger.exception("Фоновая задача завершилась с ошибкой")
        try:
            if audio_understanding is not None:
                await audio_understanding.close()
            if mini_app_server is not None:
                await mini_app_server.close()
            if deepseek is not None:
                await deepseek.close()
        finally:
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
