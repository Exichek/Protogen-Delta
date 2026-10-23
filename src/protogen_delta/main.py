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
from protogen_delta.core.initiative_topics import DEFAULT_TOPICS
from protogen_delta.core.input_operations import (
    InputOperations,
    MemoryInputsMiddleware,
    ReceiveInputsMiddleware,
)
from protogen_delta.core.logging_config import setup_logging
from protogen_delta.core.rate_limiter import MessageTokenBucket, UserRateLimiter
from protogen_delta.core.runtime_health import PollingHealthMiddleware, RuntimeHealth
from protogen_delta.core.shared_dialogue import SharedDialogueAccounts
from protogen_delta.core.state import BotState
from protogen_delta.core.telegram_commands import set_commands, set_user_commands
from protogen_delta.core.user_state import ContentMode, UserStateStore
from protogen_delta.core.user_statistics import UserStatisticsMiddleware
from protogen_delta.handlers.actions import create_actions_router
from protogen_delta.handlers.admin import create_admin_router
from protogen_delta.handlers.adult import create_adult_router
from protogen_delta.handlers.art import create_art_router
from protogen_delta.handlers.code_review import create_code_review_router
from protogen_delta.handlers.creator import create_creator_router
from protogen_delta.handlers.delivery_ledger import DeliveryLedgerMiddleware
from protogen_delta.handlers.dialogue_archive import (
    ArchiveIncomingMiddleware,
    ArchiveOutgoingMiddleware,
    archive_edit,
)
from protogen_delta.handlers.documents import create_document_router
from protogen_delta.handlers.download import create_download_router
from protogen_delta.handlers.e621 import create_e621_router
from protogen_delta.handlers.errors import register_error_handler
from protogen_delta.handlers.favorites import create_favorites_router
from protogen_delta.handlers.help import create_help_router
from protogen_delta.handlers.image_source import create_image_source_router
from protogen_delta.handlers.input_batches import InputBatchMiddleware
from protogen_delta.handlers.llm_budget import LlmUserMiddleware
from protogen_delta.handlers.media import create_media_router
from protogen_delta.handlers.memory import create_memory_router
from protogen_delta.handlers.menu import create_menu_router
from protogen_delta.handlers.proactive_activity import (
    ProactiveIncomingMiddleware,
    ProactiveOutgoingMiddleware,
)
from protogen_delta.handlers.reset import create_reset_router
from protogen_delta.handlers.rp import create_rp_router
from protogen_delta.handlers.safety import AgeSafetyMiddleware
from protogen_delta.handlers.sender import SenderMiddleware
from protogen_delta.handlers.start import create_start_router
from protogen_delta.handlers.stickers import create_sticker_admin_router
from protogen_delta.handlers.text import create_text_router
from protogen_delta.handlers.unknown_command import create_unknown_command_router
from protogen_delta.handlers.utilities import create_utilities_router
from protogen_delta.handlers.voice import create_voice_router
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.miniapp.tools import MiniAppTools
from protogen_delta.repositories.art_sources import ArtSourcesRepository
from protogen_delta.repositories.code_reviews import CodeReviewsRepository
from protogen_delta.repositories.creator_messages import CreatorMessagesRepository
from protogen_delta.repositories.delivery_ledger import DeliveryLedgerRepository
from protogen_delta.repositories.dialogue_archive import DialogueArchiveRepository
from protogen_delta.repositories.e621_history import E621HistoryRepository
from protogen_delta.repositories.favorites import FavoritesRepository
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.initiative import InitiativeRepository
from protogen_delta.repositories.initiative_followups import (
    InitiativeFollowupsRepository,
)
from protogen_delta.repositories.llm_usage import LlmUsageRepository
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.reminders import RemindersRepository
from protogen_delta.repositories.reply_supplements import ReplySupplementLimits
from protogen_delta.repositories.stickers import StickersRepository
from protogen_delta.repositories.user_facts import UserFactsRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.repositories.user_statistics import UserStatisticsRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.repositories.web_api_usage import WebApiUsageRepository
from protogen_delta.services.audio_understanding import AudioUnderstandingService
from protogen_delta.services.capabilities import build_capabilities
from protogen_delta.services.code_review import CodeReviewService
from protogen_delta.services.creator_messages import CreatorMessageService
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.dialogue_archive import DialogueArchiveService
from protogen_delta.services.e621 import E621Client
from protogen_delta.services.favorites import FavoritesService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.image_source import ImageSourceService
from protogen_delta.services.initiative_news import InitiativeNewsService
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.llm_budget_alerts import LlmBudgetAlerts
from protogen_delta.services.market_advisor import MarketAdvisor
from protogen_delta.services.market_data import MarketDataService
from protogen_delta.services.market_http import MarketHTTP
from protogen_delta.services.media_download import MediaDownloader
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.menu_sync import synchronize_menus
from protogen_delta.services.message_reactions import MessageReactionService
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.native_work import NativeWorkPool
from protogen_delta.services.proactive import ProactiveConfig, ProactiveMessenger
from protogen_delta.services.public_github import PublicGithubReader
from protogen_delta.services.release_journal import ReleaseJournal
from protogen_delta.services.reminders import ReminderService
from protogen_delta.services.reply_supplements import ReplySupplements
from protogen_delta.services.response_configuration import load_response_configuration
from protogen_delta.services.response_engine import ResponseEngine
from protogen_delta.services.shazam_music import create_music_recognizer
from protogen_delta.services.source_access import SourceAccess
from protogen_delta.services.speech import SpeechTranscriber
from protogen_delta.services.sticker_pack import StickerPackImporter
from protogen_delta.services.stickers import ContextualStickerService
from protogen_delta.services.tools import ToolExecutor, default_registry
from protogen_delta.services.translation import TranslationService
from protogen_delta.services.user_facts import UserFactsService
from protogen_delta.services.web_search import WebSearchService

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
    github_reader: PublicGithubReader | None = None
    web_search_service: WebSearchService | None = None
    proactive_messenger: ProactiveMessenger | None = None
    native_work = NativeWorkPool(2)
    health.add_diagnostics("native", native_work.snapshot)
    proactive_task: asyncio.Task[None] | None = None
    mini_app_server: MiniAppServer | None = None
    audio_understanding: AudioUnderstandingService | None = None
    transcriber: SpeechTranscriber | None = None
    reminders: ReminderService | None = None
    dialogue_archive: DialogueArchiveService | None = None
    code_reviews: CodeReviewService | None = None
    input_batches: InputBatchMiddleware | None = None
    budget_alerts: LlmBudgetAlerts | None = None
    reply_supplements: ReplySupplements | None = None

    try:
        usage_repository = LlmUsageRepository(
            settings.data_dir,
            settings.llm_daily_request_limit,
            limit_overrides=settings.llm_daily_request_limit_overrides,
            global_daily_limit=settings.llm_global_daily_request_limit,
        )
        budget_alerts = LlmBudgetAlerts(
            bot, usage_repository, settings.llm_budget_alert_user_id
        )
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
            audio_understanding.set_usage_repository(usage_repository)
        dispatcher = Dispatcher()
        dispatcher.message.outer_middleware(LlmUserMiddleware())
        dispatcher.callback_query.outer_middleware(LlmUserMiddleware())
        register_error_handler(dispatcher)

        images_repository = ImagesRepository(settings.data_dir)
        users_repository = UsersRepository(settings.data_dir)
        user_state_repository = UserStateRepository(settings.data_dir)
        shared_dialogue = SharedDialogueAccounts(settings.dialogue_shared_groups)
        delivery_ledger = DeliveryLedgerRepository(settings.data_dir)
        dialogue_archive = DialogueArchiveService(
            DialogueArchiveRepository(settings.data_dir),
            settings.dialogue_archive_user_ids,
            settings.data_dir,
            shared_dialogue,
        )
        bot.session.middleware.register(
            ArchiveOutgoingMiddleware(dialogue_archive, delivery_ledger)
        )
        memories_repository = MemoriesRepository(settings.data_dir)
        favorites = FavoritesService(
            FavoritesRepository(settings.data_dir), settings.favorites_user_ids
        )
        source_access = SourceAccess(
            settings.code_review_user_ids,
            secrets=tuple(
                value
                for value in (
                    settings.telegram_token,
                    settings.deepseek_api_key,
                    settings.audio_api_key,
                    settings.music_audd_api_token,
                    settings.saucenao_api_key,
                    settings.brave_search_api_key,
                    settings.deepseek_search_api_key,
                    settings.alpaca_api_key,
                    settings.alpaca_secret_key,
                    settings.telegram_proxy_url,
                )
                if value is not None
            ),
        )
        bot.session.middleware.register(
            ProactiveOutgoingMiddleware(memories_repository)
        )
        reminders = ReminderService(bot, RemindersRepository(settings.data_dir))
        health.add_diagnostics("reminders", reminders.snapshot)
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
            shared_dialogue=shared_dialogue,
        )
        creator_messages = CreatorMessageService(
            bot, creator_messages_repository, user_states
        )
        input_operations = InputOperations(shared_dialogue)
        dispatcher.message.outer_middleware(DeliveryLedgerMiddleware(delivery_ledger))
        receive_inputs = ReceiveInputsMiddleware(input_operations)
        dispatcher.message.outer_middleware(receive_inputs)
        dispatcher.message.outer_middleware(SenderMiddleware())
        dispatcher.message.outer_middleware(
            ProactiveIncomingMiddleware(
                memories_repository, user_states, input_operations
            )
        )
        dispatcher.message.outer_middleware(
            ArchiveIncomingMiddleware(dialogue_archive, input_operations)
        )
        dispatcher.edited_message.outer_middleware(receive_inputs)
        dispatcher.edited_message.outer_middleware(
            ArchiveIncomingMiddleware(dialogue_archive, input_operations)
        )
        if settings.dialogue_archive_user_ids:
            dispatcher.edited_message.register(archive_edit)
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

        response_engine_config = load_response_configuration(
            settings,
            fetish_triggers=fetish_triggers,
            fetish_names=fetish_names,
            loader=load_prompt,
        )
        core_prompt = response_engine_config.system_prompt

        github_reader = PublicGithubReader(
            settings.data_dir, proxy_url=settings.telegram_proxy_url
        )
        web_search_service = WebSearchService(
            WebApiUsageRepository(
                settings.data_dir,
                settings.web_api_daily_limit,
                settings.brave_monthly_search_limit,
            ),
            deepseek_key=settings.web_search_api_key,
            brave_key=settings.brave_search_api_key,
            proxy_url=settings.telegram_proxy_url,
        )
        deepseek = DeepSeekService(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            verify_web_answers=True,
            disable_thinking=settings.llm_disable_thinking,
            adaptive_thinking=(
                settings.llm_adaptive_thinking and settings.llm_provider == "deepseek"
            ),
            tools=ToolExecutor(
                default_registry(
                    github_reader=github_reader,
                    source_access=source_access,
                    favorites=favorites if settings.favorites_user_ids else None,
                    proxy_url=settings.telegram_proxy_url,
                    brave_search_api_key=settings.brave_search_api_key,
                    web_search_service=web_search_service,
                    dialogue_archive=(
                        dialogue_archive if settings.dialogue_archive_user_ids else None
                    ),
                )
            ),
        )

        health.add_diagnostics("llm", deepseek.snapshot)
        initiative_followups = (
            InitiativeFollowupsRepository(settings.data_dir)
            if settings.proactive_lively_enabled
            else None
        )
        memory = MemoryService(
            memories_repository,
            facts=UserFactsService(facts_repository, deepseek),
            creator_messages=creator_messages_repository,
            dialogue_archive=dialogue_archive,
            favorites=favorites,
            states=user_states,
            delivery_ledger=delivery_ledger,
            inputs=input_operations,
            followups=initiative_followups,
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

        deepseek.set_usage_repository(usage_repository)
        market_advisor = (
            MarketAdvisor(
                MarketDataService(
                    MarketHTTP(
                        settings.alpaca_api_key,
                        settings.alpaca_secret_key,
                        proxy_url=settings.telegram_proxy_url,
                    ),
                    brave_key=settings.brave_search_api_key,
                    web_search_service=web_search_service,
                    proxy_url=settings.telegram_proxy_url,
                ),
                deepseek.for_consultant(),
            )
            if settings.mini_app_server_enabled
            else None
        )

        code_reviews = CodeReviewService(
            bot, deepseek, source_access, CodeReviewsRepository(settings.data_dir)
        )

        async def clear_extra_memory(user_id: int) -> None:
            if market_advisor is not None:
                await market_advisor.clear(user_id)
            await github_reader.clear_user(user_id)
            if code_reviews is not None:
                await code_reviews.repository.clear_delivery(user_id)

        memory.extra_cleanup = clear_extra_memory
        capabilities = build_capabilities(settings)
        release_journal = ReleaseJournal(
            settings.data_dir, settings.recognized_creator_ids, health=health
        )
        reaction_service = (
            MessageReactionService(
                bot,
                user_states,
                chance=settings.message_reaction_chance,
                cooldown_seconds=settings.message_reaction_cooldown_seconds,
            )
            if settings.message_reactions_enabled
            else None
        )
        sticker_service = ContextualStickerService(
            bot,
            stickers_repository,
            user_states,
            chance=settings.sticker_reaction_chance,
            cooldown_seconds=settings.sticker_cooldown_seconds,
            min_replies=settings.sticker_min_replies,
        )
        dispatcher.workflow_data.update({"sticker_service": sticker_service})
        response_engine = ResponseEngine(
            code_reviews=code_reviews,
            github_reader=github_reader,
            deepseek=deepseek,
            appearance_model=(
                deepseek.for_appearance()
                if settings.llm_provider == "deepseek"
                else deepseek
            ),
            insult_classifier=insult_classifier,
            mood_classifier=mood_classifier,
            fetish_role_classifier=fetish_role_classifier,
            bot_state=bot_state,
            user_states=user_states,
            config=response_engine_config,
            memory=memory,
            creator_id=settings.creator_id,
            creator_ids=settings.recognized_creator_ids,
            source_access=source_access,
            creator_messages=creator_messages,
            reply_transform=sticker_service.correct_reply,
            release_journal=release_journal,
            interaction_classifier=InteractionClassifier(
                deepseek,
                mood_prompt,
                insult_prompt,
                favorite_prompt=load_prompt("favorite_selection"),
            ),
        )
        if settings.reply_supplements_enabled:
            reply_supplements = ReplySupplements(
                bot,
                deepseek,
                user_states,
                input_operations,
                ReplySupplementLimits(settings.data_dir),
                style_prompt=core_prompt,
                shared=shared_dialogue,
            )
            response_engine.supplements = reply_supplements
            receive_inputs.on_input = reply_supplements.cancel
            health.add_diagnostics("supplements", reply_supplements.snapshot)
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
            first_start_prompt=first_start_prompt + "\n\n" + capabilities.greeting,
            repeat_start_prompt=repeat_start_prompt + "\n\n" + capabilities.greeting,
        )

        async def on_mode_change(user_id: int, mode: ContentMode) -> None:
            async with user_states.use(user_id) as current_state:
                await set_user_commands(bot, user_id, current_state.content_mode)

        help_router = create_help_router(user_states, settings=settings)
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

        dispatcher.message.outer_middleware(AgeSafetyMiddleware(response_engine))
        adult_router = create_adult_router(
            user_states, on_mode_change, restrict_minor=response_engine.restrict_minor
        )
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
            reminders=reminders,
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
            reaction_service=reaction_service,
        )
        media_router = create_media_router(
            response_engine,
            bot,
            rate_limiter=rate_limiter,
            sticker_service=sticker_service,
            reaction_service=reaction_service,
            native_work=native_work,
        )
        document_router = create_document_router(
            response_engine,
            bot,
            rate_limiter=rate_limiter,
            sticker_service=sticker_service,
            reaction_service=reaction_service,
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
            reaction_service=reaction_service,
            native_work=native_work,
            audio_understanding=audio_understanding,
            music_recognition=create_music_recognizer(
                settings.music_recognition_provider,
                settings.music_audd_api_token,
                native_work,
            ),
        )

        input_batches = InputBatchMiddleware(
            response_engine,
            input_operations,
            delivery_ledger,
            bucket=MessageTokenBucket(
                refill_seconds=(
                    max(2.5, settings.rate_limit_seconds)
                    if settings.rate_limit_seconds
                    else 0
                ),
                privileged=settings.recognized_creator_ids,
            ),
        )
        response_engine.cancel_pending_inputs = input_batches.cancel
        health.add_diagnostics("inputs", input_batches.snapshot)
        for router in (text_router, media_router, document_router, voice_router):
            router.message.middleware(MemoryInputsMiddleware(input_operations))
            router.message.middleware(input_batches)

        dispatcher.include_router(start_router)
        dispatcher.include_router(menu_router)
        dispatcher.include_router(help_router)
        dispatcher.include_router(utilities_router)
        dispatcher.include_router(create_code_review_router(code_reviews))
        actions_router = create_actions_router(reminders)
        actions_router.message.middleware(MemoryInputsMiddleware(input_operations))
        dispatcher.include_router(actions_router)
        dispatcher.include_router(art_router)
        dispatcher.include_router(admin_router)
        dispatcher.include_router(sticker_admin_router)
        dispatcher.include_router(creator_router)
        dispatcher.include_router(adult_router)
        dispatcher.include_router(e621_router)
        dispatcher.include_router(reset_router)
        dispatcher.include_router(create_memory_router(memory, user_states))
        dispatcher.include_router(create_favorites_router(favorites, user_states))
        dispatcher.include_router(rp_router)
        downloader = MediaDownloader(settings.youtube_cookies_file)
        dispatcher.include_router(create_download_router(downloader, sticker_service))
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

        # Сохранённое отключение инициативы не сбрасывается при перезапуске.

        if settings.mini_app_server_enabled:
            mini_app_tools = MiniAppTools(
                bot, downloader, translator=TranslationService(deepseek, native_work)
            )
            health.add_diagnostics("translation", mini_app_tools.translation_snapshot)
            mini_app_server = MiniAppServer(
                settings.telegram_token,
                user_states,
                host=settings.mini_app_host,
                port=settings.mini_app_port,
                auth_max_age_seconds=settings.mini_app_auth_max_age_seconds,
                response_engine=response_engine,
                input_operations=input_operations,
                on_mode_change=on_mode_change,
                tools=mini_app_tools,
                native_work=native_work,
                llm_usage=usage_repository,
                runtime_health=health,
                fetish_triggers=fetish_triggers,
                fetish_names=fetish_names,
                market_advisor=market_advisor,
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
            user_states=user_states,
            creator_ids=settings.recognized_creator_ids,
            system_prompt=core_prompt,
            stickers=sticker_service,
            release_journal=release_journal,
            news=(
                InitiativeNewsService(web_search_service)
                if settings.proactive_news_enabled
                else None
            ),
            initiative=(
                InitiativeRepository(settings.data_dir)
                if settings.proactive_lively_enabled
                else None
            ),
            followups=initiative_followups,
            config=ProactiveConfig(
                discussion_topics=DEFAULT_TOPICS,
                check_interval_seconds=settings.proactive_check_seconds,
                idle_seconds=settings.proactive_idle_seconds,
                idle_max_seconds=settings.proactive_idle_max_seconds,
                cooldown_seconds=settings.proactive_cooldown_seconds,
                sticker_idle_seconds=1800.0,
                sticker_idle_max_seconds=7200.0,
            ),
        )
        proactive_task = asyncio.create_task(proactive_messenger.run_forever())
        await reminders.start()
        budget_alerts.start()
        dialogue_archive.start()
        code_reviews.start()

        await dispatcher.start_polling(bot, tasks_concurrency_limit=32)
    finally:
        health.stop()
        if budget_alerts is not None:
            await budget_alerts.close()
        if input_batches is not None:
            await input_batches.close()
        if reply_supplements is not None:
            await reply_supplements.close()
        if code_reviews is not None:
            await code_reviews.close()
        if reminders is not None:
            await reminders.close()
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
            if github_reader is not None:
                await github_reader.close()
            if web_search_service is not None:
                await web_search_service.close()
            if dialogue_archive is not None:
                await dialogue_archive.close()
        finally:
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
