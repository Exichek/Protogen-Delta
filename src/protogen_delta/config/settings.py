"""Настройки приложения и загрузка переменных окружения."""

import json
import os
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_E621_USER_AGENT = (
    "ProtogenDelta/0.1 (by Exichek; " "https://github.com/Exichek/Protogen-Delta)"
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки приложения, загружаемые из переменных окружения."""

    telegram_token: str
    deepseek_api_key: str
    art_chat_id: int
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    llm_provider: str = "deepseek"
    llm_disable_thinking: bool = True
    llm_adaptive_thinking: bool = True
    mini_app_url: str | None = None
    mini_app_server_enabled: bool = False
    mini_app_host: str = "127.0.0.1"
    mini_app_port: int = 8080
    mini_app_auth_max_age_seconds: int = 3600
    message_reactions_enabled: bool = True
    reply_supplements_enabled: bool = True
    message_reaction_chance: float = 0.35
    message_reaction_cooldown_seconds: float = 120.0
    sticker_reaction_chance: float = 0.35
    sticker_cooldown_seconds: float = 180.0
    sticker_min_replies: int = 3
    sticker_pack_enabled: bool = True
    pdf_ocr_enabled: bool = False
    audio_understanding_enabled: bool = False
    audio_api_key: str | None = None
    audio_base_url: str | None = None
    audio_model: str | None = None
    audio_input_data_url: bool = False
    music_audd_api_token: str | None = None
    music_recognition_provider: str = "shazam"
    saucenao_api_key: str | None = None
    telegram_proxy_url: str | None = None
    youtube_cookies_file: Path | None = None
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    admin_ids: frozenset[int] = frozenset()
    creator_id: int | None = None
    creator_ids: frozenset[int] = frozenset()
    code_review_user_ids: frozenset[int] = frozenset()
    favorites_user_ids: frozenset[int] = frozenset()
    brave_search_api_key: str | None = None
    deepseek_search_enabled: bool = True
    deepseek_search_api_key: str | None = field(default=None, repr=False)
    web_api_daily_limit: int = 200
    brave_monthly_search_limit: int = 1000
    alpaca_api_key: str | None = field(default=None, repr=False)
    alpaca_secret_key: str | None = field(default=None, repr=False)
    e621_user_agent: str = DEFAULT_E621_USER_AGENT
    e621_request_interval_seconds: float = 1.0
    rate_limit_seconds: float = 2.0
    llm_daily_request_limit: int = 300
    llm_global_daily_request_limit: int = 3000
    llm_budget_alert_user_id: int | None = field(default=None, repr=False)
    llm_daily_request_limit_overrides: dict[int, int] = field(
        default_factory=dict, repr=False
    )
    rate_limit_retention_seconds: float = 300.0
    conversation_history_limit: int = 8
    conversation_history_ttl_seconds: float = 604800.0
    dialogue_archive_user_ids: frozenset[int] = frozenset()
    dialogue_shared_groups: tuple[frozenset[int], ...] = ()
    user_state_retention_seconds: float = 86400.0
    proactive_check_seconds: float = 30.0
    proactive_lively_enabled: bool = True
    proactive_news_enabled: bool = True
    proactive_idle_seconds: float = 5400.0
    proactive_idle_max_seconds: float = 10800.0
    proactive_cooldown_seconds: float = 72000.0
    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    @property
    def web_search_api_key(self) -> str | None:
        """Не отправлять ключ стороннего совместимого LLM в поисковый API DeepSeek."""
        if not self.deepseek_search_enabled:
            return None
        if self.deepseek_search_api_key:
            return self.deepseek_search_api_key
        if self.llm_provider == "deepseek" and self.deepseek_base_url.rstrip("/") in {
            "https://api.deepseek.com",
            "https://api.deepseek.com/v1",
        }:
            return self.deepseek_api_key
        return None

    @property
    def web_search_available(self) -> bool:
        return bool(self.web_search_api_key or self.brave_search_api_key)

    @property
    def llm_api_key(self) -> str:
        """Вернуть ключ выбранного OpenAI-совместимого LLM-провайдера."""
        return self.deepseek_api_key

    @property
    def recognized_creator_ids(self) -> frozenset[int]:
        return self.creator_ids | (
            frozenset({self.creator_id}) if self.creator_id is not None else frozenset()
        )

    @property
    def llm_base_url(self) -> str:
        """Вернуть базовый URL выбранного LLM-провайдера."""
        return self.deepseek_base_url

    @property
    def llm_model(self) -> str:
        """Вернуть имя модели выбранного LLM-провайдера."""
        return self.deepseek_model


def _parse_admin_ids(value: str) -> frozenset[int]:
    """Преобразовать строку Telegram ID через запятую в множество чисел."""
    if not value.strip():
        return frozenset()

    try:
        return frozenset(
            int(user_id.strip()) for user_id in value.split(",") if user_id.strip()
        )
    except ValueError as error:
        raise ValueError(
            "ADMIN_IDS должен содержать Telegram ID через запятую"
        ) from error


def _parse_llm_limit_overrides(value: str) -> dict[int, int]:
    """Загрузить закрытую карту квот без раскрытия ID в сообщении об ошибке."""
    name = "LLM_DAILY_REQUEST_LIMIT_OVERRIDES"

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        """Не принимать неоднозначное повторное значение одного ключа."""
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(name)
            result[key] = item
        return result

    try:
        raw = json.loads(value or "{}", object_pairs_hook=unique_pairs)
        if not isinstance(raw, dict):
            raise ValueError(name)
        result: dict[int, int] = {}
        for owner, limit in raw.items():
            if (
                not owner.isascii()
                or not owner.isdecimal()
                or owner.startswith("0")
                or not 0 < int(owner) < 2**63
                or type(limit) is not int
                or not 0 < limit < 2**63
            ):
                raise ValueError(name)
            result[int(owner)] = limit
        return result
    except ValueError, TypeError:
        raise ValueError(
            name
            + " должен быть JSON-объектом с уникальными положительными ID и целыми квотами"
        ) from None


def _parse_positive_int(
    value: str,
    name: str,
) -> int:
    """Преобразовать строку в положительное целое число."""
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} должен быть целым числом") from error

    if result <= 0:
        raise ValueError(f"{name} должен быть больше нуля")

    return result


def _parse_archive_ids(
    value: str, name: str = "DIALOGUE_ARCHIVE_USER_IDS"
) -> frozenset[int]:
    """Архивировать только явно выбранные личные аккаунты."""
    ids = frozenset(
        _parse_positive_int(item.strip(), name)
        for item in value.split(",")
        if item.strip()
    )
    if any(user_id >= 2**63 for user_id in ids):
        raise ValueError(name + " содержит слишком большой ID")
    return ids


def _parse_dialogue_groups(value: str) -> tuple[frozenset[int], ...]:
    """Группы через точку с запятой; один аккаунт не входит в две группы."""
    groups = tuple(
        _parse_archive_ids(part, "DIALOGUE_SHARED_GROUPS")
        for part in value.split(";")
        if part.strip()
    )
    seen: set[int] = set()
    for group in groups:
        if len(group) < 2 or seen.intersection(group):
            raise ValueError(
                "DIALOGUE_SHARED_GROUPS: нужны непересекающиеся группы от двух ID"
            )
        seen.update(group)
    return groups


def _parse_non_negative_float(
    value: str,
    name: str,
) -> float:
    """Преобразовать строку в конечное неотрицательное число."""
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} должен быть числом") from error

    if not isfinite(result):
        raise ValueError(f"{name} должен быть конечным числом")

    if result < 0:
        raise ValueError(f"{name} не может быть отрицательным")

    return result


def _parse_positive_float(
    value: str,
    name: str,
) -> float:
    """Преобразовать строку в положительное число."""
    result = _parse_non_negative_float(value, name)

    if result == 0:
        raise ValueError(f"{name} должен быть больше нуля")

    return result


def _parse_bool(value: str, name: str) -> bool:
    """Преобразовать распространённое текстовое значение в bool."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} должен быть true или false")


def load_settings() -> Settings:
    """Загрузить настройки приложения из переменных окружения."""
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_TOKEN")
    llm_provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    music_provider = os.getenv("MUSIC_RECOGNITION_PROVIDER", "shazam").strip().lower()
    if music_provider not in {"shazam", "audd", "off"}:
        raise ValueError("MUSIC_RECOGNITION_PROVIDER должен быть shazam, audd или off")
    if llm_provider not in {"deepseek", "openai-compatible"}:
        raise ValueError("LLM_PROVIDER должен быть deepseek или openai-compatible")

    deepseek_api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    art_chat_id_raw = os.getenv("ART_CHAT_ID")

    default_base_url = "https://api.deepseek.com" if llm_provider == "deepseek" else ""
    default_model = "deepseek-flash" if llm_provider == "deepseek" else ""
    deepseek_base_url = (
        os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or default_base_url
    ).strip()
    deepseek_model = (
        os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or default_model
    ).strip()
    llm_disable_thinking = _parse_bool(
        os.getenv(
            "LLM_DISABLE_THINKING",
            "true" if llm_provider == "deepseek" else "false",
        ),
        "LLM_DISABLE_THINKING",
    )
    llm_adaptive_thinking = _parse_bool(
        os.getenv("LLM_ADAPTIVE_THINKING", "true"), "LLM_ADAPTIVE_THINKING"
    )

    telegram_proxy_url = os.getenv(
        "TELEGRAM_PROXY_URL",
        "",
    ).strip()

    brave_search_api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    brave_key_file = os.getenv("BRAVE_SEARCH_API_KEY_FILE", "").strip()
    if brave_key_file:
        try:
            with Path(brave_key_file).open("rb") as key_stream:
                raw_key = key_stream.read(4097)
            if len(raw_key) > 4096:
                raise ValueError("слишком большой файл")
            brave_search_api_key = raw_key.decode("utf-8").strip()
            if not brave_search_api_key or any(
                char.isspace() for char in brave_search_api_key
            ):
                raise ValueError("пустой или некорректный ключ")
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "Не удалось прочитать BRAVE_SEARCH_API_KEY_FILE"
            ) from error

    if not telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN не найден в окружении")

    if not deepseek_api_key:
        raise RuntimeError("LLM_API_KEY или DEEPSEEK_API_KEY не найден в окружении")

    if not deepseek_base_url:
        raise RuntimeError("LLM_BASE_URL обязателен для openai-compatible")

    if not deepseek_model:
        raise RuntimeError("LLM_MODEL обязателен для openai-compatible")

    if not art_chat_id_raw:
        raise RuntimeError("ART_CHAT_ID не найден в окружении")

    try:
        art_chat_id = int(art_chat_id_raw)
    except ValueError as error:
        raise ValueError("ART_CHAT_ID должен быть целым числом") from error

    rate_limit_seconds = _parse_non_negative_float(
        os.getenv(
            "RATE_LIMIT_SECONDS",
            "2.0",
        ),
        "RATE_LIMIT_SECONDS",
    )

    rate_limit_retention_seconds = _parse_positive_float(
        os.getenv(
            "RATE_LIMIT_RETENTION_SECONDS",
            "300.0",
        ),
        "RATE_LIMIT_RETENTION_SECONDS",
    )

    if rate_limit_retention_seconds < rate_limit_seconds:
        raise ValueError(
            "RATE_LIMIT_RETENTION_SECONDS не может быть меньше " "RATE_LIMIT_SECONDS"
        )

    conversation_history_limit = _parse_positive_int(
        os.getenv(
            "CONVERSATION_HISTORY_LIMIT",
            "8",
        ),
        "CONVERSATION_HISTORY_LIMIT",
    )

    conversation_history_ttl_seconds = _parse_positive_float(
        os.getenv("CONVERSATION_HISTORY_TTL_SECONDS", "604800.0"),
        "CONVERSATION_HISTORY_TTL_SECONDS",
    )
    user_state_retention_seconds = _parse_positive_float(
        os.getenv(
            "USER_STATE_RETENTION_SECONDS",
            "86400.0",
        ),
        "USER_STATE_RETENTION_SECONDS",
    )

    creator_id_raw = os.getenv("CREATOR_ID", "").strip()
    creator_id = (
        _parse_positive_int(creator_id_raw, "CREATOR_ID") if creator_id_raw else None
    )

    proactive_check_seconds = _parse_positive_float(
        os.getenv("PROACTIVE_CHECK_SECONDS", "30.0"), "PROACTIVE_CHECK_SECONDS"
    )
    proactive_idle_seconds = _parse_positive_float(
        os.getenv("PROACTIVE_IDLE_SECONDS", "5400.0"), "PROACTIVE_IDLE_SECONDS"
    )
    proactive_idle_max_seconds = _parse_positive_float(
        os.getenv(
            "PROACTIVE_IDLE_MAX_SECONDS", str(max(10800.0, proactive_idle_seconds))
        ),
        "PROACTIVE_IDLE_MAX_SECONDS",
    )
    if proactive_idle_max_seconds < proactive_idle_seconds:
        raise ValueError(
            "PROACTIVE_IDLE_MAX_SECONDS не может быть меньше PROACTIVE_IDLE_SECONDS"
        )
    proactive_cooldown_seconds = _parse_positive_float(
        os.getenv("PROACTIVE_COOLDOWN_SECONDS", "72000.0"),
        "PROACTIVE_COOLDOWN_SECONDS",
    )
    e621_request_interval_seconds = _parse_non_negative_float(
        os.getenv("E621_REQUEST_INTERVAL_SECONDS", "1.0"),
        "E621_REQUEST_INTERVAL_SECONDS",
    )
    message_reactions_enabled = _parse_bool(
        os.getenv("MESSAGE_REACTIONS_ENABLED", "true"), "MESSAGE_REACTIONS_ENABLED"
    )
    message_reaction_chance = _parse_non_negative_float(
        os.getenv("MESSAGE_REACTION_CHANCE", "0.35"), "MESSAGE_REACTION_CHANCE"
    )
    if message_reaction_chance > 1:
        raise ValueError("MESSAGE_REACTION_CHANCE не может быть больше 1")
    message_reaction_cooldown_seconds = _parse_non_negative_float(
        os.getenv("MESSAGE_REACTION_COOLDOWN_SECONDS", "120"),
        "MESSAGE_REACTION_COOLDOWN_SECONDS",
    )
    sticker_reaction_chance = _parse_non_negative_float(
        os.getenv("STICKER_REACTION_CHANCE", "0.35"),
        "STICKER_REACTION_CHANCE",
    )
    if sticker_reaction_chance > 1:
        raise ValueError("STICKER_REACTION_CHANCE не может быть больше 1")
    sticker_cooldown_seconds = _parse_non_negative_float(
        os.getenv("STICKER_COOLDOWN_SECONDS", "180"),
        "STICKER_COOLDOWN_SECONDS",
    )
    sticker_min_replies = _parse_positive_int(
        os.getenv("STICKER_MIN_REPLIES", "3"),
        "STICKER_MIN_REPLIES",
    )
    mini_app_url = os.getenv("MINI_APP_URL", "").strip()
    if mini_app_url and not mini_app_url.startswith("https://"):
        raise ValueError("MINI_APP_URL должен начинаться с https://")
    mini_app_server_enabled = _parse_bool(
        os.getenv("MINI_APP_SERVER_ENABLED", "false"),
        "MINI_APP_SERVER_ENABLED",
    )
    if mini_app_server_enabled and not mini_app_url:
        raise RuntimeError("MINI_APP_URL обязателен для встроенной Mini App")
    mini_app_host = os.getenv("MINI_APP_HOST", "127.0.0.1").strip()
    if not mini_app_host:
        raise ValueError("MINI_APP_HOST не может быть пустым")
    mini_app_port = _parse_positive_int(
        os.getenv("MINI_APP_PORT", "8080"), "MINI_APP_PORT"
    )
    if mini_app_port > 65535:
        raise ValueError("MINI_APP_PORT не может быть больше 65535")
    mini_app_auth_max_age_seconds = _parse_positive_int(
        os.getenv("MINI_APP_AUTH_MAX_AGE_SECONDS", "3600"),
        "MINI_APP_AUTH_MAX_AGE_SECONDS",
    )
    audio_enabled = _parse_bool(
        os.getenv("AUDIO_UNDERSTANDING_ENABLED", "false"), "AUDIO_UNDERSTANDING_ENABLED"
    )
    audio_key = os.getenv("AUDIO_API_KEY", "").strip()
    audio_url = os.getenv("AUDIO_BASE_URL", "").strip()
    audio_model = os.getenv("AUDIO_MODEL", "").strip()
    if audio_enabled and (
        not audio_key or not audio_model or not audio_url.startswith("https://")
    ):
        raise RuntimeError(
            "Для аудиомодели нужны AUDIO_API_KEY, AUDIO_MODEL и HTTPS AUDIO_BASE_URL"
        )

    return Settings(
        telegram_token=telegram_token,
        deepseek_api_key=deepseek_api_key,
        art_chat_id=art_chat_id,
        llm_provider=llm_provider,
        llm_disable_thinking=llm_disable_thinking,
        llm_adaptive_thinking=llm_adaptive_thinking,
        mini_app_url=mini_app_url or None,
        mini_app_server_enabled=mini_app_server_enabled,
        mini_app_host=mini_app_host,
        mini_app_port=mini_app_port,
        mini_app_auth_max_age_seconds=mini_app_auth_max_age_seconds,
        message_reactions_enabled=message_reactions_enabled,
        reply_supplements_enabled=_parse_bool(
            os.getenv("REPLY_SUPPLEMENTS_ENABLED", "true"), "REPLY_SUPPLEMENTS_ENABLED"
        ),
        message_reaction_chance=message_reaction_chance,
        message_reaction_cooldown_seconds=message_reaction_cooldown_seconds,
        sticker_reaction_chance=sticker_reaction_chance,
        sticker_cooldown_seconds=sticker_cooldown_seconds,
        sticker_min_replies=sticker_min_replies,
        sticker_pack_enabled=_parse_bool(
            os.getenv("STICKER_PACK_ENABLED", "true"), "STICKER_PACK_ENABLED"
        ),
        pdf_ocr_enabled=_parse_bool(
            os.getenv("PDF_OCR_ENABLED", "false"), "PDF_OCR_ENABLED"
        ),
        audio_understanding_enabled=audio_enabled,
        audio_api_key=audio_key or None,
        audio_base_url=audio_url or None,
        audio_model=audio_model or None,
        audio_input_data_url=_parse_bool(
            os.getenv("AUDIO_INPUT_DATA_URL", "false"), "AUDIO_INPUT_DATA_URL"
        ),
        music_audd_api_token=os.getenv("MUSIC_AUDD_API_TOKEN", "").strip() or None,
        music_recognition_provider=music_provider,
        saucenao_api_key=os.getenv("SAUCENAO_API_KEY", "").strip() or None,
        deepseek_model=deepseek_model,
        deepseek_base_url=deepseek_base_url,
        telegram_proxy_url=telegram_proxy_url or None,
        youtube_cookies_file=(
            Path(os.environ["YOUTUBE_COOKIES_FILE"].strip())
            if os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
            else None
        ),
        log_level=os.getenv(
            "LOG_LEVEL",
            "INFO",
        ),
        data_dir=Path(
            os.getenv(
                "DATA_DIR",
                "data",
            )
        ),
        admin_ids=_parse_admin_ids(
            os.getenv(
                "ADMIN_IDS",
                "",
            )
        ),
        creator_id=creator_id,
        creator_ids=_parse_archive_ids(os.getenv("CREATOR_IDS", ""), "CREATOR_IDS"),
        favorites_user_ids=_parse_archive_ids(
            os.getenv("FAVORITES_USER_IDS", ""), "FAVORITES_USER_IDS"
        ),
        code_review_user_ids=_parse_archive_ids(
            os.getenv("CODE_REVIEW_USER_IDS", ""), "CODE_REVIEW_USER_IDS"
        ),
        brave_search_api_key=brave_search_api_key or None,
        deepseek_search_enabled=_parse_bool(
            os.getenv("DEEPSEEK_SEARCH_ENABLED", "true"), "DEEPSEEK_SEARCH_ENABLED"
        ),
        deepseek_search_api_key=os.getenv("DEEPSEEK_SEARCH_API_KEY", "").strip()
        or None,
        web_api_daily_limit=_parse_positive_int(
            os.getenv("WEB_API_DAILY_LIMIT", "200"), "WEB_API_DAILY_LIMIT"
        ),
        brave_monthly_search_limit=_parse_positive_int(
            os.getenv("BRAVE_MONTHLY_SEARCH_LIMIT", "1000"),
            "BRAVE_MONTHLY_SEARCH_LIMIT",
        ),
        alpaca_api_key=os.getenv("ALPACA_API_KEY", "").strip() or None,
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", "").strip() or None,
        e621_user_agent=(
            os.getenv(
                "E621_USER_AGENT",
                DEFAULT_E621_USER_AGENT,
            ).strip()
            or DEFAULT_E621_USER_AGENT
        ),
        e621_request_interval_seconds=e621_request_interval_seconds,
        rate_limit_seconds=rate_limit_seconds,
        llm_daily_request_limit=_parse_positive_int(
            os.getenv("LLM_DAILY_REQUEST_LIMIT", "300"), "LLM_DAILY_REQUEST_LIMIT"
        ),
        llm_global_daily_request_limit=_parse_positive_int(
            os.getenv("LLM_GLOBAL_DAILY_REQUEST_LIMIT", "3000"),
            "LLM_GLOBAL_DAILY_REQUEST_LIMIT",
        ),
        llm_budget_alert_user_id=(
            _parse_positive_int(
                os.environ["LLM_BUDGET_ALERT_USER_ID"], "LLM_BUDGET_ALERT_USER_ID"
            )
            if os.getenv("LLM_BUDGET_ALERT_USER_ID", "").strip()
            else None
        ),
        llm_daily_request_limit_overrides=_parse_llm_limit_overrides(
            os.getenv("LLM_DAILY_REQUEST_LIMIT_OVERRIDES", "{}")
        ),
        rate_limit_retention_seconds=rate_limit_retention_seconds,
        conversation_history_limit=conversation_history_limit,
        conversation_history_ttl_seconds=conversation_history_ttl_seconds,
        dialogue_archive_user_ids=_parse_archive_ids(
            os.getenv("DIALOGUE_ARCHIVE_USER_IDS", "")
        ),
        dialogue_shared_groups=_parse_dialogue_groups(
            os.getenv("DIALOGUE_SHARED_GROUPS", "")
        ),
        user_state_retention_seconds=user_state_retention_seconds,
        proactive_check_seconds=proactive_check_seconds,
        proactive_lively_enabled=_parse_bool(
            os.getenv("PROACTIVE_LIVELY_ENABLED", "true"), "PROACTIVE_LIVELY_ENABLED"
        ),
        proactive_news_enabled=_parse_bool(
            os.getenv("PROACTIVE_NEWS_ENABLED", "true"), "PROACTIVE_NEWS_ENABLED"
        ),
        proactive_idle_seconds=proactive_idle_seconds,
        proactive_idle_max_seconds=proactive_idle_max_seconds,
        proactive_cooldown_seconds=proactive_cooldown_seconds,
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small").strip() or "small",
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu",
        whisper_compute_type=(
            os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
        ),
    )
