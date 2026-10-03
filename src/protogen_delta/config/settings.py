"""Настройки приложения и загрузка переменных окружения."""

import os
from dataclasses import dataclass
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
    mini_app_url: str | None = None
    mini_app_server_enabled: bool = False
    mini_app_host: str = "127.0.0.1"
    mini_app_port: int = 8080
    mini_app_auth_max_age_seconds: int = 3600
    sticker_reaction_chance: float = 0.15
    sticker_cooldown_seconds: float = 900.0
    sticker_min_replies: int = 4
    sticker_pack_enabled: bool = True
    pdf_ocr_enabled: bool = False
    audio_understanding_enabled: bool = False
    audio_api_key: str | None = None
    audio_base_url: str | None = None
    audio_model: str | None = None
    saucenao_api_key: str | None = None
    telegram_proxy_url: str | None = None
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    admin_ids: frozenset[int] = frozenset()
    creator_id: int | None = None
    brave_search_api_key: str | None = None
    e621_user_agent: str = DEFAULT_E621_USER_AGENT
    e621_request_interval_seconds: float = 1.0
    rate_limit_seconds: float = 2.0
    rate_limit_retention_seconds: float = 300.0
    conversation_history_limit: int = 8
    user_state_retention_seconds: float = 86400.0
    proactive_check_seconds: float = 300.0
    proactive_idle_seconds: float = 14400.0
    proactive_cooldown_seconds: float = 86400.0
    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    @property
    def llm_api_key(self) -> str:
        """Вернуть ключ выбранного OpenAI-совместимого LLM-провайдера."""
        return self.deepseek_api_key

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
        os.getenv("PROACTIVE_CHECK_SECONDS", "300.0"), "PROACTIVE_CHECK_SECONDS"
    )
    proactive_idle_seconds = _parse_positive_float(
        os.getenv("PROACTIVE_IDLE_SECONDS", "14400.0"), "PROACTIVE_IDLE_SECONDS"
    )
    proactive_cooldown_seconds = _parse_positive_float(
        os.getenv("PROACTIVE_COOLDOWN_SECONDS", "86400.0"),
        "PROACTIVE_COOLDOWN_SECONDS",
    )
    e621_request_interval_seconds = _parse_non_negative_float(
        os.getenv("E621_REQUEST_INTERVAL_SECONDS", "1.0"),
        "E621_REQUEST_INTERVAL_SECONDS",
    )
    sticker_reaction_chance = _parse_non_negative_float(
        os.getenv("STICKER_REACTION_CHANCE", "0.15"),
        "STICKER_REACTION_CHANCE",
    )
    if sticker_reaction_chance > 1:
        raise ValueError("STICKER_REACTION_CHANCE не может быть больше 1")
    sticker_cooldown_seconds = _parse_non_negative_float(
        os.getenv("STICKER_COOLDOWN_SECONDS", "900"),
        "STICKER_COOLDOWN_SECONDS",
    )
    sticker_min_replies = _parse_positive_int(
        os.getenv("STICKER_MIN_REPLIES", "4"),
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
        mini_app_url=mini_app_url or None,
        mini_app_server_enabled=mini_app_server_enabled,
        mini_app_host=mini_app_host,
        mini_app_port=mini_app_port,
        mini_app_auth_max_age_seconds=mini_app_auth_max_age_seconds,
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
        saucenao_api_key=os.getenv("SAUCENAO_API_KEY", "").strip() or None,
        deepseek_model=deepseek_model,
        deepseek_base_url=deepseek_base_url,
        telegram_proxy_url=telegram_proxy_url or None,
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
        brave_search_api_key=brave_search_api_key or None,
        e621_user_agent=(
            os.getenv(
                "E621_USER_AGENT",
                DEFAULT_E621_USER_AGENT,
            ).strip()
            or DEFAULT_E621_USER_AGENT
        ),
        e621_request_interval_seconds=e621_request_interval_seconds,
        rate_limit_seconds=rate_limit_seconds,
        rate_limit_retention_seconds=rate_limit_retention_seconds,
        conversation_history_limit=conversation_history_limit,
        user_state_retention_seconds=user_state_retention_seconds,
        proactive_check_seconds=proactive_check_seconds,
        proactive_idle_seconds=proactive_idle_seconds,
        proactive_cooldown_seconds=proactive_cooldown_seconds,
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small").strip() or "small",
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu",
        whisper_compute_type=(
            os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
        ),
    )
