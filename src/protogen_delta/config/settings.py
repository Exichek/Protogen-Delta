"""Настройки приложения и загрузка переменных окружения."""

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки приложения, загружаемые из переменных окружения."""

    telegram_token: str
    deepseek_api_key: str
    art_chat_id: int
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    telegram_proxy_url: str | None = None
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    admin_ids: frozenset[int] = frozenset()
    creator_id: int | None = None
    rate_limit_seconds: float = 2.0
    rate_limit_retention_seconds: float = 300.0
    conversation_history_limit: int = 8
    user_state_retention_seconds: float = 86400.0
    proactive_check_seconds: float = 300.0
    proactive_idle_seconds: float = 86400.0
    proactive_cooldown_seconds: float = 172800.0


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


def load_settings() -> Settings:
    """Загрузить настройки приложения из переменных окружения."""
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_TOKEN")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    art_chat_id_raw = os.getenv("ART_CHAT_ID")

    deepseek_model = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-flash",
    )

    telegram_proxy_url = os.getenv(
        "TELEGRAM_PROXY_URL",
        "",
    ).strip()

    if not telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN не найден в окружении")

    if not deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY не найден в окружении")

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
        os.getenv("PROACTIVE_IDLE_SECONDS", "86400.0"), "PROACTIVE_IDLE_SECONDS"
    )
    proactive_cooldown_seconds = _parse_positive_float(
        os.getenv("PROACTIVE_COOLDOWN_SECONDS", "172800.0"),
        "PROACTIVE_COOLDOWN_SECONDS",
    )

    return Settings(
        telegram_token=telegram_token,
        deepseek_api_key=deepseek_api_key,
        art_chat_id=art_chat_id,
        deepseek_model=deepseek_model,
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
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
        rate_limit_seconds=rate_limit_seconds,
        rate_limit_retention_seconds=rate_limit_retention_seconds,
        conversation_history_limit=conversation_history_limit,
        user_state_retention_seconds=user_state_retention_seconds,
        proactive_check_seconds=proactive_check_seconds,
        proactive_idle_seconds=proactive_idle_seconds,
        proactive_cooldown_seconds=proactive_cooldown_seconds,
    )
