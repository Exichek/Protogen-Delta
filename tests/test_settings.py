"""Тесты загрузки настроек приложения."""

from pathlib import Path

import pytest

import protogen_delta.config.settings as settings_module
from protogen_delta.config.settings import load_settings


def _disable_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Не позволять тестам читать настоящий файл .env."""
    monkeypatch.setattr(
        settings_module,
        "load_dotenv",
        lambda: None,
    )


def test_load_settings_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обязательные переменные должны загружаться с настройками по умолчанию."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123456")

    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("MINI_APP_URL", raising=False)
    monkeypatch.delenv("MINI_APP_SERVER_ENABLED", raising=False)
    monkeypatch.delenv("MINI_APP_HOST", raising=False)
    monkeypatch.delenv("MINI_APP_PORT", raising=False)
    monkeypatch.delenv("MINI_APP_AUTH_MAX_AGE_SECONDS", raising=False)
    monkeypatch.delenv("STICKER_REACTION_CHANCE", raising=False)
    monkeypatch.delenv("STICKER_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("STICKER_MIN_REPLIES", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("ADMIN_IDS", raising=False)
    monkeypatch.delenv("CREATOR_ID", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("E621_USER_AGENT", raising=False)
    monkeypatch.delenv("E621_REQUEST_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("PROACTIVE_CHECK_SECONDS", raising=False)
    monkeypatch.delenv("PROACTIVE_IDLE_SECONDS", raising=False)
    monkeypatch.delenv("PROACTIVE_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("WHISPER_MODEL_SIZE", raising=False)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("RATE_LIMIT_SECONDS", raising=False)
    monkeypatch.delenv(
        "RATE_LIMIT_RETENTION_SECONDS",
        raising=False,
    )
    monkeypatch.delenv(
        "CONVERSATION_HISTORY_LIMIT",
        raising=False,
    )
    monkeypatch.delenv(
        "USER_STATE_RETENTION_SECONDS",
        raising=False,
    )
    monkeypatch.delenv(
        "TELEGRAM_PROXY_URL",
        raising=False,
    )

    settings = load_settings()

    assert settings.telegram_token == "test-token"
    assert settings.deepseek_api_key == "test-key"
    assert settings.art_chat_id == -100123456
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.llm_provider == "deepseek"
    assert settings.llm_disable_thinking is True
    assert settings.mini_app_url is None
    assert settings.mini_app_server_enabled is False
    assert settings.mini_app_host == "127.0.0.1"
    assert settings.mini_app_port == 8080
    assert settings.mini_app_auth_max_age_seconds == 3600
    assert settings.sticker_reaction_chance == 0.15
    assert settings.sticker_cooldown_seconds == 900.0
    assert settings.sticker_min_replies == 4
    assert settings.telegram_proxy_url is None
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert settings.admin_ids == frozenset()
    assert settings.creator_id is None
    assert settings.brave_search_api_key is None
    assert settings.e621_user_agent.startswith("ProtogenDelta/0.1")
    assert settings.e621_request_interval_seconds == 1.0
    assert settings.proactive_check_seconds == 300.0
    assert settings.proactive_idle_seconds == 14400.0
    assert settings.proactive_cooldown_seconds == 86400.0
    assert settings.rate_limit_seconds == 2.0
    assert settings.rate_limit_retention_seconds == 300.0
    assert settings.conversation_history_limit == 8
    assert settings.user_state_retention_seconds == 86400.0
    assert settings.whisper_model_size == "small"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"


def test_load_settings_with_custom_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Необязательные переменные должны переопределять значения по умолчанию."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek")
    monkeypatch.setenv("ART_CHAT_ID", "-100999")

    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL",
        "https://example.com",
    )
    monkeypatch.setenv(
        "DEEPSEEK_MODEL",
        "test-model",
    )
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATA_DIR", "custom-data")
    monkeypatch.setenv(
        "ADMIN_IDS",
        "123, 456,789",
    )
    monkeypatch.setenv("CREATOR_ID", "123")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-secret")
    monkeypatch.setenv("E621_USER_AGENT", "MyBot/2.0 (by tester)")
    monkeypatch.setenv("E621_REQUEST_INTERVAL_SECONDS", "1.5")
    monkeypatch.setenv("PROACTIVE_CHECK_SECONDS", "60")
    monkeypatch.setenv("PROACTIVE_IDLE_SECONDS", "120")
    monkeypatch.setenv("PROACTIVE_COOLDOWN_SECONDS", "240")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "base")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("STICKER_REACTION_CHANCE", "0.25")
    monkeypatch.setenv("STICKER_COOLDOWN_SECONDS", "120")
    monkeypatch.setenv("STICKER_MIN_REPLIES", "2")
    monkeypatch.setenv("MINI_APP_URL", "https://delta.example/app")
    monkeypatch.setenv("MINI_APP_SERVER_ENABLED", "true")
    monkeypatch.setenv("MINI_APP_HOST", "0.0.0.0")
    monkeypatch.setenv("MINI_APP_PORT", "9000")
    monkeypatch.setenv("MINI_APP_AUTH_MAX_AGE_SECONDS", "600")

    monkeypatch.setenv(
        "RATE_LIMIT_SECONDS",
        "3.5",
    )
    monkeypatch.setenv(
        "RATE_LIMIT_RETENTION_SECONDS",
        "600",
    )
    monkeypatch.setenv(
        "CONVERSATION_HISTORY_LIMIT",
        "12",
    )
    monkeypatch.setenv(
        "USER_STATE_RETENTION_SECONDS",
        "3600",
    )
    monkeypatch.setenv(
        "TELEGRAM_PROXY_URL",
        "socks5://127.0.0.1:10808",
    )

    settings = load_settings()

    assert settings.telegram_token == "telegram"
    assert settings.deepseek_api_key == "deepseek"
    assert settings.art_chat_id == -100999
    assert settings.deepseek_base_url == "https://example.com"
    assert settings.telegram_proxy_url == "socks5://127.0.0.1:10808"
    assert settings.deepseek_model == "test-model"
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("custom-data")
    assert settings.admin_ids == frozenset({123, 456, 789})
    assert settings.creator_id == 123
    assert settings.brave_search_api_key == "brave-secret"
    assert settings.e621_user_agent == "MyBot/2.0 (by tester)"
    assert settings.e621_request_interval_seconds == 1.5
    assert settings.proactive_check_seconds == 60.0
    assert settings.proactive_idle_seconds == 120.0
    assert settings.proactive_cooldown_seconds == 240.0
    assert settings.rate_limit_seconds == 3.5
    assert settings.rate_limit_retention_seconds == 600.0
    assert settings.conversation_history_limit == 12
    assert settings.user_state_retention_seconds == 3600.0
    assert settings.whisper_model_size == "base"
    assert settings.whisper_device == "cuda"
    assert settings.whisper_compute_type == "float16"
    assert settings.sticker_reaction_chance == 0.25
    assert settings.sticker_cooldown_seconds == 120.0
    assert settings.sticker_min_replies == 2
    assert settings.mini_app_server_enabled is True
    assert settings.mini_app_host == "0.0.0.0"
    assert settings.mini_app_port == 9000
    assert settings.mini_app_auth_max_age_seconds == 600


def test_load_settings_without_telegram_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующий Telegram-токен должен приводить к ошибке."""
    _disable_dotenv(monkeypatch)

    monkeypatch.delenv(
        "TELEGRAM_TOKEN",
        raising=False,
    )
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "ART_CHAT_ID",
        "-100123",
    )

    with pytest.raises(
        RuntimeError,
        match="TELEGRAM_TOKEN",
    ):
        load_settings()


def test_load_settings_with_openai_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Нейтральные LLM_* должны полностью настроить совместимый провайдер."""
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "vision-model")
    monkeypatch.setenv("MINI_APP_URL", "https://delta.example/app")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = load_settings()

    assert settings.llm_provider == "openai-compatible"
    assert settings.llm_api_key == "generic-key"
    assert settings.llm_base_url == "https://llm.example/v1"
    assert settings.llm_model == "vision-model"
    assert settings.llm_disable_thinking is False
    assert settings.mini_app_url == "https://delta.example/app"


def test_load_settings_requires_generic_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Совместимый провайдер без endpoint не должен тихо уйти на DeepSeek."""
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        load_settings()


def test_load_settings_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Опечатка в имени провайдера должна завершать запуск с ошибкой."""
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mystery")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        load_settings()


def test_load_settings_rejects_invalid_thinking_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Булева LLM-настройка не должна принимать произвольный текст."""
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("LLM_DISABLE_THINKING", "perhaps")

    with pytest.raises(ValueError, match="LLM_DISABLE_THINKING"):
        load_settings()


def test_load_settings_rejects_non_https_mini_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telegram Mini App должна иметь HTTPS URL."""
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv("MINI_APP_URL", "http://delta.example/app")

    with pytest.raises(ValueError, match="MINI_APP_URL"):
        load_settings()


def test_load_settings_requires_url_for_embedded_mini_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Встроенный сервер без публичного URL не должен создавать мёртвую панель."""
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv("MINI_APP_SERVER_ENABLED", "true")
    monkeypatch.delenv("MINI_APP_URL", raising=False)

    with pytest.raises(RuntimeError, match="MINI_APP_URL"):
        load_settings()


def test_load_settings_without_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующий ключ DeepSeek должен приводить к ошибке."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv(
        "TELEGRAM_TOKEN",
        "test-token",
    )
    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False,
    )
    monkeypatch.setenv(
        "ART_CHAT_ID",
        "-100123",
    )

    with pytest.raises(
        RuntimeError,
        match="DEEPSEEK_API_KEY",
    ):
        load_settings()


def test_load_settings_with_invalid_art_chat_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ART_CHAT_ID должен содержать целое число."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv(
        "TELEGRAM_TOKEN",
        "test-token",
    )
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "ART_CHAT_ID",
        "not-a-number",
    )

    with pytest.raises(
        ValueError,
        match="ART_CHAT_ID",
    ):
        load_settings()


def test_load_settings_with_invalid_admin_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADMIN_IDS должен содержать только числовые Telegram ID."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv(
        "TELEGRAM_TOKEN",
        "test-token",
    )
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "ART_CHAT_ID",
        "-100123",
    )
    monkeypatch.setenv(
        "ADMIN_IDS",
        "123,abc,456",
    )

    with pytest.raises(
        ValueError,
        match="ADMIN_IDS",
    ):
        load_settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RATE_LIMIT_SECONDS", "abc"),
        ("RATE_LIMIT_SECONDS", "-1"),
        ("RATE_LIMIT_SECONDS", "nan"),
        ("RATE_LIMIT_SECONDS", "inf"),
        ("RATE_LIMIT_RETENTION_SECONDS", "0"),
        ("RATE_LIMIT_RETENTION_SECONDS", "-1"),
        ("RATE_LIMIT_RETENTION_SECONDS", "nan"),
        ("RATE_LIMIT_RETENTION_SECONDS", "inf"),
        ("USER_STATE_RETENTION_SECONDS", "0"),
        ("USER_STATE_RETENTION_SECONDS", "-1"),
        ("USER_STATE_RETENTION_SECONDS", "nan"),
        ("USER_STATE_RETENTION_SECONDS", "inf"),
        ("USER_STATE_RETENTION_SECONDS", "abc"),
        ("E621_REQUEST_INTERVAL_SECONDS", "-1"),
        ("E621_REQUEST_INTERVAL_SECONDS", "nan"),
        ("STICKER_REACTION_CHANCE", "-1"),
        ("STICKER_COOLDOWN_SECONDS", "-1"),
    ],
)
def test_load_settings_rejects_invalid_float_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    """Некорректные числовые настройки должны отклоняться."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv(
        "TELEGRAM_TOKEN",
        "test-token",
    )
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "ART_CHAT_ID",
        "-100123",
    )
    monkeypatch.setenv(
        name,
        value,
    )

    with pytest.raises(
        ValueError,
        match=name,
    ):
        load_settings()


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "abc",
        "1.5",
    ],
)
def test_load_settings_rejects_invalid_history_limit(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """Лимит истории должен быть положительным целым числом."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv(
        "TELEGRAM_TOKEN",
        "test-token",
    )
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "ART_CHAT_ID",
        "-100123",
    )
    monkeypatch.setenv(
        "CONVERSATION_HISTORY_LIMIT",
        value,
    )

    with pytest.raises(
        ValueError,
        match="CONVERSATION_HISTORY_LIMIT",
    ):
        load_settings()


def test_load_settings_rejects_sticker_chance_above_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Вероятность стикера должна оставаться в диапазоне 0..1."""
    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv("STICKER_REACTION_CHANCE", "1.1")

    with pytest.raises(ValueError, match="STICKER_REACTION_CHANCE"):
        load_settings()


def test_load_settings_rejects_retention_shorter_than_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Время хранения rate limiter не может быть меньше cooldown."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv(
        "TELEGRAM_TOKEN",
        "test-token",
    )
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "ART_CHAT_ID",
        "-100123",
    )
    monkeypatch.setenv(
        "RATE_LIMIT_SECONDS",
        "10",
    )
    monkeypatch.setenv(
        "RATE_LIMIT_RETENTION_SECONDS",
        "5",
    )

    with pytest.raises(
        ValueError,
        match="RATE_LIMIT_RETENTION_SECONDS",
    ):
        load_settings()
