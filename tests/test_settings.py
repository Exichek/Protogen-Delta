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

    monkeypatch.delenv("RATE_LIMIT_SECONDS", raising=False)
    monkeypatch.delenv(
        "RATE_LIMIT_RETENTION_SECONDS",
        raising=False,
    )
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123456")

    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("ADMIN_IDS", raising=False)

    settings = load_settings()

    assert settings.telegram_token == "test-token"
    assert settings.deepseek_api_key == "test-key"
    assert settings.art_chat_id == -100123456
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert settings.admin_ids == frozenset()
    assert settings.rate_limit_seconds == 2.0
    assert settings.rate_limit_retention_seconds == 300.0


def test_load_settings_with_custom_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Необязательные переменные должны переопределять значения по умолчанию."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv("RATE_LIMIT_SECONDS", "3.5")
    monkeypatch.setenv("RATE_LIMIT_RETENTION_SECONDS", "600")
    monkeypatch.setenv("TELEGRAM_TOKEN", "telegram")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek")
    monkeypatch.setenv("ART_CHAT_ID", "-100999")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATA_DIR", "custom-data")
    monkeypatch.setenv("ADMIN_IDS", "123, 456,789")

    settings = load_settings()

    assert settings.rate_limit_seconds == 3.5
    assert settings.rate_limit_retention_seconds == 600.0
    assert settings.deepseek_base_url == "https://example.com"
    assert settings.deepseek_model == "test-model"
    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("custom-data")
    assert settings.admin_ids == frozenset({123, 456, 789})


def test_load_settings_without_telegram_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующий Telegram-токен должен приводить к ошибке."""
    _disable_dotenv(monkeypatch)

    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")

    with pytest.raises(
        RuntimeError,
        match="TELEGRAM_TOKEN",
    ):
        load_settings()


def test_load_settings_without_deepseek_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующий ключ DeepSeek должен приводить к ошибке."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ART_CHAT_ID", "-100123")

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

    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "not-a-number")

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

    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv("ADMIN_IDS", "123,abc,456")

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
    ],
)
def test_load_settings_rejects_invalid_rate_limit_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    """Некорректные настройки rate limit должны отклоняться."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        load_settings()


def test_load_settings_rejects_retention_shorter_than_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Время хранения не может быть меньше пользовательского cooldown."""
    _disable_dotenv(monkeypatch)

    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("ART_CHAT_ID", "-100123")
    monkeypatch.setenv("RATE_LIMIT_SECONDS", "10")
    monkeypatch.setenv("RATE_LIMIT_RETENTION_SECONDS", "5")

    with pytest.raises(
        ValueError,
        match="RATE_LIMIT_RETENTION_SECONDS",
    ):
        load_settings()
