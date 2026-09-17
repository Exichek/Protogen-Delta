"""Тесты загрузчиков конфигурации и базовых компонентов."""

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot

import protogen_delta.config.json_loader as json_loader_module
import protogen_delta.config.prompt_loader as prompt_loader_module
import protogen_delta.core.logging_config as logging_config_module
from protogen_delta.config.json_loader import load_json
from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.logging_config import LOG_FORMAT, setup_logging
from protogen_delta.core.telegram_commands import set_commands


def test_load_json_returns_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Валидный JSON-объект должен корректно загружаться."""
    path = tmp_path / "config.json"

    path.write_text(
        json.dumps(
            {
                "NAME": "Дельта",
                "COUNT": 3,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        json_loader_module,
        "CONFIG_DATA_DIR",
        tmp_path,
    )

    result = load_json("config.json")

    assert result == {
        "NAME": "Дельта",
        "COUNT": 3,
    }


def test_load_json_raises_for_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующий JSON-файл должен приводить к FileNotFoundError."""
    monkeypatch.setattr(
        json_loader_module,
        "CONFIG_DATA_DIR",
        tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        load_json("missing.json")


def test_load_json_raises_for_invalid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Некорректный JSON должен приводить к JSONDecodeError."""
    path = tmp_path / "broken.json"

    path.write_text(
        "{not-json}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        json_loader_module,
        "CONFIG_DATA_DIR",
        tmp_path,
    )

    with pytest.raises(json.JSONDecodeError):
        load_json("broken.json")


def test_load_json_rejects_non_object_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Корневой элемент статического JSON должен быть объектом."""
    path = tmp_path / "list.json"

    path.write_text(
        '["one", "two"]',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        json_loader_module,
        "CONFIG_DATA_DIR",
        tmp_path,
    )

    with pytest.raises(
        TypeError,
        match="Корневой элемент JSON-конфига",
    ):
        load_json("list.json")


def test_load_prompt_returns_stripped_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Промпт должен загружаться без пробелов по краям."""
    path = tmp_path / "system.txt"

    path.write_text(
        "\n  SYSTEM PROMPT  \n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        prompt_loader_module,
        "PROMPTS_DIR",
        tmp_path,
    )

    result = load_prompt("system.txt")

    assert result == "SYSTEM PROMPT"


def test_load_prompt_raises_for_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующий промпт должен приводить к понятной ошибке."""
    monkeypatch.setattr(
        prompt_loader_module,
        "PROMPTS_DIR",
        tmp_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Файл промпта не найден",
    ):
        load_prompt("missing.txt")


def test_setup_logging_uses_requested_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Корректный уровень должен передаваться в logging.basicConfig."""
    basic_config_mock = Mock()

    monkeypatch.setattr(
        logging_config_module.logging,
        "basicConfig",
        basic_config_mock,
    )

    setup_logging("debug")

    basic_config_mock.assert_called_once_with(
        level=logging_config_module.logging.DEBUG,
        format=LOG_FORMAT,
    )


def test_setup_logging_rejects_unknown_level() -> None:
    """Неизвестный уровень логирования должен приводить к ошибке."""
    with pytest.raises(
        ValueError,
        match="Неизвестный уровень логирования",
    ):
        setup_logging("banana")


def test_set_commands_configures_telegram_menu() -> None:
    """В Telegram должно устанавливаться ожидаемое меню команд."""
    bot_mock = AsyncMock(spec=Bot)

    asyncio.run(
        set_commands(
            cast(Bot, bot_mock),
        )
    )

    bot_mock.set_my_commands.assert_awaited_once()

    call = bot_mock.set_my_commands.await_args

    assert call is not None

    commands = call.args[0]

    assert [command.command for command in commands] == [
        "start",
        "randomart",
        "reset",
        "help",
    ]

    assert [command.description for command in commands] == [
        "🚀 Запустить бота",
        "🎨 Случайный арт",
        "🧹 Очистить память диалога",
        "ℹ️ Помощь",
    ]
