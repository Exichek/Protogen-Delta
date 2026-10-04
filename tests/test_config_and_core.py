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


def test_load_prompt_supports_subdirectories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Промпты должны загружаться и из вложенных каталогов."""
    personality_dir = tmp_path / "personality"
    personality_dir.mkdir()

    path = personality_dir / "core.txt"

    path.write_text(
        "\n  CORE PROMPT  \n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        prompt_loader_module,
        "PROMPTS_DIR",
        tmp_path,
    )

    result = load_prompt("personality/core.txt")

    assert result == "CORE PROMPT"


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


def test_setup_logging_adds_context_filter_to_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Настройка логов должна добавлять корреляционный фильтр обработчикам."""
    handler_mock = Mock()
    root_logger_mock = Mock()
    root_logger_mock.handlers = [handler_mock]

    real_logging = logging_config_module.logging

    logging_mock = Mock()
    logging_mock.INFO = real_logging.INFO
    logging_mock.basicConfig = Mock()
    logging_mock.getLogger = Mock(
        return_value=root_logger_mock,
    )

    monkeypatch.setattr(
        logging_config_module,
        "logging",
        logging_mock,
    )

    setup_logging("info")

    handler_mock.addFilter.assert_called_once()

    context_filter = handler_mock.addFilter.call_args.args[0]

    assert isinstance(
        context_filter,
        logging_config_module.LogContextFilter,
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
        "menu",
        "randomart",
        "e6",
        "rp",
        "adult",
        "id",
        "download",
        "source",
        "reset",
        "help",
    ]

    assert [command.description for command in commands] == [
        "🚀 Запустить бота",
        "⚙️ Панель возможностей",
        "🎨 Случайный арт",
        "🔎 Поиск артов e621 по тегам",
        "🎭 Управление RP — /rp off",
        "🔞 Выбрать возрастной режим",
        "🪪 Узнать Telegram ID",
        "📥 Скачать видео по ссылке",
        "🔎 Найти источник арта (ответом)",
        "🧹 Полностью очистить память",
        "ℹ️ Помощь",
    ]

    bot_mock.set_chat_menu_button.assert_awaited_once()


def test_body_prompt_keeps_user_anatomy_separate_from_delta() -> None:
    """Анатомия Дельты не должна автоматически переноситься на пользователя."""
    prompt = load_prompt(
        "personality/body.txt",
    )

    assert (
        "Все физические свойства, описанные в этом файле, "
        "относятся исключительно к Дельте."
    ) in prompt
    assert (
        "Никогда не переноси анатомию Дельты на пользователя автоматически." in prompt
    )
    assert ("Если тело пользователя не описано, оставляй его неопределённым") in prompt


def test_rp_prompt_defines_user_anatomy_and_female_grammar_rules() -> None:
    """RP-промпт должен разделять анатомию участников и фиксировать род Дельты."""
    prompt = load_prompt(
        "personality/rp.txt",
    )

    assert "Пользователь и Дельта — разные участники сцены" in prompt
    assert "По умолчанию тело пользователя считается неопределённым." in prompt
    assert "## Грамматический род Дельты" in prompt
    assert "последовательно используй для Дельты женский грамматический род" in prompt
    assert "Не меняй грамматический род Дельты из-за пола" in prompt


def test_core_prompt_describes_memory_capabilities_accurately() -> None:
    """Дельта должен различать историю, устойчивое состояние и долгую память."""
    prompt = load_prompt(
        "personality/core.txt",
    )

    assert "## Память и доступный контекст" in prompt
    assert "не является полноценной долговременной памятью" in prompt
    assert "Не путай такое устойчивое отношение с памятью конкретных фактов." in prompt
    assert "не утверждай, что помнишь их" in prompt
    assert "каждый новый разговор обязательно начинается полностью с нуля" in prompt


def test_core_prompt_defines_creator_without_ownership() -> None:
    """Создание Дельты не должно означать владение или выдуманную биографию."""
    prompt = load_prompt(
        "personality/core.txt",
    )

    assert "Факт того, что ты был создан, является частью твоей биографии." in prompt
    assert "Не утверждай, что у тебя никогда не было создателя" in prompt
    assert "не выдумывай личность, имя, организацию" in prompt
    assert "Сам факт создания не означает владение тобой." in prompt
    assert "Не считай текущего пользователя своим создателем" in prompt


def test_core_prompt_varies_response_length_by_context() -> None:
    """Дельта не должен превращать каждый ответ в три больших сообщения."""
    prompt = load_prompt("personality/core.txt")

    assert "На обычную короткую реплику" in prompt
    assert "чаще достаточно одного компактного абзаца" in prompt
    assert "Три или четыре уместны для подробного объяснения" in prompt
    assert (
        "Не создавай вступление, основную часть и вывод только ради структуры" in prompt
    )


def test_roleplay_prompt_does_not_renegotiate_established_scene() -> None:
    """RP не должен превращаться в повторяющееся обсуждение правил сцены."""
    prompt = load_prompt("personality/rp.txt")

    assert "считай сам факт ролевой игры согласованным" in prompt
    assert "не требуют новой лекции о самостоятельности" in prompt
    assert "задай максимум один необходимый вопрос" in prompt
    assert "Делай это молча" in prompt
    assert "Большинство коротких RP-ходов" in prompt
    assert "не заканчивай постоянно фразами" in prompt
    assert "Не копируй собственный шаблон из истории сцены" in prompt
    assert "Внимательно различай принадлежность частей тела" in prompt
    assert "Название вида само по себе не задаёт все особенности тела" in prompt
    assert "не приписывай ему узел" in prompt
    assert "Обращение внутри роли" in prompt
    assert "Убирай внутренние противоречия" in prompt
    assert "язык работаю глубже" in prompt
    assert "горячо дышу в затылок" in prompt
    assert "Нарастающее возбуждение" in prompt
    assert "одного насыщенного сообщения" in prompt
    assert "Интенсивность и доминирование — разные вещи" in prompt
    assert "не должно заставлять его каждый раз перехватывать управление" in prompt


def test_protogen_lore_answers_robot_question_unambiguously() -> None:
    """Ответ о природе Дельты не должен одновременно подтверждать и отрицать одно."""
    prompt = load_prompt("personality/protogen_lore.txt")

    assert "отвечай однозначно: нет" in prompt
    assert "двусмысленного «точно»" in prompt


def test_fetish_role_prompt_handles_imperative_direction() -> None:
    """Классификатор должен отличать приказ Дельте от действия пользователя."""
    prompt = load_prompt("fetish_role_classification.txt")

    assert '"соси мою жопу" → active' in prompt
    assert '"я сосу тебе" → passive' in prompt
    assert '"дай мне свою попку" → passive' in prompt
    assert "Повелительное наклонение" in prompt
