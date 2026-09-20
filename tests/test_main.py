"""Тесты точки входа приложения."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import pytest

import protogen_delta.main as main_module
from protogen_delta.config.settings import Settings


def test_require_string_list_rejects_invalid_value() -> None:
    """Список должен содержать только строки."""
    with pytest.raises(
        TypeError,
        match="TEST должен содержать список строк",
    ):
        main_module._require_string_list(
            ["ok", 123],
            "TEST",
        )


def test_require_string_lists_rejects_invalid_value() -> None:
    """Словарь должен содержать списки строк."""
    with pytest.raises(
        TypeError,
        match="TEST должен содержать словарь списков строк",
    ):
        main_module._require_string_lists(
            {
                "valid": ["one"],
                "invalid": [123],
            },
            "TEST",
        )


def test_require_string_dict_rejects_invalid_value() -> None:
    """Словарь должен содержать строковые ключи и значения."""
    with pytest.raises(
        TypeError,
        match="TEST должен содержать словарь строк",
    ):
        main_module._require_string_dict(
            {
                "valid": "value",
                "invalid": 123,
            },
            "TEST",
        )


def test_main_builds_application_and_starts_polling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """main должен собрать приложение и запустить polling."""
    settings = Settings(
        telegram_token="telegram-token",
        deepseek_api_key="deepseek-key",
        art_chat_id=-1001234567890,
        deepseek_base_url="https://api.test.local",
        deepseek_model="test-model",
        log_level="INFO",
        data_dir=tmp_path,
        admin_ids=frozenset({123}),
        conversation_history_limit=12,
        user_state_retention_seconds=3600.0,
    )

    load_settings_mock = Mock(
        return_value=settings,
    )
    setup_logging_mock = Mock()

    bot_mock = Mock()
    bot_mock.delete_webhook = AsyncMock()
    bot_mock.session = Mock()
    bot_mock.session.close = AsyncMock()

    bot_constructor_mock = Mock(
        return_value=bot_mock,
    )

    dispatcher_mock = Mock()
    dispatcher_mock.include_router = Mock()
    dispatcher_mock.start_polling = AsyncMock()

    dispatcher_constructor_mock = Mock(
        return_value=dispatcher_mock,
    )

    register_error_handler_mock = Mock()

    user_states = Mock(
        name="user_states",
    )
    user_state_store_constructor_mock = Mock(
        return_value=user_states,
    )

    user_state_repository = Mock(
        name="user_state_repository",
    )
    user_state_repository_constructor_mock = Mock(
        return_value=user_state_repository,
    )

    json_data: dict[str, object] = {
        "start_messages.json": {
            "START_MESSAGES": [
                "Я уже работаю",
            ]
        },
        "fetishes_triggers.json": {
            "bondage": [
                "связал",
            ]
        },
        "fetish_names.json": {
            "bondage": "бондаж",
        },
    }

    load_json_mock = Mock(
        side_effect=lambda filename: json_data[filename],
    )

    prompts = {
        "insult_classification.txt": "INSULT PROMPT",
        "mood_classification.txt": "MOOD PROMPT",
        "fetish_role_classification.txt": "ROLE PROMPT",
        "personality/core.txt": "CORE PROMPT",
        "personality/protogen_lore.txt": "LORE PROMPT",
        "personality/body.txt": "BODY PROMPT",
        "personality/rp.txt": "RP PROMPT",
    }

    load_prompt_mock = Mock(
        side_effect=lambda filename: prompts[filename],
    )

    deepseek_mock = Mock()
    deepseek_mock.close = AsyncMock()

    deepseek_constructor_mock = Mock(
        return_value=deepseek_mock,
    )

    start_router = Mock(
        name="start_router",
    )
    help_router = Mock(
        name="help_router",
    )
    art_router = Mock(
        name="art_router",
    )
    admin_router = Mock(
        name="admin_router",
    )
    reset_router = Mock(
        name="reset_router",
    )
    unknown_router = Mock(
        name="unknown_router",
    )
    text_router = Mock(
        name="text_router",
    )

    create_start_router_mock = Mock(
        return_value=start_router,
    )
    create_help_router_mock = Mock(
        return_value=help_router,
    )
    create_art_router_mock = Mock(
        return_value=art_router,
    )
    create_admin_router_mock = Mock(
        return_value=admin_router,
    )
    create_reset_router_mock = Mock(
        return_value=reset_router,
    )
    create_unknown_router_mock = Mock(
        return_value=unknown_router,
    )
    create_text_router_mock = Mock(
        return_value=text_router,
    )

    set_commands_mock = AsyncMock()

    monkeypatch.setattr(
        main_module,
        "load_settings",
        load_settings_mock,
    )
    monkeypatch.setattr(
        main_module,
        "setup_logging",
        setup_logging_mock,
    )
    monkeypatch.setattr(
        main_module,
        "Bot",
        bot_constructor_mock,
    )
    monkeypatch.setattr(
        main_module,
        "Dispatcher",
        dispatcher_constructor_mock,
    )
    monkeypatch.setattr(
        main_module,
        "register_error_handler",
        register_error_handler_mock,
    )
    monkeypatch.setattr(
        main_module,
        "UserStateStore",
        user_state_store_constructor_mock,
    )
    monkeypatch.setattr(
        main_module,
        "load_json",
        load_json_mock,
    )
    monkeypatch.setattr(
        main_module,
        "load_prompt",
        load_prompt_mock,
    )
    monkeypatch.setattr(
        main_module,
        "DeepSeekService",
        deepseek_constructor_mock,
    )
    monkeypatch.setattr(
        main_module,
        "create_start_router",
        create_start_router_mock,
    )
    monkeypatch.setattr(
        main_module,
        "create_help_router",
        create_help_router_mock,
    )
    monkeypatch.setattr(
        main_module,
        "create_art_router",
        create_art_router_mock,
    )
    monkeypatch.setattr(
        main_module,
        "create_admin_router",
        create_admin_router_mock,
    )
    monkeypatch.setattr(
        main_module,
        "create_reset_router",
        create_reset_router_mock,
    )
    monkeypatch.setattr(
        main_module,
        "create_unknown_command_router",
        create_unknown_router_mock,
    )
    monkeypatch.setattr(
        main_module,
        "create_text_router",
        create_text_router_mock,
    )
    monkeypatch.setattr(
        main_module,
        "set_commands",
        set_commands_mock,
    )

    monkeypatch.setattr(
        main_module,
        "UserStateRepository",
        user_state_repository_constructor_mock,
    )

    asyncio.run(
        main_module.main(),
    )

    load_settings_mock.assert_called_once_with()

    assert load_prompt_mock.call_args_list == [
        call("insult_classification.txt"),
        call("mood_classification.txt"),
        call("fetish_role_classification.txt"),
        call("personality/core.txt"),
        call("personality/protogen_lore.txt"),
        call("personality/body.txt"),
        call("personality/rp.txt"),
    ]

    setup_logging_mock.assert_called_once_with(
        "INFO",
    )

    user_state_repository_constructor_mock.assert_called_once_with(
        tmp_path,
    )

    user_state_store_constructor_mock.assert_called_once_with(
        history_limit=12,
        retention_seconds=3600.0,
        persistence=user_state_repository,
    )

    bot_constructor_mock.assert_called_once_with(
        token="telegram-token",
    )

    register_error_handler_mock.assert_called_once_with(
        dispatcher_mock,
    )

    deepseek_constructor_mock.assert_called_once_with(
        api_key="deepseek-key",
        base_url="https://api.test.local",
        model="test-model",
    )

    create_reset_router_mock.assert_called_once()

    assert dispatcher_mock.include_router.call_count == 7

    dispatcher_mock.include_router.assert_any_call(
        start_router,
    )
    dispatcher_mock.include_router.assert_any_call(
        help_router,
    )
    dispatcher_mock.include_router.assert_any_call(
        art_router,
    )
    dispatcher_mock.include_router.assert_any_call(
        admin_router,
    )
    dispatcher_mock.include_router.assert_any_call(
        reset_router,
    )
    dispatcher_mock.include_router.assert_any_call(
        unknown_router,
    )
    dispatcher_mock.include_router.assert_any_call(
        text_router,
    )

    bot_mock.delete_webhook.assert_awaited_once_with(
        drop_pending_updates=True,
    )

    set_commands_mock.assert_awaited_once_with(
        bot_mock,
    )

    dispatcher_mock.start_polling.assert_awaited_once_with(
        bot_mock,
    )

    deepseek_mock.close.assert_awaited_once_with()
    bot_mock.session.close.assert_awaited_once_with()
