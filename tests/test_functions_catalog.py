"""Пользовательская справка учитывает режим и подключения без LLM."""

import asyncio
import re
from dataclasses import replace

import pytest
from aiogram.filters import Command
from test_simple_handlers import _create_message_mock

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.config.settings import Settings
from protogen_delta.core.telegram_commands import commands_for_mode
from protogen_delta.core.user_state import ContentMode, UserStateStore
from protogen_delta.handlers.help import create_help_router
from protogen_delta.handlers.start import (
    FIRST_START_FALLBACK_BODY,
    REPEAT_START_FALLBACK,
)
from protogen_delta.services.capabilities import build_capabilities
from protogen_delta.services.functions_catalog import functions_text


@pytest.mark.parametrize("mode", ["unselected", "soft", "adult"])
def test_function_commands_match_telegram_menu(mode: ContentMode) -> None:
    text = functions_text(mode)
    names = re.findall(r"^/(\w+) —", text, re.M)
    assert set(names) == {c.command for c in commands_for_mode(mode)}
    assert len(names) == len(set(names))
    assert ("/randomart" in text) == (mode == "adult")
    assert ("только safe-арты" in text) == (mode != "adult")
    assert "/funcs" in text and "100 МБ" in text
    assert "/ownhelp" not in text and "/proactive" not in text
    assert len(text) < 4096


def test_functions_reflect_connected_optional_services() -> None:
    settings = Settings(
        telegram_token="synthetic", deepseek_api_key="synthetic", art_chat_id=1
    )
    basic = functions_text(settings=settings)
    assert "сейчас сервис не подключён" in basic
    assert "отдельный поисковый сервис не подключён" in basic
    assert "Сканированные PDF" not in basic and "Музыка и звуки" not in basic
    rich = functions_text(
        "adult",
        settings=replace(
            settings,
            saucenao_api_key="synthetic",
            brave_search_api_key="synthetic",
            pdf_ocr_enabled=True,
            audio_understanding_enabled=True,
            mini_app_url="https://unused.invalid",
        ),
    )
    assert "не подключён" not in rich
    assert "Музыка и звуки" in rich and "Сканированные PDF" in rich
    assert "исправление описания и вида" in rich
    assert len(rich) < 4096


@pytest.mark.parametrize("restricted", [False, True])
def test_funcs_router_registers_alias_and_respects_age(restricted: bool) -> None:
    async def scenario() -> None:
        states = UserStateStore()
        message, answer, _ = _create_message_mock("/funcs")
        assert message.from_user is not None
        state = states.get(message.from_user.id)
        state.content_mode = "adult"
        state.age_restricted = restricted
        router = create_help_router(states)
        filters = router.message.handlers[0].filters
        assert filters is not None
        command_filter = filters[0].callback
        assert isinstance(command_filter, Command)
        assert {"help", "funcs"}.issubset(command_filter.commands)
        await router.message.handlers[0].callback(message)
        call = answer.await_args
        assert call is not None
        text = call.args[0]
        assert ("/randomart" in text) is not restricted
        assert call.kwargs["parse_mode"] is None

    asyncio.run(scenario())


def test_greetings_use_short_inventory_instead_of_generic_tool_advertising() -> None:
    settings = Settings(
        telegram_token="synthetic", deepseek_api_key="synthetic", art_chat_id=1
    )
    for connected in (settings, replace(settings, brave_search_api_key="synthetic")):
        text = build_capabilities(connected).greeting
        assert "/funcs" in text
        assert all(
            word not in text.lower() for word in ("погод", "курс", "время", "/e6")
        )
    for text in (
        load_prompt("start_greeting"),
        load_prompt("repeat_start_greeting"),
        FIRST_START_FALLBACK_BODY,
        REPEAT_START_FALLBACK,
    ):
        assert "/funcs" in text
        assert all(word not in text.lower() for word in ("погод", "курс", "/e6"))
