"""Уведомление об обработке данных доступно независимо от модели и режима."""

import asyncio

import pytest
from aiogram.filters import Command
from test_simple_handlers import _create_message_mock

from protogen_delta.core.user_state import ContentMode
from protogen_delta.handlers.help import create_help_router
from protogen_delta.services.functions_catalog import functions_text
from protogen_delta.services.privacy_notice import DATA_NOTICE, PRIVACY_TEXT


def test_privacy_command_is_local_and_explains_reset_exceptions() -> None:
    router = create_help_router()
    handler = next(
        h
        for h in router.message.handlers
        if h.filters
        and isinstance(h.filters[0].callback, Command)
        and "privacy" in h.filters[0].callback.commands
    )
    message, answer, _ = _create_message_mock()
    asyncio.run(handler.callback(message))
    answer.assert_awaited_once_with(PRIVACY_TEXT, parse_mode=None)
    assert "внешний API" in PRIVACY_TEXT
    assert "по умолчанию он выключен" in PRIVACY_TEXT
    assert (
        "возрастного ограничения" in PRIVACY_TEXT and "Резервные копии" in PRIVACY_TEXT
    )
    assert "/reset" in PRIVACY_TEXT


@pytest.mark.parametrize("mode", ["unselected", "soft", "adult"])
def test_every_help_mode_contains_data_notice(mode: ContentMode) -> None:
    text = functions_text(mode)
    assert DATA_NOTICE in text and "/privacy —" in text
