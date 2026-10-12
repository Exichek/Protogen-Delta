"""Ложное отрицание стикеров не отправляется и не записывается в историю."""

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from test_response_engine import TEST_USER_ID, _create_engine
from test_start_and_art_handlers import _call_handler, _create_message_mock
from test_stickers import _entry

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.start import create_start_router
from protogen_delta.repositories.stickers import StickersRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.sticker_capabilities import (
    STICKER_CAPABILITY_REPLY,
    correct_sticker_capability,
)
from protogen_delta.services.stickers import ContextualStickerService

SCREENSHOT_REPLY = (
    "Стикеры у меня всё ещё только словесные — «:3» вместо картинки, зато честно. "
    "Если найдёшь что-то поинтереснее, тащи, гляну."
)


@pytest.mark.parametrize(
    "denial",
    [
        SCREENSHOT_REPLY,
        "Я не умею отправлять стикеры.",
        "А про «стикеры, которые ты можешь использовать» — я картинки отправлять не умею.",
        "Я не могу отправлять настоящие Telegram-стикеры!",
        "У меня стикеры только текстовые, вместо картинки.",
        "У меня нет стикеров.",
        "Стикеров у меня нет, лишь текст.",
        "Я могу использовать стикеры максимум словесно.",
        "В приложении отправлять тебе готовые стикеры — нет, такой функции нет у меня.",
        "СТИКЕРЫ У МЕНЯ ТОЛЬКО МЫСЛЕННЫЕ.",
    ],
)
def test_false_denial_is_replaced_without_claiming_delivery(denial: str) -> None:
    result = correct_sticker_capability(denial)
    assert result.startswith(STICKER_CAPABILITY_REPLY)
    assert "отправлен" not in result
    if denial == SCREENSHOT_REPLY:
        assert result.endswith("Если найдёшь что-то поинтереснее, тащи, гляну.")


@pytest.mark.parametrize(
    "reply",
    [
        "У меня есть Telegram-стикеры, могу прислать.",
        "Я могу отправить стикер, но не каждый раз.",
        "Я не умею рисовать новые стикеры, зато готовые отправляю.",
        "Я не могу отправлять стикеры каждые две секунды.",
        "Сейчас не могу отправить стикер: Telegram вернул ошибку.",
        "Я не могу отправить именно этот стикер.",
        "Я не могу отправить этот стикер.",
        "Я не могу понять, почему тебе нравятся эти стикеры.",
        "Я не умею определять автора стикера без подписи.",
        "Ты спросил «у тебя только словесные стикеры?». У меня есть настоящие.",
        'В коде строка `Я не могу отправлять стикеры` и "У меня нет стикеров".',
        "```python\nreply = 'У меня нет стикеров.'\n```",
        "Я говорил, что не умею отправлять стикеры, но это было ошибкой.",
        "Раньше я не мог отправлять стикеры, сейчас могу.",
        "Я узнал, что у него нет стикеров.",
        "У моего друга нет стикеров.",
        "Я не могу скачать этот ролик. Зато могу отправлять стикеры.",
        "Я не могу отправить тебе файл больше лимита.",
        "Сегодня дождь.",
    ],
)
def test_quotes_other_people_and_real_limitations_are_preserved(reply: str) -> None:
    assert correct_sticker_capability(reply) == reply


def test_repair_preserves_paragraphs_and_does_not_duplicate_fact() -> None:
    reply = "Привет!\n\nЯ не могу отправлять стикеры. У меня нет стикеров.\nКак дела?"
    result = correct_sticker_capability(reply)
    assert result.startswith("Привет!\n\n" + STICKER_CAPABILITY_REPLY)
    assert result.endswith("\nКак дела?")
    assert result.count(STICKER_CAPABILITY_REPLY) == 1


@pytest.mark.parametrize("availability", ["safe", "adult_only", "disabled", "empty"])
def test_guard_checks_actual_pack_and_does_not_send_stickers(
    tmp_path: Path, availability: str
) -> None:
    repository = StickersRepository(tmp_path)
    if availability != "empty":
        repository.upsert(
            _entry(
                "one",
                "greeting",
                rating="adult" if availability == "adult_only" else "safe",
            )
        )
    bot = AsyncMock(spec=Bot)
    states = UserStateStore()
    service = ContextualStickerService(
        cast(Bot, bot),
        repository,
        states,
        chance=0 if availability == "disabled" else 1,
    )
    corrected = service.correct_reply(TEST_USER_ID, SCREENSHOT_REPLY)
    assert (corrected != SCREENSHOT_REPLY) == (availability == "safe")
    bot.send_sticker.assert_not_awaited()
    states.get(TEST_USER_ID).content_mode = "adult"
    if availability == "adult_only":
        assert service.correct_reply(TEST_USER_ID, SCREENSHOT_REPLY) != SCREENSHOT_REPLY


@pytest.mark.parametrize("delivery_fails", [False, True])
def test_corrected_text_is_delivered_and_committed_together(
    tmp_path: Path, delivery_fails: bool
) -> None:
    engine, _, model, *_ = _create_engine()
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("one", "greeting"))
    service = ContextualStickerService(
        cast(Bot, AsyncMock(spec=Bot)), repository, engine._user_states
    )
    engine._reply_transform = service.correct_reply
    model.chat.return_value = SCREENSHOT_REPLY
    sent: list[str] = []

    async def deliver(reply: str) -> None:
        assert not engine._user_states.get(TEST_USER_ID).history
        sent.append(reply)
        if delivery_fails:
            raise RuntimeError("delivery failed")

    async def scenario() -> None:
        if delivery_fails:
            with pytest.raises(RuntimeError, match="delivery failed"):
                await engine.respond_and_deliver(
                    TEST_USER_ID, "Как работают стикеры?", deliver
                )
        else:
            await engine.respond_and_deliver(
                TEST_USER_ID, "Как работают стикеры?", deliver
            )

    asyncio.run(scenario())
    assert sent == [correct_sticker_capability(SCREENSHOT_REPLY)]
    model.chat.assert_awaited_once()
    history = engine._user_states.get(TEST_USER_ID).history
    if delivery_fails:
        assert not history
    else:
        assert history[-1].assistant_message == sent[0]


@pytest.mark.parametrize("repeat", [False, True])
def test_start_corrects_denial_before_greeting_sticker(
    tmp_path: Path, repeat: bool
) -> None:
    users = UsersRepository(tmp_path)
    if repeat:
        users.add(123)
    repository = StickersRepository(tmp_path)
    repository.upsert(_entry("hello", "greeting"))
    states = UserStateStore()
    states.get(123).content_mode = "adult"
    bot = AsyncMock(spec=Bot)
    service = ContextualStickerService(cast(Bot, bot), repository, states)
    model = AsyncMock(spec=DeepSeekService)
    model.chat.return_value = SCREENSHOT_REPLY
    router = create_start_router(
        users,
        cast(DeepSeekService, model),
        "Приветствие",
        user_states=states,
        sticker_service=service,
    )
    message, _, answer, _ = _create_message_mock()
    asyncio.run(_call_handler(router, 0, message))
    sent_text = "\n".join(call.args[0] for call in answer.await_args_list)
    assert STICKER_CAPABILITY_REPLY in sent_text
    assert "только словесные" not in sent_text
    bot.send_sticker.assert_awaited_once()
    model.chat.assert_awaited_once()
