"""Уточнения должны повторять пиксели, сохраняя границы личного диалога."""

import asyncio
from dataclasses import replace

import pytest
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.core.reply_capture import ReplySource, current_reply_sources
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.deepseek import DeepSeekAPIError, ImageInput
from protogen_delta.services.recent_visuals import RecentVisuals

IMAGE = ImageInput(b"original-pixels", "image/png", "фотографию")
TURN = ConversationTurn("Посмотри", "Похоже на ноутбук")


@pytest.mark.parametrize(
    "text",
    [
        "Там нет ноутбука блять, где ты его увидел там)",
        "Это не ноутбук, это кот",
        "Пересмотри, пожалуйста",
        "Что справа?",
        "Что написано на скриншоте?",
        "На фото два кота",
        "На рисунке другой персонаж",
        "Ты ошибся, присмотрись",
    ],
)
def test_corrections_reuse_exact_original_images(text: str) -> None:
    cache = RecentVisuals()
    album = (IMAGE, replace(IMAGE, data=b"second-picture"))
    cache.remember(1, album, TURN)
    assert cache.recall(1, (TURN,), text) == album


@pytest.mark.parametrize(
    "text", ["Спасибо", "Как дела?", "Завтра пришлю фото", "Давай про код"]
)
def test_new_topic_discards_old_image(text: str) -> None:
    cache = RecentVisuals()
    cache.remember(1, (IMAGE,), TURN)
    assert not cache.recall(1, (TURN,), text)
    assert not cache.recall(1, (TURN,), "Пересмотри")


def test_expiry_is_fixed_and_user_isolation_preserved() -> None:
    clock = [100.0]
    cache = RecentVisuals(clock=lambda: clock[0], ttl=10)
    cache.remember(1, (IMAGE,), TURN)
    assert not cache.recall(2, (TURN,), "Пересмотри")
    clock[0] = 109
    assert cache.recall(1, (TURN,), "Пересмотри")
    clock[0] = 110
    assert not cache.recall(1, (TURN,), "Пересмотри")


@pytest.mark.parametrize(
    "history",
    [
        (),
        (replace(TURN),),
        (replace(TURN, context_closed=True),),
        (TURN, ConversationTurn("Закрыли", "Хорошо", context_closed=True)),
        (TURN, ConversationTurn("", "Инициатива", proactive=True)),
        (TURN,) * 0 + (ConversationTurn("x", "y"),) * 3,
    ],
)
def test_reset_reloaded_state_and_stale_context_cannot_reuse_bytes(
    history: tuple[ConversationTurn, ...],
) -> None:
    cache = RecentVisuals()
    cache.remember(1, (IMAGE,), TURN)
    assert not cache.recall(1, history, "Пересмотри")


def test_closed_older_scene_does_not_block_new_picture() -> None:
    cache = RecentVisuals()
    cache.remember(1, (IMAGE,), TURN)
    assert cache.recall(1, (replace(TURN, context_closed=True), TURN), "Пересмотри")


def test_memory_limits_replace_and_evict_whole_albums() -> None:
    cache = RecentVisuals(max_bytes=10, max_item_bytes=6, max_entries=2)
    small = replace(IMAGE, data=b"12345")
    cache.remember(1, (small,), TURN)
    cache.remember(2, (small,), TURN)
    cache.remember(3, (small,), TURN)
    assert not cache.recall(1, (TURN,), "Пересмотри")
    assert cache.recall(2, (TURN,), "Пересмотри")
    cache.remember(2, (small, small), TURN)
    assert not cache.recall(2, (TURN,), "Пересмотри")
    cache.remember(3, (), TURN)
    assert not cache.recall(3, (TURN,), "Пересмотри")
    bytes_limited = RecentVisuals(max_bytes=6, max_item_bytes=6)
    bytes_limited.remember(1, (small,), TURN)
    bytes_limited.remember(2, (small,), TURN)
    assert not bytes_limited.recall(1, (TURN,), "Пересмотри")


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        RecentVisuals(ttl=0)


def test_expiry_releases_idle_images_without_another_message() -> None:
    async def run() -> None:
        cache = RecentVisuals(ttl=0.001)
        cache.remember(1, (IMAGE,), TURN)
        previous = cache._expiry[1]
        cache.remember(1, (IMAGE,), TURN)
        assert previous.cancelled()
        await asyncio.sleep(0.01)
        assert not cache._entries
        assert not cache._expiry

    asyncio.run(run())


def test_engine_rechecks_pixels_and_distinguishes_old_attachment() -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        model.chat.return_value = "Рядом ноутбук"
        await engine.respond(TEST_USER_ID, "Фото", images=(IMAGE,))
        model.chat.return_value = "Да, это второй кот"
        await engine.respond(TEST_USER_ID, "Там нет ноутбука")
        call = model.chat.await_args
        assert call is not None
        assert call.kwargs["images"] == (IMAGE,)
        assert call.kwargs["user_message"] == "Там нет ноутбука"
        assert "не новое вложение" in call.kwargs["system_prompt"]
        assert "не является доказательством" in call.kwargs["system_prompt"]
        assert (
            "К текущему сообщению приложено ровно" not in call.kwargs["system_prompt"]
        )
        assert len(engine._user_states.get(TEST_USER_ID).history) == 2
        # Один ответ на каждый ход; повторной платной генерации не добавлено.
        assert model.chat.await_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("scope", ["group", "forward", "document", "rp", "other"])
def test_engine_never_attaches_private_picture_in_other_context(scope: str) -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        await engine.respond(TEST_USER_ID, "Фото", images=(IMAGE,))
        owner = TEST_USER_ID
        options: dict[str, object] = {}
        if scope == "group":
            options["chat_id"] = -100
        elif scope == "forward":
            options["use_personal_facts"] = False
        elif scope == "document":
            options["attachment_text"] = "Текст файла"
        elif scope == "rp":
            engine._user_states.get(TEST_USER_ID).roleplay_active = True
        else:
            owner += 1
        await engine.respond(owner, "Пересмотри", **options)  # type: ignore[arg-type]
        call = model.chat.await_args
        assert call is not None
        assert "images" not in call.kwargs

    asyncio.run(run())


@pytest.mark.parametrize("reset", ["context", "all", "stop", "minor"])
def test_engine_reset_discards_recent_visuals(reset: str) -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        await engine.respond(TEST_USER_ID, "Фото", images=(IMAGE,))
        if reset == "context":
            await engine.reset_user_context(TEST_USER_ID)
        elif reset == "all":
            await engine.reset_user(TEST_USER_ID)
        elif reset == "stop":
            await engine.disable_roleplay(TEST_USER_ID)
        else:
            await engine.restrict_minor(TEST_USER_ID)
        await engine.respond(TEST_USER_ID, "Пересмотри")
        call = model.chat.await_args
        assert call is not None
        assert "images" not in call.kwargs

    asyncio.run(run())


def test_failed_new_image_cannot_resurrect_previous_one() -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        await engine.respond(TEST_USER_ID, "Фото", images=(IMAGE,))
        model.chat.side_effect = DeepSeekAPIError("Сбой")
        await engine.respond(
            TEST_USER_ID, "Новое фото", images=(replace(IMAGE, data=b"new"),)
        )
        model.chat.side_effect = None
        await engine.respond(TEST_USER_ID, "Пересмотри")
        call = model.chat.await_args
        assert call is not None
        assert "images" not in call.kwargs

    asyncio.run(run())


def test_capability_overview_closes_previous_visual_topic() -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        await engine.respond(TEST_USER_ID, "Фото", images=(IMAGE,))
        await engine.respond(TEST_USER_ID, "Что ты умеешь?")
        await engine.respond(TEST_USER_ID, "Пересмотри")
        call = model.chat.await_args
        assert call is not None
        assert "images" not in call.kwargs

    asyncio.run(run())


def test_reply_to_unidentified_message_cannot_reuse_wrong_image() -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        await engine.respond(TEST_USER_ID, "Фото", images=(IMAGE,))
        source = ReplySource("На фото кто?", "На фото кто?", True, True, True)
        token = current_reply_sources.set((source,))
        try:
            await engine.respond(TEST_USER_ID, source.text)
        finally:
            current_reply_sources.reset(token)
        call = model.chat.await_args
        assert call is not None
        assert "images" not in call.kwargs

    asyncio.run(run())


def test_failed_delivery_does_not_create_visual_memory() -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()

        async def fail(_: str) -> None:
            raise RuntimeError("Доставка не удалась")

        with pytest.raises(RuntimeError):
            await engine.respond(TEST_USER_ID, "Фото", images=(IMAGE,), deliver=fail)
        await engine.respond(TEST_USER_ID, "Пересмотри")
        call = model.chat.await_args
        assert call is not None
        assert "images" not in call.kwargs

    asyncio.run(run())
