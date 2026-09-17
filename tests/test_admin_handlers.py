"""Тесты административных обработчиков."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, call

import pytest
from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

import protogen_delta.handlers.admin as admin_module
from protogen_delta.core.state import BotState
from protogen_delta.handlers.admin import create_admin_router
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.users import UsersRepository


async def _call_handler(
    router: Router,
    handler_name: str,
    message: Message,
) -> None:
    """Вызвать message-handler роутера по имени функции."""
    for handler in router.message.handlers:
        if handler.callback.__name__ == handler_name:
            await handler.callback(message)
            return

    raise AssertionError(f"Handler не найден: {handler_name}")


def _create_message_mock(
    *,
    user_id: int | None = 123,
    text: str | None = None,
) -> tuple[Message, Mock, AsyncMock, AsyncMock]:
    """Создать Message с нужными Telegram-полями и async-методами."""
    message_mock = Mock()

    if user_id is None:
        message_mock.from_user = None
    else:
        message_mock.from_user = SimpleNamespace(id=user_id)

    message_mock.text = text

    answer_mock = AsyncMock()
    answer_photo_mock = AsyncMock()

    message_mock.answer = answer_mock
    message_mock.answer_photo = answer_photo_mock

    return (
        cast(Message, message_mock),
        message_mock,
        answer_mock,
        answer_photo_mock,
    )


def _create_router(
    images_mock: Mock,
    users_mock: Mock,
    bot_state: BotState | None = None,
) -> Router:
    """Создать админский роутер с тестовыми зависимостями."""
    return create_admin_router(
        images_repository=cast(
            ImagesRepository,
            images_mock,
        ),
        users_repository=cast(
            UsersRepository,
            users_mock,
        ),
        bot_state=bot_state or BotState(),
        admin_ids=frozenset({123}),
    )


def test_admin_command_denies_access_for_regular_user() -> None:
    """Обычный пользователь не должен получать доступ к админским командам."""
    images_mock = Mock(spec=ImagesRepository)
    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        user_id=999,
        text="/ping",
    )

    asyncio.run(
        _call_handler(
            router,
            "ping",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "⛔ У тебя нет доступа к этой команде.",
    )


def test_admin_command_denies_access_without_user() -> None:
    """Сообщение без пользователя не должно считаться админским."""
    images_mock = Mock(spec=ImagesRepository)
    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        user_id=None,
        text="/ping",
    )

    asyncio.run(
        _call_handler(
            router,
            "ping",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "⛔ У тебя нет доступа к этой команде.",
    )


def test_list_images_reports_empty_database() -> None:
    """При пустой базе /listimages должен сообщать об этом."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = []

    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock(
        text="/listimages",
    )

    asyncio.run(
        _call_handler(
            router,
            "list_images",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "📂 База артов пуста.",
    )
    answer_photo_mock.assert_not_awaited()


def test_list_images_sends_requested_last_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Команда должна показывать последние N артов от новых к старым."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = [
        "file-id-1",
        "file-id-2",
        "file-id-3",
    ]

    users_mock = Mock(spec=UsersRepository)

    sleep_mock = AsyncMock()

    monkeypatch.setattr(
        admin_module.asyncio,
        "sleep",
        sleep_mock,
    )

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock(
        text="/listimages 2",
    )

    asyncio.run(
        _call_handler(
            router,
            "list_images",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "📂 Показываю: 2 арта.",
    )

    answer_photo_mock.assert_has_awaits(
        [
            call(
                "file-id-3",
                caption="<code>file-id-3</code>",
                parse_mode="HTML",
            ),
            call(
                "file-id-2",
                caption="<code>file-id-2</code>",
                parse_mode="HTML",
            ),
        ]
    )

    assert answer_photo_mock.await_count == 2
    assert sleep_mock.await_count == 1


def test_list_images_uses_one_image_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без числа /listimages должен показывать один последний арт."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = [
        "file-id-1",
        "file-id-2",
    ]

    users_mock = Mock(spec=UsersRepository)

    monkeypatch.setattr(
        admin_module.asyncio,
        "sleep",
        AsyncMock(),
    )

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock(
        text="/listimages",
    )

    asyncio.run(
        _call_handler(
            router,
            "list_images",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "📂 Показываю: 1 арт.",
    )

    answer_photo_mock.assert_awaited_once_with(
        "file-id-2",
        caption="<code>file-id-2</code>",
        parse_mode="HTML",
    )


def test_list_images_handles_photo_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ошибка Telegram не должна раскрывать технические детали."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = [
        "broken-file-id",
    ]

    users_mock = Mock(spec=UsersRepository)

    monkeypatch.setattr(
        admin_module.asyncio,
        "sleep",
        AsyncMock(),
    )

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock(
        text="/listimages 1",
    )

    answer_photo_mock.side_effect = TelegramAPIError(
        method=Mock(),
        message="Telegram secret error",
    )

    asyncio.run(
        _call_handler(
            router,
            "list_images",
            message,
        )
    )

    assert answer_mock.await_count == 2

    answer_mock.assert_has_awaits(
        [
            call("📂 Показываю: 1 арт."),
            call("⚠️ Не удалось отправить арт с ID: broken-file-id"),
        ]
    )

    assert "Telegram secret error" not in str(answer_mock.await_args_list)


def test_remove_image_requires_ids() -> None:
    """Команда удаления без ID должна показывать подсказку."""
    images_mock = Mock(spec=ImagesRepository)
    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/removeimage",
    )

    asyncio.run(
        _call_handler(
            router,
            "remove_image",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "⚠️ Укажи ID артов через запятую.\n" "Пример: /removeimage id1,id2,id3"
    )

    images_mock.remove.assert_not_called()


def test_remove_image_reports_removed_and_missing() -> None:
    """Удаление должно отдельно считать удалённые и отсутствующие арты."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.remove.side_effect = [
        True,
        False,
        True,
    ]

    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/removeimage id1, id2, id3",
    )

    asyncio.run(
        _call_handler(
            router,
            "remove_image",
            message,
        )
    )

    assert images_mock.remove.call_args_list == [
        call("id1"),
        call("id2"),
        call("id3"),
    ]

    answer_mock.assert_awaited_once_with("✅ Удалено: 2 арта\n" "⚠️ Не найдено: 1 арт")


def test_remove_image_handles_empty_id_list() -> None:
    """Пустой список ID после запятых не должен приводить к ошибке."""
    images_mock = Mock(spec=ImagesRepository)
    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/removeimage , ,",
    )

    asyncio.run(
        _call_handler(
            router,
            "remove_image",
            message,
        )
    )

    images_mock.remove.assert_not_called()

    answer_mock.assert_awaited_once_with(
        "⚠️ Укажи ID артов через запятую.\n" "Пример: /removeimage id1,id2,id3"
    )


def test_art_count_reports_empty_database() -> None:
    """При нуле артов команда /artcount должна сообщать о пустой базе."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.count.return_value = 0

    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/artcount",
    )

    asyncio.run(
        _call_handler(
            router,
            "art_count",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "📂 База артов пуста.",
    )


def test_art_count_reports_number_of_images() -> None:
    """Команда /artcount должна показывать количество артов."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.count.return_value = 42

    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/artcount",
    )

    asyncio.run(
        _call_handler(
            router,
            "art_count",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "📂 В базе 42 арта.",
    )


def test_status_reports_bot_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Команда /status должна показывать uptime и состояние бота."""
    images_mock = Mock(spec=ImagesRepository)

    users_mock = Mock(spec=UsersRepository)
    users_mock.count.return_value = 5

    bot_state = BotState(
        reply_count=17,
        mood="sweet",
        start_time=1000.0,
    )

    monkeypatch.setattr(
        admin_module.time,
        "monotonic",
        lambda: 4661.0,
    )

    router = _create_router(
        images_mock,
        users_mock,
        bot_state,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/status",
    )

    asyncio.run(
        _call_handler(
            router,
            "status",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "📊 Статус бота:\n"
        "• Uptime: 01:01:01\n"
        "• Пользователей: 5\n"
        "• Ответов отправлено: 17\n"
        "• Настроение: sweet"
    )


def test_own_help_sends_admin_commands() -> None:
    """Команда /ownhelp должна показывать административную справку."""
    images_mock = Mock(spec=ImagesRepository)
    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/ownhelp",
    )

    asyncio.run(
        _call_handler(
            router,
            "own_help",
            message,
        )
    )

    answer_mock.assert_awaited_once()

    await_args = answer_mock.await_args

    assert await_args is not None

    reply = await_args.args[0]

    assert "/listimages" in reply
    assert "/removeimage" in reply
    assert "/artcount" in reply
    assert "/status" in reply
    assert "/ping" in reply
    assert "/ownhelp" in reply


def test_ping_returns_pong() -> None:
    """Администратор должен получать ответ на /ping."""
    images_mock = Mock(spec=ImagesRepository)
    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/ping",
    )

    asyncio.run(
        _call_handler(
            router,
            "ping",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "🏓 Pong от админского роутера!",
    )


@pytest.mark.parametrize(
    "text",
    [
        "/listimages abc",
        "/listimages 0",
        "/listimages -1",
        "/listimages 1.5",
    ],
)
def test_list_images_rejects_invalid_count(
    text: str,
) -> None:
    """Некорректное количество артов должно отклоняться."""
    images_mock = Mock(spec=ImagesRepository)
    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock(
        text=text,
    )

    asyncio.run(
        _call_handler(
            router,
            "list_images",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "⚠️ Укажи положительное целое количество артов.\n" "Пример: /listimages 10"
    )

    images_mock.get_all.assert_not_called()
    answer_photo_mock.assert_not_awaited()


def test_list_images_limits_count_to_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Количество отправляемых артов должно ограничиваться максимумом."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.get_all.return_value = [f"file-id-{index}" for index in range(250)]

    users_mock = Mock(spec=UsersRepository)

    monkeypatch.setattr(
        admin_module.asyncio,
        "sleep",
        AsyncMock(),
    )

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, answer_photo_mock = _create_message_mock(
        text="/listimages 999",
    )

    asyncio.run(
        _call_handler(
            router,
            "list_images",
            message,
        )
    )

    answer_mock.assert_awaited_once_with(
        "📂 Показываю: 200 артов.",
    )

    assert answer_photo_mock.await_count == 200


def test_remove_image_ignores_duplicate_ids() -> None:
    """Одинаковый file_id должен обрабатываться только один раз."""
    images_mock = Mock(spec=ImagesRepository)
    images_mock.remove.return_value = True

    users_mock = Mock(spec=UsersRepository)

    router = _create_router(
        images_mock,
        users_mock,
    )

    message, _, answer_mock, _ = _create_message_mock(
        text="/removeimage id1,id1,id2,id1",
    )

    asyncio.run(
        _call_handler(
            router,
            "remove_image",
            message,
        )
    )

    assert images_mock.remove.call_args_list == [
        call("id1"),
        call("id2"),
    ]

    answer_mock.assert_awaited_once_with(
        "✅ Удалено: 2 арта",
    )


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "арт"),
        (2, "арта"),
        (5, "артов"),
        (11, "артов"),
        (21, "арт"),
        (22, "арта"),
        (25, "артов"),
    ],
)
def test_art_word_uses_correct_russian_form(
    count: int,
    expected: str,
) -> None:
    """Количество должно использовать правильную форму слова «арт»."""
    assert admin_module._art_word(count) == expected
