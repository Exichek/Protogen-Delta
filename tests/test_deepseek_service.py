"""Тесты сервиса DeepSeek."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import protogen_delta.services.deepseek as deepseek_module
from protogen_delta.services.deepseek import (
    DeepSeekAPIError,
    DeepSeekAuthError,
    DeepSeekConnectionError,
    DeepSeekRateLimitError,
    DeepSeekService,
    DeepSeekTimeoutError,
)


def _create_service(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    DeepSeekService,
    Mock,
    AsyncMock,
    AsyncMock,
]:
    """Создать DeepSeekService с подменённым клиентом OpenAI."""
    create_mock = AsyncMock()
    close_mock = AsyncMock()

    client_mock = Mock()
    client_mock.chat = Mock()
    client_mock.chat.completions = Mock()
    client_mock.chat.completions.create = create_mock
    client_mock.close = close_mock
    client_mock.with_options = Mock(
        return_value=client_mock,
    )

    constructor_mock = Mock(
        return_value=client_mock,
    )

    monkeypatch.setattr(
        deepseek_module,
        "AsyncOpenAI",
        constructor_mock,
    )

    service = DeepSeekService(
        api_key="test-api-key",
        base_url="https://api.test.local",
        model="test-model",
    )

    return (
        service,
        constructor_mock,
        create_mock,
        close_mock,
    )


def _create_response(
    content: str | None,
) -> SimpleNamespace:
    """Создать минимальный ответ, похожий на ответ OpenAI API."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                )
            )
        ]
    )


def _create_request() -> Mock:
    """Создать HTTP-запрос для исключений OpenAI SDK."""
    return Mock()


def _create_error_response(
    status_code: int,
) -> Mock:
    """Создать HTTP-ответ для ошибок OpenAI SDK."""
    response = Mock()
    response.status_code = status_code
    response.request = _create_request()
    return response


def test_deepseek_service_creates_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сервис должен создавать OpenAI-клиент с нужными настройками."""
    (
        _,
        constructor_mock,
        _,
        _,
    ) = _create_service(monkeypatch)

    constructor_mock.assert_called_once_with(
        api_key="test-api-key",
        base_url="https://api.test.local",
        timeout=15.0,
        max_retries=1,
    )


def test_deepseek_chat_sends_expected_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обычный запрос должен передавать модель, промпт и сообщение."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.return_value = _create_response(
        "Ответ DeepSeek",
    )

    result = asyncio.run(
        service.chat(
            system_prompt="SYSTEM",
            user_message="Привет",
        )
    )

    assert result == "Ответ DeepSeek"

    create_mock.assert_awaited_once_with(
        model="test-model",
        messages=[
            {
                "role": "system",
                "content": "SYSTEM",
            },
            {
                "role": "user",
                "content": "Привет",
            },
        ],
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )


def test_deepseek_chat_returns_empty_string_for_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Отсутствующий текст ответа должен превращаться в пустую строку."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.return_value = _create_response(None)

    result = asyncio.run(
        service.chat(
            system_prompt="SYSTEM",
            user_message="Привет",
        )
    )

    assert result == ""


def test_deepseek_classify_normalizes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Классификатор должен очищать и приводить ответ к нижнему регистру."""
    (
        service,
        constructor_mock,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.return_value = _create_response(
        "  ACTIVE \n",
    )

    result = asyncio.run(
        service.classify(
            system_prompt="CLASSIFIER",
            user_message="Текст",
        )
    )

    assert result == "active"

    constructor_mock.return_value.with_options.assert_called_once_with(
        timeout=5.0,
        max_retries=0,
    )

    create_mock.assert_awaited_once_with(
        model="test-model",
        messages=[
            {
                "role": "system",
                "content": "CLASSIFIER",
            },
            {
                "role": "user",
                "content": "Текст",
            },
        ],
        max_tokens=5,
        temperature=0,
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )


def test_deepseek_classify_returns_empty_string_for_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Пустой ответ классификатора должен безопасно превращаться в строку."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.return_value = _create_response(None)

    result = asyncio.run(
        service.classify(
            system_prompt="CLASSIFIER",
            user_message="Текст",
        )
    )

    assert result == ""


def test_deepseek_service_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Закрытие сервиса должно закрывать HTTP-клиент."""
    (
        service,
        _,
        _,
        close_mock,
    ) = _create_service(monkeypatch)

    asyncio.run(service.close())

    close_mock.assert_awaited_once_with()


def test_deepseek_service_rejects_invalid_timeout() -> None:
    """Timeout должен быть положительным числом."""
    with pytest.raises(
        ValueError,
        match="timeout",
    ):
        DeepSeekService(
            api_key="test-api-key",
            base_url="https://api.test.local",
            model="test-model",
            timeout=0,
        )


def test_deepseek_service_rejects_negative_retries() -> None:
    """Количество повторных попыток не может быть отрицательным."""
    with pytest.raises(
        ValueError,
        match="max_retries",
    ):
        DeepSeekService(
            api_key="test-api-key",
            base_url="https://api.test.local",
            model="test-model",
            max_retries=-1,
        )


def test_deepseek_chat_translates_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout OpenAI SDK должен превращаться в ошибку DeepSeek."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.side_effect = deepseek_module.APITimeoutError(
        request=_create_request(),
    )

    with pytest.raises(
        DeepSeekTimeoutError,
        match="не ответил вовремя",
    ):
        asyncio.run(
            service.chat(
                system_prompt="SYSTEM",
                user_message="Привет",
            )
        )


def test_deepseek_chat_translates_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 429 должен превращаться в ошибку лимита DeepSeek."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.side_effect = deepseek_module.RateLimitError(
        "Rate limit",
        response=_create_error_response(429),
        body=None,
    )

    with pytest.raises(
        DeepSeekRateLimitError,
        match="лимит запросов",
    ):
        asyncio.run(
            service.chat(
                system_prompt="SYSTEM",
                user_message="Привет",
            )
        )


def test_deepseek_chat_translates_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP 401 должен превращаться в ошибку доступа DeepSeek."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.side_effect = deepseek_module.AuthenticationError(
        "Unauthorized",
        response=_create_error_response(401),
        body=None,
    )

    with pytest.raises(
        DeepSeekAuthError,
        match="авторизации",
    ):
        asyncio.run(
            service.chat(
                system_prompt="SYSTEM",
                user_message="Привет",
            )
        )


def test_deepseek_chat_translates_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Сетевая ошибка должна превращаться в ошибку соединения DeepSeek."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.side_effect = deepseek_module.APIConnectionError(
        request=_create_request(),
    )

    with pytest.raises(
        DeepSeekConnectionError,
        match="подключиться",
    ):
        asyncio.run(
            service.chat(
                system_prompt="SYSTEM",
                user_message="Привет",
            )
        )


def test_deepseek_chat_translates_status_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Прочая HTTP-ошибка должна сохранять статус-код DeepSeek."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.side_effect = deepseek_module.APIStatusError(
        "Server error",
        response=_create_error_response(500),
        body=None,
    )

    with pytest.raises(
        DeepSeekAPIError,
        match="ошибку API",
    ) as exc_info:
        asyncio.run(
            service.chat(
                system_prompt="SYSTEM",
                user_message="Привет",
            )
        )

    assert exc_info.value.status_code == 500
