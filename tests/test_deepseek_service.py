"""Тесты сервиса DeepSeek."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import protogen_delta.services.deepseek as deepseek_module
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.deepseek import (
    DeepSeekAPIError,
    DeepSeekAuthError,
    DeepSeekConnectionError,
    DeepSeekRateLimitError,
    DeepSeekService,
    DeepSeekTimeoutError,
    ImageInput,
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
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Создать минимальный ответ, похожий на ответ OpenAI API."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                )
            )
        ],
        usage=usage,
    )


def _create_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cache_hit_tokens: int | None = None,
    cache_miss_tokens: int | None = None,
) -> SimpleNamespace:
    """Создать usage-данные ответа DeepSeek."""
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_cache_hit_tokens=cache_hit_tokens,
        prompt_cache_miss_tokens=cache_miss_tokens,
    )


def _create_request() -> Mock:
    """Создать HTTP-запрос для исключений OpenAI SDK."""
    return Mock()


def test_fact_extraction_uses_bounded_json_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, constructor, create, _ = _create_service(monkeypatch)
    create.return_value = _create_response(' {"changes":[]} ')
    assert (
        asyncio.run(service.extract_user_facts("fact rules", "data"))
        == '{"changes":[]}'
    )
    constructor.return_value.with_options.assert_called_once_with(
        timeout=6.0, max_retries=0
    )
    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "fact rules"},
        {"role": "user", "content": "data"},
    ]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] == 600 and kwargs["temperature"] == 0
    assert "tools" not in kwargs
    create.return_value = _create_response(None)
    assert asyncio.run(service.extract_user_facts("rules", "data")) == ""
    create.side_effect = deepseek_module.APITimeoutError(_create_request())
    with pytest.raises(DeepSeekTimeoutError):
        asyncio.run(service.extract_user_facts("rules", "data"))


def _create_error_response(
    status_code: int,
) -> Mock:
    """Создать HTTP-ответ для ошибок OpenAI SDK."""
    response = Mock()
    response.status_code = status_code
    response.request = _create_request()
    return response


def test_visual_observation_is_json_without_chat_history_or_tools(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service, _, create, _ = _create_service(monkeypatch)
    service._tools = Mock()
    images = (ImageInput(b"first", "image/png"), ImageInput(b"second", "image/jpeg"))
    create.return_value = _create_response(
        '{"readable":true}',
        usage=_create_usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )
    with caplog.at_level(logging.INFO, logger="protogen_delta.services.deepseek"):
        result = asyncio.run(
            service.analyze_visual_features("PRIVATE RULES", "PRIVATE CAPTION", images)
        )
    assert result == '{"readable":true}'
    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] == 4096 and "tools" not in kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "PRIVATE RULES"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "PRIVATE CAPTION"},
                *[
                    {"type": "image_url", "image_url": {"url": image.data_url()}}
                    for image in images
                ],
            ],
        },
    ]
    service._tools.available_tools_prompt.assert_not_called()
    assert "appearance_observation" in caplog.text and "history_turns=0" in caplog.text
    assert "PRIVATE" not in caplog.text and "base64" not in caplog.text
    assert service.snapshot()["completed"] == 1


def test_visual_observation_timeout_preserves_metrics_and_translates_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, create, _ = _create_service(monkeypatch)
    create.side_effect = deepseek_module.APITimeoutError(_create_request())
    with pytest.raises(DeepSeekTimeoutError):
        asyncio.run(
            service.analyze_visual_features(
                "rules", "data", (ImageInput(b"image", "image/png"),)
            )
        )
    assert service.snapshot()["failed"] == 1 and service.snapshot()["active"] == 0


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
        max_tokens=4096,
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


def test_deepseek_chat_sends_history_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обычный запрос должен передавать историю диалога в правильном порядке."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.return_value = _create_response(
        "Новый ответ",
    )

    history = [
        ConversationTurn(
            user_message="Первое сообщение",
            assistant_message="Первый ответ",
        ),
        ConversationTurn(
            user_message="Второе сообщение",
            assistant_message="Второй ответ",
        ),
    ]

    result = asyncio.run(
        service.chat(
            system_prompt="SYSTEM",
            user_message="Текущее сообщение",
            history=history,
        )
    )

    assert result == "Новый ответ"

    create_mock.assert_awaited_once_with(
        model="test-model",
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": "SYSTEM",
            },
            {
                "role": "user",
                "content": "Первое сообщение",
            },
            {
                "role": "assistant",
                "content": "Первый ответ",
            },
            {
                "role": "user",
                "content": "Второе сообщение",
            },
            {
                "role": "assistant",
                "content": "Второй ответ",
            },
            {
                "role": "user",
                "content": "Текущее сообщение",
            },
        ],
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    )


def test_deepseek_chat_sends_inline_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Изображение должно передаваться мультимодальным блоком data URL."""
    service, _, create_mock, _ = _create_service(monkeypatch)
    create_mock.return_value = _create_response("Вижу PNG")

    result = asyncio.run(
        service.chat(
            system_prompt="SYSTEM",
            user_message="Что здесь?",
            images=(
                ImageInput(
                    data=b"\x89PNG\r\n\x1a\n",
                    mime_type="image/png",
                ),
            ),
        )
    )

    assert result == "Вижу PNG"
    call = create_mock.await_args
    assert call is not None
    content = call.kwargs["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "Что здесь?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,iVBORw0KGgo="
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


def test_compatible_provider_can_omit_deepseek_thinking_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Чужому OpenAI-совместимому API не следует слать расширение DeepSeek."""
    create_mock = AsyncMock(return_value=_create_response("Ответ"))
    client_mock = Mock()
    client_mock.chat.completions.create = create_mock
    monkeypatch.setattr(
        deepseek_module,
        "AsyncOpenAI",
        Mock(return_value=client_mock),
    )
    service = DeepSeekService(
        api_key="key",
        base_url="https://llm.example/v1",
        model="model",
        disable_thinking=False,
    )

    result = asyncio.run(service.chat("SYSTEM", "Привет"))

    assert result == "Ответ"
    call = create_mock.await_args
    assert call is not None
    assert "extra_body" not in call.kwargs


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


def test_deepseek_chat_logs_request_metrics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Обычный запрос должен логировать latency, usage и размеры контекста."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.return_value = _create_response(
        "Ответ",
        usage=_create_usage(
            prompt_tokens=1200,
            completion_tokens=80,
            total_tokens=1280,
            cache_hit_tokens=1000,
            cache_miss_tokens=200,
        ),
    )

    history = [
        ConversationTurn(
            user_message="Предыдущее сообщение",
            assistant_message="Предыдущий ответ",
        )
    ]

    with caplog.at_level(
        logging.INFO,
        logger="protogen_delta.services.deepseek",
    ):
        result = asyncio.run(
            service.chat(
                system_prompt="СЕКРЕТНЫЙ SYSTEM",
                user_message="СЕКРЕТНОЕ СООБЩЕНИЕ",
                history=history,
            )
        )

    assert result == "Ответ"

    assert "DeepSeek chat" in caplog.text
    assert "duration=" in caplog.text
    assert "prompt_tokens=1200" in caplog.text
    assert "completion_tokens=80" in caplog.text
    assert "total_tokens=1280" in caplog.text
    assert "cache_hit_tokens=1000" in caplog.text
    assert "cache_miss_tokens=200" in caplog.text
    assert "history_turns=1" in caplog.text

    assert "СЕКРЕТНЫЙ SYSTEM" not in caplog.text
    assert "СЕКРЕТНОЕ СООБЩЕНИЕ" not in caplog.text
    assert "Предыдущее сообщение" not in caplog.text
    assert "Предыдущий ответ" not in caplog.text


def test_deepseek_classify_logs_request_metrics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Классификатор должен логировать usage без содержимого запроса."""
    (
        service,
        _,
        create_mock,
        _,
    ) = _create_service(monkeypatch)

    create_mock.return_value = _create_response(
        "neutral",
        usage=_create_usage(
            prompt_tokens=300,
            completion_tokens=1,
            total_tokens=301,
            cache_hit_tokens=256,
            cache_miss_tokens=44,
        ),
    )

    with caplog.at_level(
        logging.INFO,
        logger="protogen_delta.services.deepseek",
    ):
        result = asyncio.run(
            service.classify(
                system_prompt="СЕКРЕТНЫЙ CLASSIFIER",
                user_message="СЕКРЕТНЫЙ ТЕКСТ",
            )
        )

    assert result == "neutral"

    assert "DeepSeek classify" in caplog.text
    assert "duration=" in caplog.text
    assert "prompt_tokens=300" in caplog.text
    assert "completion_tokens=1" in caplog.text
    assert "total_tokens=301" in caplog.text
    assert "cache_hit_tokens=256" in caplog.text
    assert "cache_miss_tokens=44" in caplog.text
    assert "history_turns=0" in caplog.text

    assert "СЕКРЕТНЫЙ CLASSIFIER" not in caplog.text
    assert "СЕКРЕТНЫЙ ТЕКСТ" not in caplog.text
