"""Сервис для работы с DeepSeek API."""

from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from protogen_delta.core.user_state import ConversationTurn

_CLASSIFY_TIMEOUT = 5.0
_CLASSIFY_MAX_RETRIES = 0


class DeepSeekError(RuntimeError):
    """Базовая ошибка при работе с DeepSeek."""


class DeepSeekTimeoutError(DeepSeekError):
    """DeepSeek не успел ответить за установленное время."""


class DeepSeekRateLimitError(DeepSeekError):
    """DeepSeek отклонил запрос из-за ограничения частоты."""


class DeepSeekAuthError(DeepSeekError):
    """DeepSeek отклонил запрос из-за проблем с доступом."""


class DeepSeekConnectionError(DeepSeekError):
    """Не удалось установить соединение с DeepSeek."""


class DeepSeekAPIError(DeepSeekError):
    """DeepSeek вернул необработанную ошибку API."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ) -> None:
        """Сохранить сообщение и HTTP-код ошибки."""
        super().__init__(message)
        self.status_code = status_code


def _translate_openai_error(error: OpenAIError) -> DeepSeekError:
    """Преобразовать исключение OpenAI SDK в ошибку DeepSeek."""
    if isinstance(error, APITimeoutError):
        return DeepSeekTimeoutError("DeepSeek не ответил вовремя")

    if isinstance(error, RateLimitError):
        return DeepSeekRateLimitError("Превышен лимит запросов DeepSeek")

    if isinstance(
        error,
        (AuthenticationError, PermissionDeniedError),
    ):
        return DeepSeekAuthError("Ошибка авторизации DeepSeek")

    if isinstance(error, APIConnectionError):
        return DeepSeekConnectionError("Не удалось подключиться к DeepSeek")

    if isinstance(error, APIStatusError):
        return DeepSeekAPIError(
            "DeepSeek вернул ошибку API",
            status_code=error.status_code,
        )

    return DeepSeekAPIError("Неизвестная ошибка DeepSeek API")


class DeepSeekService:
    """Выполнять текстовые запросы к DeepSeek."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 15.0,
        max_retries: int = 1,
    ) -> None:
        """Инициализировать клиент DeepSeek."""
        if timeout <= 0:
            raise ValueError("timeout должен быть больше нуля")

        if max_retries < 0:
            raise ValueError("max_retries не может быть отрицательным")

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[ConversationTurn] = (),
    ) -> str:
        """Получить обычный текстовый ответ модели с учётом истории диалога."""
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        for turn in history:
            messages.append(
                {
                    "role": "user",
                    "content": turn.user_message,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.assistant_message,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                extra_body={
                    "thinking": {
                        "type": "disabled",
                    }
                },
            )
        except OpenAIError as error:
            raise _translate_openai_error(error) from error

        return response.choices[0].message.content or ""

    async def classify(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Получить короткий ответ модели для классификации текста."""
        client = self._client.with_options(
            timeout=_CLASSIFY_TIMEOUT,
            max_retries=_CLASSIFY_MAX_RETRIES,
        )

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_message,
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
        except OpenAIError as error:
            raise _translate_openai_error(error) from error

        return (response.choices[0].message.content or "").strip().lower()

    async def close(self) -> None:
        """Закрыть HTTP-клиент DeepSeek."""
        await self._client.close()
