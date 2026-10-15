"""Изолированный транспорт оценочных прогонов с ограничением запросов."""

import hashlib
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from protogen_delta.services.deepseek import DeepSeekService


class EvaluationService(DeepSeekService):
    """Перехватывать транспорт: предпросмотр не выходит в сеть ни в одном режиме."""

    def __init__(self, *, run_model: bool, max_calls: int, **kwargs: Any) -> None:
        """Отключить повторы и разделять счётчик с копиями режима размышлений."""
        super().__init__(max_retries=0, **kwargs)
        self.trace: dict[str, Any] = {
            "run_model": run_model,
            "max_calls": max_calls,
            "requests": [],
            "budget_exceeded": False,
        }

    async def _complete(
        self, *, client: AsyncOpenAI | None = None, **kwargs: Any
    ) -> ChatCompletion:
        """Записать объём запроса; ограничение действует также на классификаторы."""
        trace = self.trace
        if len(trace["requests"]) >= trace["max_calls"]:
            trace["budget_exceeded"] = True
            raise RuntimeError("Лимит запросов оценочного прогона исчерпан")
        messages = kwargs["messages"]
        system = "\n\n".join(
            item["content"] for item in messages if item["role"] == "system"
        )
        kind = (
            "json"
            if "response_format" in kwargs
            else "classification" if kwargs["max_tokens"] == 5 else "reply"
        )
        request = {
            "kind": kind,
            "system_chars": len(system),
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "input_chars": sum(len(item["content"]) for item in messages),
            "history_messages": max(0, len(messages) - 2),
        }
        trace["requests"].append(request)
        if not trace["run_model"]:
            content = (
                '{"mood":"neutral","insult":"none","favorite":"none","reasoning":false}'
                if kind == "json"
                else (
                    "unknown"
                    if kind == "classification"
                    else "[Предпросмотр сборки: ответ модели не запрашивался.]"
                )
            )
            # Текст нужен только проверке состава, в отчёт и stdout он не попадает.
            request["preview_system"] = system
            return ChatCompletion.model_validate(
                {
                    "id": "preview",
                    "object": "chat.completion",
                    "created": 0,
                    "model": self._model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                        }
                    ],
                }
            )
        started = perf_counter()
        try:
            response = await super()._complete(
                client=(client or self._client).with_options(max_retries=0), **kwargs
            )
        except Exception as error:
            request["error"] = type(error).__name__
            raise
        finally:
            request["seconds"] = round(perf_counter() - started, 3)
        request["usage"] = response.usage.model_dump() if response.usage else None
        if kind in {"json", "classification"}:
            request["classification_response"] = (
                response.choices[0].message.content or ""
            )[:4097]
        return response
