"""Полный обмен assistant/tool с сохранением идентификаторов вызовов."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletionMessage

import protogen_delta.services.deepseek as module
from protogen_delta.services.deepseek import DeepSeekAPIError, DeepSeekService
from protogen_delta.services.tools import Tool, ToolExecutor, ToolRegistry


def response(count: int = 0, args: str = "{}") -> SimpleNamespace:
    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": "Итог" if not count else None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "test", "arguments": args},
                }
                for i in range(count)
            ]
            or None,
        }
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.mark.parametrize("invalid", [False, True])
def test_loop_returns_tool_results_with_matching_id(
    monkeypatch: pytest.MonkeyPatch, invalid: bool
) -> None:
    handler = AsyncMock(return_value={"value": 123})
    client = Mock()
    client.chat.completions.create = AsyncMock(
        side_effect=[response(1, "{" if invalid else "{}"), response()]
    )
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "key",
        "https://test.local",
        "test",
        tools=ToolExecutor(ToolRegistry([Tool("test", "test", {}, (), handler)])),
    )
    assert asyncio.run(service.chat("system", "question")) == "Итог"
    messages = client.chat.completions.create.await_args_list[-1].kwargs["messages"]
    output = next(m for m in messages if m["role"] == "tool")
    assert output["tool_call_id"] == "call_0"
    assert json.loads(output["content"])["ok"] is not invalid
    assert handler.await_count == (0 if invalid else 1)
    assert (
        client.chat.completions.create.await_args.kwargs["extra_body"]["thinking"][
            "type"
        ]
        == "disabled"
    )


def test_loop_caps_execution_and_stops_ignoring_model_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = AsyncMock(return_value={})
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=response(4))
    monkeypatch.setattr(module, "AsyncOpenAI", Mock(return_value=client))
    service = DeepSeekService(
        "key",
        "https://test.local",
        "test",
        tools=ToolExecutor(ToolRegistry([Tool("test", "test", {}, (), handler)])),
    )
    assert "Не удалось" in asyncio.run(service.chat("system", "question"))
    assert handler.await_count == 6
    assert client.chat.completions.create.await_count == 4
    assert client.chat.completions.create.await_args.kwargs["tool_choice"] == "none"
    client.chat.completions.create.return_value = response(7)
    with pytest.raises(DeepSeekAPIError):
        asyncio.run(service.chat("system", "question"))
