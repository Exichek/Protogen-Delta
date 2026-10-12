"""Comparison never retries, leaks SDK errors, or exceeds its request budget."""

import asyncio
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx2 as httpx
import pytest
from appearance_fixtures import observation, verified
from openai import RateLimitError
from openai.types.chat import ChatCompletion

from protogen_delta.services.deepseek import ImageInput

_script = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "compare_vision.py")
)
ComparisonModel = _script["ComparisonModel"]
Case = _script["Case"]
evaluate_case = _script["evaluate_case"]


def _completion(text: str, reason: str = "stop") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "test-only",
            "object": "chat.completion",
            "created": 0,
            "model": "test-vision",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": reason,
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }
    )


@pytest.mark.parametrize("provider", ["deepseek", "gemini"])
def test_same_analyzer_records_both_passes_without_history(provider: str) -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(observation()),
            _completion(
                verified("Тёмный персонаж со светлой грудью и длинным хвостом.")
            ),
        ]
    )
    model = ComparisonModel(client, "test-model", provider)
    case = Case("test", "Neutral reference", (ImageInput(b"image", "image/png"),), ())
    result = asyncio.run(evaluate_case(model, case))
    assert result["ok"] is True
    assert len(result["calls"]) == 2
    assert result["calls"][0]["usage"]["total_tokens"] == 20
    assert result["calls"][0]["image_sha256"] == result["calls"][1]["image_sha256"]
    requests = client.chat.completions.create.call_args_list
    assert all(len(request.kwargs["messages"]) == 2 for request in requests)
    assert all("tools" not in request.kwargs for request in requests)
    assert all(
        request.kwargs["response_format"] == {"type": "json_object"}
        for request in requests
    )
    if provider == "gemini":
        assert requests[0].kwargs["reasoning_effort"] == "low"
        assert "extra_body" not in requests[0].kwargs
    else:
        assert requests[0].kwargs["extra_body"]["thinking"]["type"] == "disabled"
        assert "reasoning_effort" not in requests[0].kwargs
        assert all(request.kwargs["temperature"] == 0 for request in requests)


def test_budget_counts_requests_before_response() -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=_completion("{}"))
    model = ComparisonModel(client, "test-model", "gemini")

    async def run() -> None:
        for _ in range(6):
            await model.analyze_visual_features("system", "input", ())
        with pytest.raises(ValueError, match="лимит"):
            await model.analyze_visual_features("system", "input", ())

    asyncio.run(run())
    assert client.chat.completions.create.await_count == 6


def test_remote_error_is_not_serialized_and_not_retried() -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        side_effect=RateLimitError(
            "secret-test-token private response body",
            response=httpx.Response(
                429, request=httpx.Request("POST", "https://example.test")
            ),
            body={"private": "secret-test-token"},
        )
    )
    model = ComparisonModel(client, "test-model", "gemini")
    case = Case("test", "Reference", (ImageInput(b"image", "image/png"),), ())
    result = asyncio.run(evaluate_case(model, case))
    assert result["ok"] is False and result["http_status"] == 429
    assert "secret-test-token" not in json.dumps(result)
    assert client.chat.completions.create.await_count == 1


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_incomplete_or_blocked_response_never_starts_verification(reason: str) -> None:
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=_completion(observation(), reason)
    )
    model = ComparisonModel(client, "test-model", "gemini")
    case = Case("test", "Reference", (ImageInput(b"image", "image/png"),), ())
    result = asyncio.run(evaluate_case(model, case))
    assert result["ok"] is False
    assert client.chat.completions.create.await_count == 1
    assert result["calls"][0]["finish_reason"] == reason
