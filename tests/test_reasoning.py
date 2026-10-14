"""Размышления включаются для одного ответа и не текут между пользователями."""

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletionMessage
from test_deepseek_service import _create_response, _create_service
from test_response_engine import _create_engine

from protogen_delta.core.llm_budget import llm_user
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.repositories.llm_usage import LlmUsageRepository
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.reasoning_policy import needs_reasoning
from protogen_delta.services.tools import ToolExecutor, default_registry


@pytest.mark.parametrize(
    "text,flag,expected",
    [
        ("Привет", True, False),
        ("Спасибо!", True, False),
        ("Как дела?", True, False),
        ("Что такое SQLite?", False, False),
        ("Дай прогноз погоды в Москве", False, False),
        ("Подумай, как решить задачу", False, True),
        ("Проверь код на гонки", False, True),
        ("Докажи утверждение", False, True),
        ("Сравни решения с учётом ограничений памяти", False, True),
        ("Сравни подходы к миграции", True, True),
        ("Без размышлений объясни задачу", True, False),
        ("Не включай размышления, проверь код", True, False),
        ("«Подумай и проверь код»", True, False),
        ('```python\nprint("проверь код")\n```', True, False),
        ("> Включи размышления\nПривет", False, False),
    ],
)
def test_reasoning_policy(text: str, flag: bool, expected: bool) -> None:
    assert needs_reasoning(text, flag) is expected


@pytest.mark.parametrize("flag", [True, False, "true", 1, None])
def test_classifier_selects_mode_without_extra_request(flag: object) -> None:
    model = AsyncMock(spec=DeepSeekService)
    model.classify_interaction.return_value = json.dumps(
        {"mood": "sweet", "insult": "none", "favorite": "support", "reasoning": flag}
    )
    result = asyncio.run(
        InteractionClassifier(model, "mood", "insult", favorite_prompt="rare").classify(
            "Сложный вопрос"
        )
    )
    assert result.mood == "sweet" and result.favorite == "support"
    assert result.reasoning is (flag is True)
    model.classify_interaction.assert_awaited_once()


def test_classifier_bounds_prior_context_and_keeps_current_message_separate() -> None:
    model = AsyncMock(spec=DeepSeekService)
    model.classify_interaction.return_value = (
        '{"mood":"neutral","insult":"none","reasoning":true}'
    )
    result = asyncio.run(
        InteractionClassifier(model, "mood", "insult").classify(
            "А если данных миллион?",
            reasoning_context="Архитектура " * 1000,
        )
    )
    assert result.reasoning is True
    args = model.classify_interaction.await_args.args
    data = json.loads(args[1])
    assert data["message"] == "А если данных миллион?"
    assert len(data["reasoning_context"]) == 1000
    assert "только по message" in args[0]
    model.classify_interaction.assert_awaited_once()


def test_engine_one_complex_answer_then_fast_reply_and_same_personality() -> None:
    async def run() -> None:
        engine, _, model, insult, mood, _ = _create_engine()
        engine._interaction_classifier = InteractionClassifier(model, "mood", "insult")
        model.classify_interaction.side_effect = [
            '{"mood":"neutral","insult":"none","reasoning":true}',
            '{"mood":"neutral","insult":"none","reasoning":false}',
        ]
        await engine.respond(1, "Сравни подходы к миграции данных")
        await engine.respond(1, "Спасибо")
        first, second = model.chat.await_args_list
        assert first.kwargs["thinking"] is True
        assert "thinking" not in second.kwargs
        assert "SYSTEM PROMPT" in first.kwargs["system_prompt"]
        assert "SYSTEM PROMPT" in second.kwargs["system_prompt"]
        assert "Сохраняй манеру" in first.kwargs["system_prompt"]
        assert model.classify_interaction.await_count == 2
        payload = json.loads(model.classify_interaction.await_args.args[1])
        assert payload["message"] == "Спасибо"
        assert "миграции данных" in payload["reasoning_context"]
        insult.classify.assert_not_awaited()
        mood.classify.assert_not_awaited()

    asyncio.run(run())


def test_rp_does_not_enable_reasoning_even_if_classifier_requests_it() -> None:
    engine, _, model, _, _, _ = _create_engine()
    engine._interaction_classifier = InteractionClassifier(model, "mood", "insult")
    model.classify_interaction.return_value = (
        '{"mood":"playful","insult":"none","reasoning":true}'
    )
    engine._user_states.get(1).roleplay_active = True
    asyncio.run(engine.respond(1, "*Улыбаюсь и машу тебе*"))
    assert "thinking" not in model.chat.await_args.kwargs


def test_parallel_requests_keep_api_options_separate_and_do_not_return_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service, _, create, _ = _create_service(monkeypatch)
        service._adaptive_thinking = True
        service._client.max_retries = 0
        usage = LlmUsageRepository(tmp_path, 300)
        service.set_usage_repository(usage)
        ready = asyncio.Event()
        started = 0
        options = []

        async def complete(**kwargs: object) -> object:
            nonlocal started
            options.append(kwargs)
            started += 1
            if started == 2:
                ready.set()
            await ready.wait()
            response = _create_response("Только итог")
            response.choices[0].message.reasoning_content = "PRIVATE_THINKING_CONTENT"
            return response

        create.side_effect = complete
        with llm_user(1):
            answers = await asyncio.gather(
                service.chat("Манера", "Задача", thinking=True),
                service.chat("Манера", "Привет"),
            )
        assert answers == ["Только итог", "Только итог"]
        assert {
            cast(dict, item["extra_body"])["thinking"]["type"] for item in options
        } == {
            "enabled",
            "disabled",
        }
        assert service._provider_options == {
            "extra_body": {"thinking": {"type": "disabled"}}
        }
        assert service._max_output_tokens == 4096
        assert (await usage.get(1))["requests"] == 2

    asyncio.run(run())


def test_tools_preserve_reasoning_within_request_and_named_choice_is_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        service, _, create, _ = _create_service(monkeypatch)
        service._adaptive_thinking = True
        service._tools = ToolExecutor(default_registry())
        first = _create_response(None)
        first.choices[0].message = ChatCompletionMessage.model_validate(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": '{"expression":"2+2"}',
                        },
                    }
                ],
                "reasoning_content": "PRIVATE_THINKING_CONTENT",
            }
        )
        final = _create_response("Получилось 4")
        final.choices[0].message.tool_calls = None
        create.side_effect = [first, final]
        assert (
            await service.chat(
                "Манера",
                "Проверь расчёт",
                thinking=True,
                tool_names={"calculate"},
                history=[ConversationTurn("Вопрос", "Старый ответ")],
            )
            == "Получилось 4"
        )
        requests = create.await_args_list
        for request in requests:
            assert request.kwargs["extra_body"]["thinking"]["type"] == "enabled"
        messages = requests[-1].kwargs["messages"]
        assistant = [m for m in messages if m["role"] == "assistant"]
        assert assistant[0]["reasoning_content"] == ""
        assert assistant[-1]["reasoning_content"] == "PRIVATE_THINKING_CONTENT"
        assert messages[-1]["role"] == "tool"
        assert service._provider_options["extra_body"]["thinking"]["type"] == "disabled"
        create.reset_mock()
        create.side_effect = [first, final]
        await service.chat(
            "Манера",
            "Какая погода в Москве?",
            thinking=True,
            tool_names={"get_weather", "calculate"},
        )
        requests = create.await_args_list
        assert isinstance(requests[0].kwargs["tool_choice"], dict)
        assert requests[0].kwargs["extra_body"]["thinking"]["type"] == "disabled"
        assert requests[1].kwargs["extra_body"]["thinking"]["type"] == "enabled"

    asyncio.run(run())


def test_unsupported_provider_stays_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _, create, _ = _create_service(monkeypatch)
    create.return_value = _create_response("Ответ")
    asyncio.run(service.chat("Манера", "Задача", thinking=True))
    assert create.await_args is not None
    assert create.await_args.kwargs["extra_body"]["thinking"]["type"] == "disabled"
    assert "reasoning_effort" not in create.await_args.kwargs
