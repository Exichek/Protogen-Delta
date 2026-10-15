"""Предпросмотр проверяет сборку, но не выдаёт заглушки за оценку речи."""

import asyncio
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.config.settings import Settings
from protogen_delta.core.sender import current_sender
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.response_configuration import load_response_configuration
from protogen_delta.services.response_engine import ResponseEngine

_spec = importlib.util.spec_from_file_location(
    "behavior_evaluation_script",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_behavior.py",
)
assert _spec is not None and _spec.loader is not None
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def _settings() -> Settings:
    """Ни одной реальной учётной записи или ключа в стенде."""
    return Settings(telegram_token="", deepseek_api_key="", art_chat_id=0)


def _service(*, run_model: bool = False, max_calls: int = 200) -> Any:
    """Создать перехватываемый транспорт без Telegram и рабочих хранилищ."""
    return evaluation.EvaluationService(
        run_model=run_model,
        max_calls=max_calls,
        api_key="unused-test-key",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
    )


def test_preview_runs_every_case_without_network_or_quality_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = AsyncMock(side_effect=AssertionError("Сеть запрещена в предпросмотре"))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)
    cases = evaluation.load_cases()
    assert len(cases) == 36
    assert len({case["category"] for case in cases}) == 7

    async def scenario() -> None:
        service = _service()
        try:
            report = await evaluation.evaluate(
                cases, service, _settings(), output=tmp_path / "preview.json"
            )
        finally:
            await service.close()
        assert report["mode"] == "preview"
        assert len(report["results"]) == 44
        assert all(row["checks"] == "passed" for row in report["results"])
        assert all(
            row["quality_review"] == "not_run"
            and row["reply"] is None
            and row["style"] is None
            for row in report["results"]
        )
        assert report["calls"] == 82
        assert not report["budget_exceeded"]
        assert current_sender.get() is None
        raw = (tmp_path / "preview.json").read_text(encoding="utf-8")
        assert "preview_system" not in raw and "unused-test-key" not in raw
        assert json.loads(raw) == report

    asyncio.run(scenario())
    transport.assert_not_awaited()


@pytest.mark.parametrize("run_model", [False, True])
def test_budget_and_snapshot_survive_reasoning_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_model: bool,
) -> None:
    response = ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Ответ",
                    },
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
        }
    )
    transport = AsyncMock(return_value=response)
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def scenario() -> None:
        service = _service(run_model=run_model, max_calls=1)
        try:
            copy = service.for_reasoning()
            await copy.chat("Правила", "Привет")
            assert len(service.trace["requests"]) == 1
            with pytest.raises(RuntimeError, match="Лимит запросов"):
                await service.classify_interaction("Правила", "Привет")
            assert service.trace["budget_exceeded"]
            assert len(service.trace["requests"]) == 1
        finally:
            await service.close()

    asyncio.run(scenario())
    assert transport.await_count == int(run_model)


def test_mocked_model_keeps_human_review_pending_and_records_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def complete(**kwargs: Any) -> ChatCompletion:
        text = (
            '{"mood":"neutral","insult":"none","favorite":"none","reasoning":false}'
            if "response_format" in kwargs
            else "Повторяю неверный ответ"
        )
        return ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    }
                ],
            }
        )

    monkeypatch.setattr(DeepSeekService, "_complete", AsyncMock(side_effect=complete))
    case = {
        "id": "failure",
        "category": "голос",
        "criteria": ["Понятный ответ"],
        "steps": [
            {
                "message": "Привет",
                "required": ["Искомое"],
                "forbidden": ["неверный"],
                "max_chars": 5,
            }
        ],
    }

    async def scenario() -> None:
        service = _service(run_model=True)
        try:
            report = await evaluation.evaluate(
                [case], service, _settings(), output=tmp_path / "model.json"
            )
        finally:
            await service.close()
        row = report["results"][0]
        assert row["quality_review"] == "pending"
        assert row["checks"] == "failed" and len(row["errors"]) == 3
        assert row["style"]["chars"] == len(row["reply"])
        assert all(
            "seconds" in request and "usage" in request for request in row["requests"]
        )

    asyncio.run(scenario())


def test_frozen_state_makes_identical_labels_produce_identical_reply_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Задержка API не должна менять затухание настроения и время событий стенда."""

    async def complete(**kwargs: Any) -> ChatCompletion:
        await asyncio.sleep(0.001)
        content = (
            '{"mood":"neutral","insult":"none","favorite":"none","reasoning":false}'
            if "response_format" in kwargs
            else "Одинаковый синтетический ответ."
        )
        return ChatCompletion.model_validate(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            }
        )

    monkeypatch.setattr(DeepSeekService, "_complete", AsyncMock(side_effect=complete))

    async def scenario() -> None:
        signatures = []
        for contract in ("legacy", "single"):
            service = _service(run_model=True)
            try:
                report = await evaluation.evaluate(
                    evaluation.load_cases(),
                    service,
                    _settings(),
                    output=tmp_path / (contract + ".json"),
                    prompt_contract=contract,
                )
            finally:
                await service.close()
            signatures.append(
                [
                    request["system_sha256"]
                    for row in report["results"]
                    for request in row["requests"]
                    if request["kind"] == "reply"
                ]
            )
            assert report["state_clock"]["kind"] == "synthetic"
        assert signatures[0] and signatures[0] == signatures[1]

    asyncio.run(scenario())


def test_preview_failure_and_exhausted_budget_are_not_success(
    tmp_path: Path,
) -> None:
    case = {
        "id": "failure",
        "category": "сцена",
        "criteria": ["Нет сцены"],
        "steps": [
            {
                "message": "Привет",
                "prompt_required": ["ОТСУТСТВУЕТ_591"],
                "expected_rp": True,
                "expected_restricted": True,
                "expected_model_calls": 0,
            }
        ],
    }

    async def scenario() -> None:
        service = _service(max_calls=1)
        try:
            report = await evaluation.evaluate(
                [case], service, _settings(), output=tmp_path / "budget.json"
            )
        finally:
            await service.close()
        assert report["budget_exceeded"]
        assert report["results"][0]["checks"] == "failed"
        assert len(report["results"][0]["errors"]) == 4

    asyncio.run(scenario())


def test_api_failure_cannot_pass_as_a_regular_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Служебный ответ движка после сбоя не считается удачным ответом модели."""
    transport = AsyncMock(side_effect=DeepSeekError("synthetic failure"))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def scenario() -> None:
        service = _service(run_model=True)
        try:
            report = await evaluation.evaluate(
                evaluation.load_cases()[:1],
                service,
                _settings(),
                output=tmp_path / "error.json",
            )
        finally:
            await service.close()
        assert report["results"][0]["checks"] == "failed"
        assert "Ошибка запроса API" in report["results"][0]["errors"]
        assert report["results"][0]["quality_review"] == "pending"

    asyncio.run(scenario())


def test_default_cli_does_not_read_env_and_selects_only_requested_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Случайно указанный env-файл не включает платный режим."""
    env = Mock(side_effect=AssertionError("Не нужно читать ключи в предпросмотре"))
    transport = AsyncMock(side_effect=AssertionError("Сеть запрещена"))
    monkeypatch.setattr(evaluation, "dotenv_values", env)
    monkeypatch.setattr(DeepSeekService, "_complete", transport)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_behavior",
            "--output",
            str(tmp_path / "one.json"),
            "--env-file",
            "unused",
            "--case",
            "greeting",
        ],
    )
    asyncio.run(evaluation.main())
    assert json.loads(capsys.readouterr().out)["turns"] == 1
    env.assert_not_called()
    transport.assert_not_awaited()


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unknown",
        "empty_steps",
        "no_criteria",
        "bad_pattern",
        "bad_mode",
        "bad_emotions",
        "bad_initial",
        "bad_limit",
        "bad_flag",
        "bad_calls",
        "bad_attachment",
        "bad_patterns",
        "empty_message",
    ],
)
def test_invalid_suite_is_rejected_before_client_creation(
    tmp_path: Path, mutation: str
) -> None:
    cases = evaluation.load_cases()[:1]
    case, step = cases[0], cases[0]["steps"][0]
    if mutation == "duplicate":
        cases.append(case)
    elif mutation == "unknown":
        case["extra"] = True
    elif mutation == "empty_steps":
        case["steps"] = []
    elif mutation == "no_criteria":
        case["criteria"] = []
    elif mutation == "bad_pattern":
        step["required"] = ["["]
    elif mutation == "bad_mode":
        case["initial"] = {"content_mode": []}
    elif mutation == "bad_emotions":
        case["initial"] = {"emotions": {"joy": 1.1}}
    elif mutation == "bad_initial":
        case["initial"] = {"unknown": True}
    elif mutation == "bad_limit":
        step["max_chars"] = True
    elif mutation == "bad_flag":
        step["expected_rp"] = "false"
    elif mutation == "bad_calls":
        step["expected_model_calls"] = -1
    elif mutation == "bad_attachment":
        step["attachment_text"] = []
    elif mutation == "bad_patterns":
        step["required"] = "answer"
    else:
        step["message"] = ""
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises((ValueError, re.error)):
        evaluation.load_cases(path)


@pytest.mark.parametrize(
    "args",
    [
        ["--run-model"],
        ["--max-calls", "0"],
        ["--case", "missing"],
    ],
)
def test_cli_rejects_invalid_request_before_creating_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    factory = Mock()
    monkeypatch.setattr(evaluation, "EvaluationService", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_behavior", "--output", str(tmp_path / "report.json"), *args],
    )
    with pytest.raises(SystemExit) as error:
        asyncio.run(evaluation.main())
    assert error.value.code == 2
    factory.assert_not_called()


def test_shared_configuration_uses_current_timers_and_prompt_modules() -> None:
    config = load_response_configuration(
        Settings(
            telegram_token="",
            deepseek_api_key="",
            art_chat_id=0,
            proactive_idle_seconds=7200,
            proactive_idle_max_seconds=14400,
            proactive_cooldown_seconds=21600,
        ),
        fetish_triggers={},
        fetish_names={},
    )
    assert "между 2 и 4 часами" in config.system_prompt
    assert "через 6 часов" in config.system_prompt
    assert load_prompt("emotional_system") in config.system_prompt
    assert config.technical_prompt and config.visual_prompt and config.voice_prompt
    inventory = evaluation.prompt_inventory()
    assert inventory == evaluation.prompt_inventory()
    assert len(inventory["revision"]) == 64
    assert "personality/core.txt" in inventory["files"]


_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures" / "behavior_prompt_baseline.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda case: case["id"])
def test_runtime_prompt_matches_previous_deployed_revision(
    case: dict[str, Any],
) -> None:
    """Сверять с прежней сборкой, а не с ожидаемым результатом нового кода."""
    model = AsyncMock(spec=DeepSeekService)
    config = replace(
        load_response_configuration(_settings(), fetish_triggers={}, fetish_names={}),
        capabilities_context="Подключённые функции тестового стенда.",
    )
    engine = ResponseEngine(
        model,
        InsultClassifier(model, "Оскорбления"),
        MoodClassifier(model, "Настроение"),
        FetishRoleClassifier(model, "Роль"),
        BotState(),
        UserStateStore(),
        config,
    )
    prompt = engine._build_prompt(**case["arguments"])
    assert len(prompt) == case["chars"]
    assert hashlib.sha256(prompt.encode()).hexdigest() == case["sha256"]
