"""Сравнение классификаторов отличает отсутствие замера от нулевой ошибки."""

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.evaluation import EvaluationService
from protogen_delta.services.interaction_classification import InteractionClassifier

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "interaction_evaluator", _ROOT / "scripts/evaluate_interaction.py"
)
assert _spec is not None and _spec.loader is not None
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def _service(*, measured: bool = False, limit: int = 80) -> EvaluationService:
    """Создать транспорт с вымышленным ключом и без повторов."""
    return EvaluationService(
        run_model=measured,
        max_calls=limit,
        api_key="unused-test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )


def test_preview_validates_both_variants_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Нейтральные заглушки не превращаются в оценки размеченной выборки."""
    network = AsyncMock(side_effect=AssertionError("Сеть запрещена"))
    monkeypatch.setattr(DeepSeekService, "_complete", network)
    cases = evaluation.load_cases()
    assert len(cases) == 40

    async def scenario() -> None:
        service = _service()
        try:
            report = await evaluation.compare(
                cases, service, output=tmp_path / "preview.json"
            )
        finally:
            await service.close()
        assert report["complete"] and report["calls"] == 80
        assert report["summary"] == {
            "measured": False,
            "regressions": None,
            "improvements": None,
        }
        assert all(
            row["labels"] is None
            and row["label_errors"] is None
            and row["wire_errors"] is None
            for row in report["rows"]
        )
        assert "preview_system" not in (tmp_path / "preview.json").read_text(
            encoding="utf-8"
        )

    asyncio.run(scenario())
    network.assert_not_awaited()


def test_pair_comparison_records_regression_and_format_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ошибку смысловой метки и сбой JSON видно раздельно в одинаковых парах."""
    cases = evaluation.load_cases()[:2]

    async def complete(**kwargs: Any) -> ChatCompletion:
        prompt = kwargs["messages"][0]["content"]
        message = kwargs["messages"][1]["content"]
        candidate = "Общий формат ответа" in prompt
        response = {
            "mood": "neutral",
            "insult": "none",
            "favorite": "none",
            "reasoning": False,
        }
        if message == cases[0]["message"] and candidate:
            response.update(mood="angry", insult="direct")
        if message == cases[1]["message"] and not candidate:
            response["reasoning"] = "false"
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
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(response),
                        },
                    }
                ],
            }
        )

    network = AsyncMock(side_effect=complete)
    monkeypatch.setattr(DeepSeekService, "_complete", network)

    async def scenario() -> None:
        service = _service(measured=True)
        try:
            report = await evaluation.compare(
                cases, service, output=tmp_path / "model.json"
            )
        finally:
            await service.close()
        summary = report["summary"]
        assert summary["regressions"] == [cases[0]["id"]]
        assert summary["improvements"] == [cases[1]["id"]]
        assert summary["variants"]["legacy"]["wire_failures"] == 1
        assert summary["variants"]["single"]["label_failures"] == 1
        assert len(summary["changes"]) == 1
        assert report["calls"] == 4
        assert [row["contract"] for row in report["rows"]] == [
            "legacy",
            "single",
            "single",
            "legacy",
        ]

    asyncio.run(scenario())


def test_budget_exhaustion_never_completes_the_comparison(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = _service(limit=1)
        try:
            report = await evaluation.compare(
                evaluation.load_cases(), service, output=tmp_path / "partial.json"
            )
        finally:
            await service.close()
        assert not report["complete"] and report["budget_exceeded"]
        assert report["calls"] == 1
        assert (
            json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
            == report
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("not json", ["json"]),
        ("[]", ["fields"]),
        ("{}", ["fields"]),
        (
            '{"mood":"neutral","insult":"none","favorite":"none","reasoning":1}',
            ["reasoning"],
        ),
        (
            '{"mood":"neutral","insult":"none","favorite":"none","reasoning":"false"}',
            ["reasoning"],
        ),
        ('{"mood":[],"insult":"none","favorite":"none","reasoning":false}', ["mood"]),
        ('{"mood":"neutral","insult":"none","favorite":"none","reasoning":false}', []),
    ],
)
def test_wire_contract_checks_types_before_values(
    raw: str, expected: list[str]
) -> None:
    assert evaluation.wire_errors(raw, favorites=True) == expected


@pytest.mark.parametrize(
    "row",
    json.loads(
        (_ROOT / "tests/fixtures/interaction_prompt_baseline.json").read_text(
            encoding="utf-8"
        )
    )["cases"],
)
def test_legacy_classifier_prompt_matches_previous_release(
    row: dict[str, Any],
) -> None:
    classifier = InteractionClassifier(
        AsyncMock(spec=DeepSeekService),
        load_prompt("mood_classification"),
        load_prompt("insult_classification"),
        favorite_prompt=load_prompt("favorite_selection") if row["favorites"] else None,
        prompt_contract="legacy",
    )
    assert len(classifier._prompt) == row["chars"]
    assert hashlib.sha256(classifier._prompt.encode()).hexdigest() == row["sha256"]


def test_default_classifier_uses_the_evaluated_single_contract() -> None:
    """Рабочий бот и стандартный стенд получают одинаковую версию по умолчанию."""
    model = AsyncMock(spec=DeepSeekService)
    default = InteractionClassifier(model, "Настроение", "Оскорбление")
    selected = InteractionClassifier(
        model, "Настроение", "Оскорбление", prompt_contract="single"
    )
    assert default._prompt == selected._prompt


@pytest.mark.parametrize("favorites", [False, True])
def test_candidate_has_one_unambiguous_output_contract(favorites: bool) -> None:
    classifier = InteractionClassifier(
        AsyncMock(spec=DeepSeekService),
        load_prompt("mood_classification"),
        load_prompt("insult_classification"),
        favorite_prompt=load_prompt("favorite_selection") if favorites else None,
        prompt_contract="single",
    )
    prompt = classifier._prompt
    assert "Поправка к формату" not in prompt and "JSON с двумя полями" not in prompt
    assert prompt.count("Общий формат ответа") == 1
    sample = json.loads(prompt.split(". Пример: ")[1].split(". Другие поля")[0])
    assert evaluation.wire_errors(json.dumps(sample), favorites=favorites) == []
    for name in ("mood_classification", "insult_classification"):
        assert load_prompt(name) in prompt


def test_unknown_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="Неизвестный контракт"):
        InteractionClassifier(
            AsyncMock(spec=DeepSeekService),
            "Настроение",
            "Оскорбление",
            prompt_contract=cast(Any, "unknown"),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unknown",
        "missing_expected",
        "bad_reasoning",
        "bad_label",
        "bad_favorites",
        "bad_context",
        "empty_message",
    ],
)
def test_invalid_annotations_are_rejected(tmp_path: Path, mutation: str) -> None:
    cases = evaluation.load_cases()[:1]
    case = cases[0]
    if mutation == "duplicate":
        cases.append(case)
    elif mutation == "unknown":
        case["unknown"] = True
    elif mutation == "missing_expected":
        del case["expected"]["insult"]
    elif mutation == "bad_reasoning":
        case["expected"]["reasoning"] = [1]
    elif mutation == "bad_label":
        case["expected"]["mood"] = ["unknown"]
    elif mutation == "bad_favorites":
        case["favorites"] = "false"
    elif mutation == "bad_context":
        case["reasoning_context"] = []
    else:
        case["message"] = ""
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_cases(path)


@pytest.mark.parametrize(
    "args", [["--run-model"], ["--case", "unknown"], ["--max-calls", "0"]]
)
def test_cli_validates_before_creating_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    factory = Mock()
    monkeypatch.setattr(evaluation, "EvaluationService", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_interaction", "--output", str(tmp_path / "result.json"), *args],
    )
    with pytest.raises(SystemExit) as error:
        asyncio.run(evaluation.main())
    assert error.value.code == 2
    factory.assert_not_called()
