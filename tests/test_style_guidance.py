"""Проверка кандидата без готовых реплик: прежний режим, реальные факты и RP."""

import asyncio
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion

from protogen_delta.config.settings import Settings
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.services.deepseek import DeepSeekConnectionError, DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.prompt_composer import (
    ExampleContract,
    ExampleGuidance,
    PromptComposer,
    PromptSections,
)
from protogen_delta.services.response_configuration import (
    load_example_guidance,
    load_response_configuration,
)
from protogen_delta.services.response_engine import ResponseEngine

_SPEC = importlib.util.spec_from_file_location(
    "guidance_comparison_test",
    Path(__file__).parents[1] / "scripts/compare_example_contracts.py",
)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)
_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/behavior_prompt_baseline.json").read_text(
        "utf-8"
    )
)


def settings() -> Settings:
    """Не читать рабочее окружение."""
    return Settings(telegram_token="", deepseek_api_key="", art_chat_id=0)


def engine(contract: ExampleContract) -> ResponseEngine:
    """Одинаковые основные правила и отдельный явно переданный кандидат."""
    model = AsyncMock(spec=DeepSeekService)
    config = replace(
        load_response_configuration(settings(), fetish_triggers={}, fetish_names={}),
        capabilities_context="Подключённые функции тестового стенда.",
    )
    return ResponseEngine(
        model,
        InsultClassifier(model, "Правила"),
        MoodClassifier(model, "Правила"),
        FetishRoleClassifier(model, "Правила"),
        BotState(),
        UserStateStore(),
        config,
        example_contract=contract,
        example_guidance=load_example_guidance() if contract == "guidance" else None,
    )


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda case: case["id"])
def test_guidance_changes_only_the_example_block_at_its_original_position(
    case: dict[str, Any],
) -> None:
    old, new = engine("legacy"), engine("guidance")
    before = old._assemble_prompt(**case["arguments"])
    after = new._assemble_prompt(**case["arguments"])
    assert hashlib.sha256(before.render().encode()).hexdigest() == case["sha256"]
    assert [b.name for b in before.blocks] == [b.name for b in after.blocks]
    assert [b for b in before.blocks if b.name != "examples"] == [
        b for b in after.blocks if b.name != "examples"
    ]
    candidate = next(b.text for b in after.blocks if b.name == "examples")
    assert "Ориентиры голоса" in candidate and "→" not in candidate
    assert not any(
        word in candidate.casefold()
        for word in ("инвентар", "аптеч", "фонарь", "карту", "прочность")
    )
    assert old._config == new._config


@pytest.mark.parametrize(
    "guidance", [None, ExampleGuidance("", "Сцена"), ExampleGuidance("Голос", " ")]
)
def test_candidate_requires_both_explicit_guidance_texts(
    guidance: ExampleGuidance | None,
) -> None:
    with pytest.raises(ValueError, match="нужны оба"):
        PromptComposer(
            PromptSections(core="CORE"), example_contract="guidance", guidance=guidance
        )


def test_ordinary_configuration_does_not_load_candidate_files() -> None:
    names = []

    def loader(name: str) -> str:
        names.append(name)
        return "Правила"

    load_response_configuration(
        settings(), fetish_triggers={}, fetish_names={}, loader=loader
    )
    assert "voice_guidance" not in names and "scene_guidance" not in names
    names.clear()
    load_example_guidance(loader=loader)
    assert names == ["voice_guidance", "scene_guidance"]


def test_real_context_is_not_removed_with_examples() -> None:
    args = dict(_BASELINE["cases"][0]["arguments"])
    args.update(
        state_context=["У игры шесть ячеек инвентаря."],
        scene_context=["Пользователь держит карту."],
        is_rp=True,
    )
    prompt = engine("guidance")._build_prompt(**args)
    assert "шесть ячеек инвентаря" in prompt and "Пользователь держит карту." in prompt
    assert "Сцена Дельты" in prompt and "Ориентиры голоса в сцене" in prompt


def test_multiturn_preview_is_paired_reproducible_and_never_calls_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = AsyncMock(side_effect=AssertionError("Сеть запрещена"))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=False,
            max_calls=48,
            api_key="NEVER_PRINT",
            model="test",
            base_url="https://test.local",
        )
        cases = runner.behavior.load_cases(
            runner.CASES_PATH.with_name("style_guidance_cases.json")
        )
        assert len(cases) == 8 and sum(len(c["steps"]) for c in cases) == 12
        try:
            report = await runner.compare(
                cases,
                service,
                settings(),
                output=tmp_path / "pairs.json",
                candidate="guidance",
            )
        finally:
            await service.close()
        assert (
            report["complete"]
            and report["calls"] == 48
            and report["contracts"] == ["legacy", "guidance"]
        )
        repeated: dict[tuple[str, str, int], set[str]] = {}
        for row in report["rows"]:
            result = row["result"]
            request = result["requests"][0]
            assert (
                result["checks"] == "passed" and result["quality_review"] == "not_run"
            )
            assert request["history_messages"] == (0 if result["step"] == 1 else 2)
            repeated.setdefault(
                (row["case"], row["contract"], result["step"]), set()
            ).add(request["system_sha256"])
        assert len(repeated) == 24 and all(
            len(values) == 1 for values in repeated.values()
        )
        assert "NEVER_PRINT" not in (tmp_path / "pairs.json").read_text("utf-8")

    asyncio.run(run())
    transport.assert_not_awaited()


def response(text: str) -> ChatCompletion:
    """Синтетический ответ транспорта, без реального сервиса."""
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


def test_second_turn_failure_stops_before_any_later_variant_or_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = AsyncMock(
        side_effect=[
            response("Не знаю деталей, можем придумать их сейчас."),
            DeepSeekConnectionError("unavailable"),
        ]
    )
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=True,
            max_calls=48,
            api_key="test",
            model="test",
            base_url="https://test.local",
        )
        try:
            report = await runner.compare(
                runner.behavior.load_cases(
                    runner.CASES_PATH.with_name("style_guidance_cases.json")
                ),
                service,
                settings(),
                output=tmp_path / "partial.json",
                candidate="guidance",
            )
        finally:
            await service.close()
        assert (
            not report["complete"] and report["calls"] == 2 and len(report["rows"]) == 2
        )
        assert [row["result"]["step"] for row in report["rows"]] == [1, 2]
        assert report["rows"][-1]["result"]["checks"] == "failed"
        assert json.loads((tmp_path / "partial.json").read_text("utf-8")) == report

    asyncio.run(run())
    assert transport.await_count == 2


def test_remaining_shared_budget_is_checked_before_any_new_request(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=False,
            max_calls=4,
            api_key="test",
            model="test",
            base_url="https://test.local",
        )
        service.trace["requests"].append({"kind": "reply"})
        try:
            with pytest.raises(ValueError, match="бюджет"):
                await runner.compare(
                    runner.behavior.load_cases(
                        runner.CASES_PATH.with_name("style_guidance_cases.json")
                    )[:1],
                    service,
                    settings(),
                    output=tmp_path / "none.json",
                    candidate="guidance",
                    repeats=1,
                )
            assert (
                len(service.trace["requests"]) == 1
                and not (tmp_path / "none.json").exists()
            )
        finally:
            await service.close()

    asyncio.run(run())


def test_explicitly_requested_quote_is_not_filtered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quote = "Ладно, этот заход был так себе. Считай, проверил шутку на прочность :D"
    transport = AsyncMock(return_value=response(quote))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=True,
            max_calls=1,
            api_key="test",
            model="test",
            base_url="https://test.local",
        )
        case = runner.behavior.load_cases(
            runner.CASES_PATH.with_name("style_guidance_cases.json")
        )[-1]
        try:
            report = await runner.behavior.evaluate(
                [case],
                service,
                settings(),
                output=tmp_path / "quote.json",
                example_contract="guidance",
            )
        finally:
            await service.close()
        assert (
            report["results"][0]["reply"] == quote
            and report["results"][0]["checks"] == "passed"
        )
        assert report["results"][0]["quality_review"] == "pending"

    asyncio.run(run())
    assert transport.await_count == 1


def test_second_turn_receives_actual_synthetic_messages_not_an_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Исправить стенд: запрет постоянных фактов не должен отключать историю в RAM."""
    seen = []

    async def complete(**kwargs: Any) -> ChatCompletion:
        seen.append(kwargs["messages"])
        return response("Ты назвал ровно шесть ячеек инвентаря.")

    transport = AsyncMock(side_effect=complete)
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=True,
            max_calls=2,
            api_key="test",
            model="test",
            base_url="https://test.local",
        )
        case = runner.behavior.load_cases(
            runner.CASES_PATH.with_name("style_guidance_cases.json")
        )[1]
        try:
            result = await runner.behavior.evaluate(
                [case],
                service,
                settings(),
                output=tmp_path / "history.json",
                example_contract="guidance",
            )
        finally:
            await service.close()
        assert result["history_scope"] == "synthetic_scenario_ram"
        assert len(seen) == 2 and len(seen[1]) == 4
        assert (
            seen[1][1]["role"] == "user"
            and case["steps"][0]["message"] in seen[1][1]["content"]
        )
        assert seen[1][2] == {
            "role": "assistant",
            "content": "Ты назвал ровно шесть ячеек инвентаря.",
        }
        assert (
            seen[1][-1]["role"] == "user"
            and case["steps"][1]["message"] in seen[1][-1]["content"]
        )
        assert all(row["checks"] == "passed" for row in result["results"])

    asyncio.run(run())
    assert transport.await_count == 2


def test_post_transport_failure_is_not_a_model_response_or_a_later_api_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ошибка после успешного HTTP тоже не должна превращаться в замер голоса."""
    from protogen_delta.services.deepseek import DeepSeekAPIError

    original = runner.FixedLabelsService.chat

    async def fail_after_transport(self: Any, *args: Any, **kwargs: Any) -> str:
        await original(self, *args, **kwargs)
        raise DeepSeekAPIError("synthetic processing failure")

    transport = AsyncMock(return_value=response("Неважный ответ"))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)
    monkeypatch.setattr(runner.FixedLabelsService, "chat", fail_after_transport)

    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=True,
            max_calls=48,
            api_key="test",
            model="test",
            base_url="https://test.local",
        )
        try:
            report = await runner.compare(
                runner.behavior.load_cases(
                    runner.CASES_PATH.with_name("style_guidance_cases.json")
                ),
                service,
                settings(),
                output=tmp_path / "error.json",
                candidate="guidance",
            )
        finally:
            await service.close()
        assert not report["complete"] and report["calls"] == 1
        row = report["rows"][0]["result"]
        assert row["generation_error"] and row["reply"] is None and row["style"] is None
        assert row["quality_review"] == "not_run" and row["checks"] == "failed"
        assert not any("error" in request for request in row["requests"])

    asyncio.run(run())
    assert transport.await_count == 1
