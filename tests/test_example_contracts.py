"""Изолировать учебные примеры, сохранив прежнюю сборку по умолчанию."""

import asyncio
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from protogen_delta.config.settings import Settings
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.services.deepseek import DeepSeekConnectionError, DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.prompt_composer import (
    ExampleContract,
    PromptComposer,
    PromptSections,
)
from protogen_delta.services.response_configuration import load_response_configuration
from protogen_delta.services.response_engine import ResponseEngine

_SPEC = importlib.util.spec_from_file_location(
    "compare_example_contracts_test",
    Path(__file__).parents[1] / "scripts/compare_example_contracts.py",
)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)
_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/behavior_prompt_current.json").read_text("utf-8")
)


def settings() -> Settings:
    """Синтетическая конфигурация без доступа к рабочему окружению."""
    return Settings(telegram_token="", deepseek_api_key="", art_chat_id=0)


@pytest.mark.parametrize("roleplay", [False, True])
def test_isolated_examples_preserve_exact_texts_without_becoming_dialogue(
    roleplay: bool,
) -> None:
    sections = PromptSections(
        core="CORE", voice='Голос "\n— Пример → Ответ', scene_voice="Сцена <данные>"
    )
    composer = PromptComposer(sections, example_contract="isolated")
    block = composer.examples(is_roleplay=roleplay)
    data = json.loads(
        block.split("<style_examples>\n", 1)[1].split("\n</style_examples>", 1)[0]
    )
    assert data["образцы_голоса"] == sections.voice
    assert data.get("образцы_сцен") == (sections.scene_voice if roleplay else None)
    assert "не общую историю" in block and "Новые идеи" in block
    assert (
        PromptComposer(
            PromptSections(core="CORE"), example_contract="isolated"
        ).examples(is_roleplay=roleplay)
        == ""
    )
    with pytest.raises(ValueError, match="контракт"):
        PromptComposer(sections, example_contract=cast(ExampleContract, "typo"))


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda case: case["id"])
def test_only_example_block_and_position_change_in_candidate(
    case: dict[str, Any],
) -> None:
    """Проверить утверждённый SHA и неизменность остальных блоков кандидата."""
    model = AsyncMock(spec=DeepSeekService)
    config = replace(
        load_response_configuration(settings(), fetish_triggers={}, fetish_names={}),
        capabilities_context="Подключённые функции тестового стенда.",
    )

    def engine(contract: ExampleContract) -> ResponseEngine:
        return ResponseEngine(
            model,
            InsultClassifier(model, "Правила"),
            MoodClassifier(model, "Правила"),
            FetishRoleClassifier(model, "Правила"),
            BotState(),
            UserStateStore(),
            config,
            example_contract=contract,
        )

    old, new = engine("legacy"), engine("isolated")
    legacy = old._build_prompt(**case["arguments"])
    isolated = new._build_prompt(**case["arguments"])
    assert hashlib.sha256(legacy.encode()).hexdigest() == case["sha256"]
    roleplay = case["arguments"]["is_rp"]
    before = old._prompt_composer.examples(is_roleplay=roleplay)
    after = new._prompt_composer.examples(is_roleplay=roleplay)
    assert legacy.replace("\n\n" + before, "", 1) == isolated.replace(
        "\n\n" + after, "", 1
    )
    if "Контекст текущего сообщения" in isolated:
        assert isolated.index("</style_examples>") < isolated.index(
            "Контекст текущего сообщения"
        )
    default = PromptComposer(PromptSections(core="CORE", voice="VOICE"))
    assert (
        default.example_contract == "legacy"
        and default.examples(is_roleplay=False) == "VOICE"
    )


def test_preview_pairs_have_fixed_labels_clock_and_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = AsyncMock(side_effect=AssertionError("Сеть запрещена"))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=False,
            max_calls=48,
            api_key="NEVER_PRINT_KEY",
            model="test",
            base_url="https://test.local",
        )
        try:
            report = await runner.compare(
                runner.behavior.load_cases(runner.CASES_PATH),
                service,
                settings(),
                output=tmp_path / "pairs.json",
            )
        finally:
            await service.close()
        assert report["complete"] and report["calls"] == 48
        assert report["fixed_classifications"] >= 48
        assert all(item["kind"] == "reply" for item in service.trace["requests"])
        pairs: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in report["rows"]:
            assert row["result"]["checks"] == "passed"
            assert row["result"]["quality_review"] == "not_run"
            pairs.setdefault((row["case"], row["repeat"]), []).append(row)
        assert len(pairs) == 24
        for pair in pairs.values():
            assert {row["contract"] for row in pair} == {"legacy", "isolated"}
            assert pair[0]["state_clock"] == pair[1]["state_clock"]
            assert pair[0]["cases_sha256"] == pair[1]["cases_sha256"]
            assert pair[0]["prompts"] == pair[1]["prompts"]
            assert (
                pair[0]["result"]["requests"][0]["system_sha256"]
                != pair[1]["result"]["requests"][0]["system_sha256"]
            )
        assert "NEVER_PRINT_KEY" not in (tmp_path / "pairs.json").read_text("utf-8")

    asyncio.run(run())
    transport.assert_not_awaited()


def test_insufficient_budget_rejected_before_requests(tmp_path: Path) -> None:
    async def run() -> None:
        service = runner.FixedLabelsService(
            run_model=False,
            max_calls=1,
            api_key="test",
            model="test",
            base_url="https://test.local",
        )
        try:
            with pytest.raises(ValueError, match="бюджет"):
                await runner.compare(
                    runner.behavior.load_cases(runner.CASES_PATH)[:1],
                    service,
                    settings(),
                    output=tmp_path / "pairs.json",
                )
            assert service.trace["requests"] == []
        finally:
            await service.close()

    asyncio.run(run())


def test_api_failure_stops_comparison_and_saves_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = AsyncMock(side_effect=DeepSeekConnectionError("unavailable"))
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
                runner.behavior.load_cases(runner.CASES_PATH),
                service,
                settings(),
                output=tmp_path / "pairs.json",
            )
        finally:
            await service.close()
        assert not report["complete"] and report["calls"] == 1
        assert (
            len(report["rows"]) == 1
            and report["rows"][0]["result"]["checks"] == "failed"
        )
        assert json.loads((tmp_path / "pairs.json").read_text("utf-8")) == report

    asyncio.run(run())
    transport.assert_awaited_once()
