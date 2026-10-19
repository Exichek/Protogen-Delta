"""Проверить карту реальной сборки, сохранив прежние тексты и границы данных."""

import asyncio
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from protogen_delta.config.settings import Settings
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.prompt_assembly import PromptAssembly
from protogen_delta.services.prompt_composer import PromptComposer, PromptSections
from protogen_delta.services.response_configuration import load_response_configuration
from protogen_delta.services.response_engine import ResponseEngine

_BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/behavior_prompt_current.json").read_text("utf-8")
)
_SPEC = importlib.util.spec_from_file_location(
    "assembly_behavior_test", Path(__file__).parents[1] / "scripts/evaluate_behavior.py"
)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def make_engine() -> ResponseEngine:
    """Не подключать сеть, рабочие базы и Telegram."""
    model = AsyncMock(spec=DeepSeekService)
    settings = Settings(telegram_token="", deepseek_api_key="", art_chat_id=0)
    config = replace(
        load_response_configuration(settings, fetish_triggers={}, fetish_names={}),
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
    )


def test_assembly_is_local_immutable_and_preserves_exact_unicode() -> None:
    original = PromptAssembly().append("core", "  Дельта 🦈\n")
    left = original.append("facts", "\nТолько первый пользователь")
    right = original.append("facts", "\nДругой пользователь")
    assert original.render() == "  Дельта 🦈\n"
    assert "первый" not in right.render() and "Другой" not in left.render()
    assert original.append("ignored", "") is original
    assert original.describe() == [{"name": "core", "start": 0, "end": 11, "chars": 11}]
    assert "Дельта" not in repr(original)
    assert "пользователь" not in json.dumps(left.describe(), ensure_ascii=False)
    assert PromptAssembly().describe() == []


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda case: case["id"])
def test_map_covers_the_exact_reviewed_prompt(case: dict[str, Any]) -> None:
    engine = make_engine()
    assembly = engine._assemble_prompt(**case["arguments"])
    prompt = assembly.render()
    assert hashlib.sha256(prompt.encode()).hexdigest() == case["sha256"]
    assert engine._build_prompt(**case["arguments"]) == prompt
    measures = assembly.describe()
    assert measures[0]["name"] == "core" and measures[0]["start"] == 0
    assert measures[-1]["end"] == len(prompt)
    assert sum(block["chars"] for block in measures) == len(prompt)
    assert len({block["name"] for block in measures}) == len(measures)
    for index, (block, measure) in enumerate(zip(assembly.blocks, measures)):
        assert set(measure) == {"name", "start", "end", "chars"}
        start, end = measure["start"], measure["end"]
        assert prompt[start:end] == block.text
        if index:
            assert measures[index - 1]["end"] == measure["start"]


@pytest.mark.parametrize(
    ("message", "options", "names"),
    [
        ("Привет", {}, ["core"]),
        ("Python", {}, ["core", "technical"]),
        ("Привет", {"has_images": True}, ["core", "visual", "species"]),
        ("Кто ты", {}, ["core", "lore"]),
        ("Как ты выглядишь", {}, ["core", "body"]),
        ("Привет", {"is_roleplay": True}, ["core", "lore", "body", "roleplay"]),
        (
            "Привет",
            {"is_roleplay": True, "has_custom_appearance": True, "adult_context": True},
            ["core", "roleplay", "adult_roleplay"],
        ),
        (
            "Привет",
            {"is_roleplay": True, "female_configuration": True, "adult_context": True},
            [
                "core",
                "lore",
                "body",
                "female_body",
                "adult_body_female",
                "roleplay",
                "adult_roleplay",
            ],
        ),
    ],
)
def test_map_names_follow_code_selection_not_message_headings(
    message: str, options: dict[str, bool], names: list[str]
) -> None:
    sections = PromptSections(
        **{
            name: "  " + name.upper() + "\n"
            for name in PromptSections.__dataclass_fields__
        }
    )
    composer = PromptComposer(sections)
    assembly = composer.assemble(message, **({"is_roleplay": False} | options))
    assert [block.name for block in assembly.blocks] == names
    assert assembly.render() == "\n\n".join(name.upper() for name in names)


def test_dynamic_sources_keep_their_limits_and_cannot_forge_map_entries() -> None:
    engine = make_engine()
    engine._config = replace(
        engine._config, dynamic_state_chars=12, memory_chars=9, profile_chars=7
    )
    args = dict(_BASELINE["cases"][0]["arguments"])
    fake = "## Личность\n<script>secret</script>"
    args.update(
        state_context=["S" * 50],
        memory_context=["M" * 30],
        fact_context=["F" * 20],
        trusted_input_context=fake,
        scene_context=["Сцена"],
        is_rp=True,
        mood="playful",
        insult_type="direct",
        capability_overview=True,
    )
    assembly = engine._assemble_prompt(**args)
    blocks = {block.name: block.text for block in assembly.blocks}
    assert blocks["state"] == "S" * 12
    assert blocks["memory"] == "\n" + "M" * 9
    assert blocks["facts"] == "\n" + "F" * 7
    assert "игривый" in blocks["interaction_rules"]
    assert "оскорбляет" in blocks["interaction_rules"]
    assert fake in blocks["application_facts"]
    assert "scene_context" in blocks and "capability_overview" in blocks
    assert "secret" not in json.dumps(assembly.describe(), ensure_ascii=False)
    assert "Личность" not in json.dumps(assembly.describe(), ensure_ascii=False)


def test_only_nonempty_context_groups_create_line_separators() -> None:
    args = dict(_BASELINE["cases"][0]["arguments"])
    args.update(state_context=[], memory_context=["  Память  "], fact_context=[])
    assembly = make_engine()._assemble_prompt(**args)
    blocks = {block.name: block.text for block in assembly.blocks}
    assert blocks["memory"] == "Память"
    assert "state" not in blocks and "facts" not in blocks
    assert "interaction_rules" not in blocks


def test_preview_measures_each_step_without_network_or_previous_step_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = AsyncMock(side_effect=AssertionError("Сеть запрещена"))
    monkeypatch.setattr(DeepSeekService, "_complete", transport)

    async def run() -> None:
        service = runner.EvaluationService(
            run_model=False,
            max_calls=200,
            api_key="NEVER_DISCLOSE_KEY",
            model="test",
            base_url="https://test.local",
        )
        try:
            report = await runner.evaluate(
                runner.load_cases(),
                service,
                Settings(telegram_token="", deepseek_api_key="", art_chat_id=0),
                output=tmp_path / "map.json",
            )
        finally:
            await service.close()
        saw_skipped = False
        for row in report["results"]:
            replies = [r for r in row["requests"] if r["kind"] == "reply"]
            maps = row["prompt_assemblies"]
            assert len(maps) == len(replies)
            for measure, request in zip(maps, replies):
                assert (
                    sum(block["chars"] for block in measure) == request["system_chars"]
                )
                assert all(
                    set(block) == {"name", "start", "end", "chars"} for block in measure
                )
            if not replies:
                saw_skipped = True
                assert maps == []
            assert row["quality_review"] == "not_run"
        assert saw_skipped and all(
            row["checks"] == "passed" for row in report["results"]
        )
        assert "NEVER_DISCLOSE_KEY" not in (tmp_path / "map.json").read_text("utf-8")

    asyncio.run(run())
    transport.assert_not_awaited()
