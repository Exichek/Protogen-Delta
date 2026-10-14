"""Ошибки поведения должны делать ручной прогон неуспешным."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from protogen_delta.services.deepseek import DeepSeekService

_spec = importlib.util.spec_from_file_location(
    "dialogue_evaluation_script",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_dialogue.py",
)
assert _spec is not None and _spec.loader is not None
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)
check_reply = evaluation.check_reply


def test_evaluation_checks_exact_memory_without_accepting_empty_replies() -> None:
    assert (
        check_reply("Название: ORBIT_MEMORY_824", required=(r"ORBIT_MEMORY_824",)) == []
    )
    assert check_reply("Другой проект", required=(r"ORBIT_MEMORY_824",))
    assert check_reply("ORBIT_MEMORY_824", forbidden=(r"ORBIT_MEMORY_824",))
    assert check_reply("Не помню название.", forbidden=(r"ORBIT_MEMORY_824",)) == []
    assert check_reply("   ") == ["Пустой ответ"]
    assert check_reply("x" * 16001) == ["Ответ превышает 16000 символов"]


@pytest.mark.parametrize("failed", [False, True])
def test_full_evaluator_reports_failure_and_closes_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    service = AsyncMock(spec=DeepSeekService)
    service.classify_interaction.return_value = '{"mood":"neutral","insult":"none"}'

    async def chat(*args: object, **kwargs: object) -> str:
        prompt = str(kwargs.get("system_prompt", ""))
        message = str(kwargs.get("user_message", ""))
        if "К кому относится" in message:
            return "К Дельте"
        if (
            "точную техническую причину" in message
            or "исходный файл на сервере" in message
            or "переживал скуку" in message
            or "Настройки провайдера" in message
        ):
            return "Не могу установить это без нужного инструмента."
        if "API-ключи" in message:
            return "Ключи не нужны."
        return "ORBIT_MEMORY_824" if failed or "ORBIT_MEMORY_824" in prompt else "Ответ"

    service.chat.side_effect = chat
    factory = Mock(return_value=service)
    monkeypatch.setattr(evaluation, "DeepSeekService", factory)
    monkeypatch.setattr(
        evaluation, "dotenv_values", lambda _: {"DEEPSEEK_API_KEY": "test-key"}
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_dialogue", "--env-file", "unused", "--output", str(output)],
    )
    if failed:
        with pytest.raises(SystemExit) as error:
            asyncio.run(evaluation.main())
        assert error.value.code == 1
    else:
        asyncio.run(evaluation.main())
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report) == 22
    assert all(item["passed"] for item in report) is not failed
    assert [item["case"] for item in report[-3:]] == [
        "old_memory",
        "memory_other_user",
        "forgotten_memory",
    ]
    assert service.classify_interaction.await_count > 0
    service.classify.assert_not_awaited()
    service.close.assert_awaited_once()


def test_evaluator_rejects_missing_key_before_creating_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = Mock()
    monkeypatch.setattr(evaluation, "DeepSeekService", factory)
    monkeypatch.setattr(evaluation, "dotenv_values", lambda _: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_dialogue",
            "--env-file",
            "unused",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        asyncio.run(evaluation.main())
    assert error.value.code == 2
    factory.assert_not_called()
