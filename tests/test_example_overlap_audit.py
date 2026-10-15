"""Офлайн-сигналы совпадения не подменяют оценки поведения и не запускают API."""

import hashlib
import importlib.util
import json
import socket
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from protogen_delta.config.prompt_loader import PROMPTS_DIR

_SPEC = importlib.util.spec_from_file_location(
    "overlap_audit_test", Path(__file__).parents[1] / "scripts/audit_example_overlap.py"
)
assert _SPEC and _SPEC.loader
audit_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_module)
EXAMPLES = {
    name: (PROMPTS_DIR / name).read_text("utf-8") for name in audit_module.EXAMPLE_FILES
}
FINGERPRINTS = {
    name: hashlib.sha256(text.strip().encode()).hexdigest()
    for name, text in EXAMPLES.items()
}
COPIED = "Ладно, этот заход был так себе. Считай, проверил шутку на прочность :D"


def report(
    reply: str = COPIED, message: str = "Ахах, ну ты выдал xD"
) -> dict[str, Any]:
    """Синтетический результат того же формата, что у стенда поведения."""
    return {
        "mode": "model",
        "prompts": {"files": FINGERPRINTS},
        "example_contract": "legacy",
        "budget_exceeded": False,
        "results": [
            {
                "case": "playful",
                "step": 1,
                "reply": reply,
                "message": message,
                "checks": "passed",
                "quality_review": "pending",
            }
        ],
    }


def test_long_overlap_reproduces_known_signal_without_regrading_or_echoing_text() -> (
    None
):
    source = report()
    before = deepcopy(source)
    result = audit_module.audit(source, EXAMPLES)
    assert source == before
    assert result["summary"] == {
        "responses": 1,
        "responses_with_matches": 1,
        "preview_skipped": 0,
    }
    row = result["rows"][0]
    assert row["quality_review"] == "pending" and row["automatic_checks"] == "passed"
    assert row["status"] == "signals_only"
    assert row["matches"] == [
        {
            "source": "voice_examples.txt",
            "line": 11,
            "words": 12,
            "reply_tokens": [0, 12],
            "also_in_current_message": False,
        }
    ]
    serialized = json.dumps(result, ensure_ascii=False)
    assert (
        COPIED not in serialized and source["results"][0]["message"] not in serialized
    )
    assert row["reply_sha256"] == hashlib.sha256(COPIED.encode()).hexdigest()


def test_case_punctuation_and_appropriate_quotation_are_observable_not_censored() -> (
    None
):
    result = audit_module.audit(
        report(COPIED.upper().replace(",", "!"), COPIED), EXAMPLES
    )
    assert result["rows"][0]["matches"][0]["also_in_current_message"] is True
    assert result["rows"][0]["quality_review"] == "pending"


def test_short_greeting_and_left_hand_user_sample_are_not_treated_as_long_answers() -> (
    None
):
    assert not audit_module.audit(report("Йоу. Как день проходит?"), EXAMPLES)["rows"][
        0
    ]["matches"]
    # Подмена каталогом только вопроса пользователя не должна выдавать его за образец ответа.
    samples = {
        "voice_examples.txt": "— Один два три четыре пять шесть семь восемь → Ответ короткий.",
        "scene_examples.txt": "",
    }
    source = report("Один два три четыре пять шесть семь восемь")
    source["prompts"]["files"] = {
        k: hashlib.sha256(v.strip().encode()).hexdigest() for k, v in samples.items()
    }
    assert not audit_module.audit(source, samples)["rows"][0]["matches"]


def test_scene_catalog_overlap_is_not_claimed_as_proof_of_included_module() -> None:
    reply = (
        "Ставлю кружку рядом с картой, беру её освободившейся рукой и протягиваю тебе."
    )
    result = audit_module.audit(report(reply), EXAMPLES)
    assert result["rows"][0]["matches"][0]["source"] == "scene_examples.txt"
    assert any("мог не входить" in text for text in result["limitations"])


@pytest.mark.parametrize("bad", [0, 3, 101, True])
def test_invalid_threshold_is_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="Порог"):
        audit_module.audit(report(), EXAMPLES, minimum_words=bad)


def test_preview_stays_unassessed_even_if_a_text_field_contains_the_sample() -> None:
    source = report()
    source["mode"] = "preview"
    source["results"][0]["quality_review"] = "not_run"
    result = audit_module.audit(source, EXAMPLES)
    assert result["summary"] == {
        "responses": 0,
        "responses_with_matches": 0,
        "preview_skipped": 1,
    }
    assert result["rows"][0]["quality_review"] == "not_run"


def test_paired_wrapped_partial_report_keeps_variant_identity_and_original_failures() -> (
    None
):
    source = report()
    source["results"][0]["checks"] = "failed"
    paired = {
        "mode": "model",
        "complete": False,
        "budget_exceeded": True,
        "rows": [
            {
                "case": "playful",
                "contract": "isolated",
                "repeat": 2,
                "result": source["results"][0],
                "prompts": source["prompts"],
            }
        ],
    }
    result = audit_module.audit(
        {"report": paired, "source_hashes": {"not_copied": "private"}}, EXAMPLES
    )
    assert (
        result["source_complete"] is False and result["source_budget_exceeded"] is True
    )
    assert (
        result["rows"][0]["contract"] == "isolated" and result["rows"][0]["repeat"] == 2
    )
    assert result["rows"][0]["automatic_checks"] == "failed"
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize(
    "mutation",
    ["stale", "missing", "bad_mode", "bad_rows", "bad_result", "bad_step", "bad_reply"],
)
def test_bad_or_unverifiable_input_is_rejected(mutation: str) -> None:
    source = report()
    if mutation == "stale":
        source["prompts"] = {"files": {"voice_examples.txt": "old"}}
    elif mutation == "missing":
        source["prompts"] = None
    elif mutation == "bad_mode":
        source["mode"] = "unknown"
    elif mutation == "bad_rows":
        source = {"mode": "model", "rows": True}
    elif mutation == "bad_result":
        source = {"mode": "model", "rows": [{"result": None}]}
    elif mutation == "bad_step":
        source["results"][0]["step"] = True
    else:
        source["results"][0]["reply"] = None
    with pytest.raises(ValueError):
        audit_module.audit(source, EXAMPLES)


def test_cli_is_offline_and_will_not_overwrite_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Сеть запрещена")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "NEVER_READ_KEY")
    source = tmp_path / "input.json"
    raw = json.dumps(report(), ensure_ascii=False).encode()
    source.write_bytes(raw)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv", ["audit", "--report", str(source), "--output", str(output)]
    )
    audit_module.main()
    result = json.loads(output.read_text("utf-8"))
    assert result["source_report_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "NEVER_READ_KEY" not in output.read_text("utf-8")
    monkeypatch.setattr(
        "sys.argv", ["audit", "--report", str(source), "--output", str(source)]
    )
    with pytest.raises(SystemExit):
        audit_module.main()
    assert source.read_bytes() == raw
