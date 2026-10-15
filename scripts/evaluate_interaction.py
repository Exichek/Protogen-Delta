"""Сравнить два контракта классификации на размеченных синтетических репликах."""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.favorite_messages import FAVORITE_LABELS
from protogen_delta.services.evaluation import EvaluationService
from protogen_delta.services.interaction_classification import InteractionClassifier

CASES_PATH = Path(__file__).with_name("interaction_cases.json")
LABELS: dict[str, set[Any]] = {
    "mood": {"sweet", "horny", "angry", "playful", "neutral"},
    "insult": {"general", "direct", "question", "none"},
    "favorite": {"none", *FAVORITE_LABELS},
    "reasoning": {False, True},
}


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """Отклонить неполную разметку до создания клиента и расхода API."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Нужен непустой список размеченных реплик")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {
            "id",
            "category",
            "message",
            "reasoning_context",
            "expected",
            "favorites",
            "note",
        }:
            raise ValueError("Неверные поля сценария")
        for key in ("id", "category", "message"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError("Не задано поле " + key)
        if case["id"] in seen:
            raise ValueError("Повтор ID: " + case["id"])
        seen.add(case["id"])
        if type(case.get("favorites", True)) is not bool:
            raise ValueError("favorites должен быть boolean")
        for key in ("reasoning_context", "note"):
            if key in case and not isinstance(case[key], str):
                raise ValueError(key + " должен быть строкой")
        expected = case.get("expected")
        fields = set(LABELS) - (set() if case.get("favorites", True) else {"favorite"})
        if not isinstance(expected, dict) or set(expected) != fields:
            raise ValueError("Разметка должна содержать все поля контракта")
        for field, choices in expected.items():
            if (
                not isinstance(choices, list)
                or not choices
                or any(
                    type(value) is not (bool if field == "reasoning" else str)
                    or value not in LABELS[field]
                    for value in choices
                )
            ):
                raise ValueError("Неверная разметка поля " + field)
    return cases


def wire_errors(raw: str, *, favorites: bool) -> list[str]:
    """Отделить ошибки формата от неверного выбора смысловой метки."""
    try:
        value = json.loads(raw)
    except ValueError:
        return ["json"]
    fields = set(LABELS) - (set() if favorites else {"favorite"})
    if not isinstance(value, dict) or set(value) != fields:
        return ["fields"]
    return [
        field
        for field, label in value.items()
        if (
            type(label) is not (bool if field == "reasoning" else str)
            or label not in LABELS[field]
        )
    ]


def summarize(rows: list[dict[str, Any]], *, measured: bool) -> dict[str, Any]:
    """Сравнить пары по разметке; отсутствие замера не превращать в нулевую ошибку."""
    if not measured:
        return {"measured": False, "regressions": None, "improvements": None}
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["contract"]] = row
    regressions, improvements, changes, invalid_pairs = [], [], [], []
    for case_id, pair in by_case.items():
        if set(pair) != {"legacy", "single"}:
            continue
        old, new = pair["legacy"], pair["single"]
        if "api" in old["wire_errors"] or "api" in new["wire_errors"]:
            invalid_pairs.append(case_id)
            continue
        changed = {
            field: {"legacy": old["labels"][field], "single": value}
            for field, value in new["labels"].items()
            if value != old["labels"][field]
        }
        if changed:
            changes.append({"case": case_id, "fields": changed})
        before, after = set(old["label_errors"]), set(new["label_errors"])
        if after - before or (new["wire_errors"] and not old["wire_errors"]):
            regressions.append(case_id)
        if before - after or (old["wire_errors"] and not new["wire_errors"]):
            improvements.append(case_id)
    return {
        "measured": True,
        "regressions": regressions,
        "improvements": improvements,
        "changes": changes,
        "invalid_pairs": invalid_pairs,
        "variants": {
            name: {
                "rows": sum(row["contract"] == name for row in rows),
                "wire_failures": sum(
                    bool(row["wire_errors"]) for row in rows if row["contract"] == name
                ),
                "label_failures": sum(
                    bool(row["label_errors"]) for row in rows if row["contract"] == name
                ),
            }
            for name in ("legacy", "single")
        },
    }


async def compare(
    cases: list[dict[str, Any]],
    service: EvaluationService,
    *,
    output: Path,
) -> dict[str, Any]:
    """Дать обоим вариантам одинаковые реплики, без истории людей и Telegram."""
    measured = service.trace["run_model"]
    report: dict[str, Any] = {
        "mode": "model" if measured else "preview",
        "model": service._model,
        "cases_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "annotation_note": (
            "Допустимые варианты заданы до запуска. Это малая выборка, "
            "не универсальная оценка точности."
        ),
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        """Сохранить частичный результат, включая ошибки и исчерпание бюджета."""
        report["summary"] = summarize(report["rows"], measured=measured)
        report["calls"] = len(service.trace["requests"])
        report["budget_exceeded"] = service.trace["budget_exceeded"]
        report["complete"] = (
            len(report["rows"]) == 2 * len(cases) and not report["budget_exceeded"]
        )
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    try:
        for index, case in enumerate(cases):
            # Чередование исключает постоянный приоритет одного варианта по времени.
            contracts: tuple[Literal["legacy", "single"], ...] = (
                ("legacy", "single") if index % 2 == 0 else ("single", "legacy")
            )
            for contract in contracts:
                favorites = case.get("favorites", True)
                classifier = InteractionClassifier(
                    service,
                    load_prompt("mood_classification"),
                    load_prompt("insult_classification"),
                    favorite_prompt=(
                        load_prompt("favorite_selection") if favorites else None
                    ),
                    prompt_contract=contract,
                )
                before = len(service.trace["requests"])
                result = await classifier.classify(
                    case["message"], reasoning_context=case.get("reasoning_context", "")
                )
                requests = service.trace["requests"][before:]
                raw = (
                    requests[-1].get("classification_response", "") if requests else ""
                )
                labels = asdict(result)
                labels["favorite"] = labels["favorite"] or "none"
                errors = wire_errors(raw, favorites=favorites) if measured else None
                if measured and any("error" in request for request in requests):
                    errors = ["api"]
                report["rows"].append(
                    {
                        "case": case["id"],
                        "category": case["category"],
                        "contract": contract,
                        "message": case["message"],
                        "expected": case["expected"],
                        "labels": labels if measured else None,
                        "wire_errors": errors,
                        "label_errors": (
                            [
                                field
                                for field, choices in case["expected"].items()
                                if labels[field] not in choices
                            ]
                            if measured
                            else None
                        ),
                        "prompt_sha256": hashlib.sha256(
                            classifier._prompt.encode()
                        ).hexdigest(),
                        "prompt_chars": len(classifier._prompt),
                        "requests": [
                            {
                                key: value
                                for key, value in request.items()
                                if key != "preview_system"
                            }
                            for request in requests
                        ],
                    }
                )
                save()
                if service.trace["budget_exceeded"]:
                    return report
    finally:
        save()
    return report


async def main() -> None:
    """Предпросмотр безопасен по умолчанию; реальный замер включается явно."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--max-calls", type=int, default=80)
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--env-file", type=Path)
    credentials.add_argument("--use-environment", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 1000:
        parser.error("--max-calls должен быть от 1 до 1000")
    try:
        cases = load_cases(args.cases)
        unknown = set(args.case) - {case["id"] for case in cases}
        if unknown:
            raise ValueError("Неизвестные сценарии: " + ", ".join(sorted(unknown)))
        if args.case:
            cases = [case for case in cases if case["id"] in args.case]
    except (ValueError, OSError) as error:
        parser.error(str(error))
    env = dotenv_values(args.env_file) if args.run_model and args.env_file else {}
    if args.run_model and args.use_environment:
        env = {
            name: os.environ.get(name)
            for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL")
        }
    if args.run_model and not env.get("DEEPSEEK_API_KEY"):
        parser.error(
            "Для --run-model нужен --env-file или --use-environment с DEEPSEEK_API_KEY"
        )
    service = EvaluationService(
        run_model=args.run_model,
        max_calls=args.max_calls,
        api_key=env.get("DEEPSEEK_API_KEY") or "preview-no-network",
        base_url=env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        model=env.get("DEEPSEEK_MODEL") or "deepseek-flash",
    )
    try:
        report = await compare(cases, service, output=args.output)
    finally:
        await service.close()
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("mode", "complete", "calls", "budget_exceeded", "summary")
            },
            ensure_ascii=False,
        )
    )
    if not report["complete"] or (
        args.run_model
        and (
            report["summary"]["regressions"]
            or report["summary"]["invalid_pairs"]
            or report["summary"]["variants"]["single"]["wire_failures"]
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
