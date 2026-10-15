"""Парное сравнение учебных примеров: фиксированные метки, часы и общий бюджет."""

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from protogen_delta.config.settings import Settings
from protogen_delta.services.evaluation import EvaluationService
from protogen_delta.services.prompt_composer import ExampleContract

_SPEC = importlib.util.spec_from_file_location(
    "example_behavior_evaluation", Path(__file__).with_name("evaluate_behavior.py")
)
assert _SPEC is not None and _SPEC.loader is not None
behavior = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(behavior)
CASES_PATH = Path(__file__).with_name("example_contract_cases.json")


class FixedLabelsService(EvaluationService):
    """Сравнивать только сборку примеров, не случайность повторной классификации."""

    async def classify_interaction(self, system_prompt: str, user_message: str) -> str:
        """Одинаковая нейтральная метка для обеих сборок, без сетевого запроса."""
        self.trace["fixed_classifications"] = (
            self.trace.get("fixed_classifications", 0) + 1
        )
        return '{"mood":"neutral","insult":"none","favorite":"none","reasoning":false}'

    async def classify(self, system_prompt: str, user_message: str) -> str:
        """Не тратить запросы на служебное определение роли в нейтральной сцене."""
        self.trace["fixed_classifications"] = (
            self.trace.get("fixed_classifications", 0) + 1
        )
        return "unknown"


async def compare(
    cases: list[dict[str, Any]],
    service: FixedLabelsService,
    settings: Settings,
    *,
    output: Path,
    repeats: int = 2,
    candidate: ExampleContract = "isolated",
) -> dict[str, Any]:
    """Чередовать порядок вариантов и сохранять каждую завершённую пару частями."""
    if not cases or any(not case["steps"] for case in cases):
        raise ValueError("Нужны непустые сценарии")
    if candidate not in {"isolated", "guidance"}:
        raise ValueError("Нужен отдельный кандидат isolated или guidance")
    if not 1 <= repeats <= 3:
        raise ValueError("Допустимо от одного до трёх повторов")
    expected = sum(len(case["steps"]) for case in cases) * repeats * 2
    if expected > service.trace["max_calls"] - len(service.trace["requests"]):
        raise ValueError("Общий бюджет меньше запланированного числа ответов")
    report: dict[str, Any] = {
        "mode": "model" if service.trace["run_model"] else "preview",
        "complete": False,
        "planned_reply_requests": expected,
        "contracts": ["legacy", candidate],
        "classification_source": "fixed_neutral_and_unknown_role",
        "history_scope": "synthetic_scenario_ram",
        "limitations": [
            "Это сравнение сборки примеров, не оценка классификаторов и полного рабочего диалога.",
            "Часы и метки фиксированы; различается контракт стилевых примеров.",
            "Отсутствие совпадений с шаблоном не доказывает достоверность и сохранение характера.",
        ],
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        """Не терять частичный результат при сбое или исчерпании лимита."""
        report["calls"] = len(service.trace["requests"])
        report["budget_exceeded"] = service.trace["budget_exceeded"]
        report["fixed_classifications"] = service.trace.get("fixed_classifications", 0)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")

    try:
        for repeat in range(repeats):
            for index, case in enumerate(cases):
                contracts = (
                    ("legacy", candidate)
                    if (index + repeat) % 2 == 0
                    else (candidate, "legacy")
                )
                for contract in contracts:
                    before = len(service.trace["requests"])
                    result = await behavior.evaluate(
                        [case],
                        service,
                        settings,
                        output=output.with_suffix(".partial.json"),
                        example_contract=contract,
                        stop_on_error=True,
                    )
                    for item in result["results"]:
                        report["rows"].append(
                            {
                                "case": case["id"],
                                "repeat": repeat + 1,
                                "contract": contract,
                                "result": item,
                                "prompts": result["prompts"],
                                "cases_sha256": result["cases_sha256"],
                                "state_clock": result["state_clock"],
                            }
                        )
                    save()
                    requests = service.trace["requests"][before:]
                    if (
                        len(requests) != len(case["steps"])
                        or any(item["kind"] != "reply" for item in requests)
                        or any("error" in item for item in requests)
                        or any(item["generation_error"] for item in result["results"])
                    ):
                        return report
        report["complete"] = True
        return report
    except Exception as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        save()


async def main() -> None:
    """Без явного --run-model сеть и чтение ключей отключены."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=48)
    parser.add_argument(
        "--candidate", choices=("isolated", "guidance"), default="isolated"
    )
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--env-file", type=Path)
    credentials.add_argument("--use-environment", action="store_true")
    args = parser.parse_args()
    try:
        cases = behavior.load_cases(args.cases)
        if not 1 <= args.repeats <= 3 or not 1 <= args.max_calls <= 1000:
            raise ValueError("Неверное число повторов или общий бюджет")
        if (
            sum(len(case["steps"]) for case in cases) * args.repeats * 2
            > args.max_calls
        ):
            raise ValueError("Бюджет меньше запланированного числа ответов")
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
            "Для --run-model нужен ключ через --env-file или --use-environment"
        )
    settings = Settings(
        telegram_token="",
        deepseek_api_key="",
        art_chat_id=0,
        deepseek_model=env.get("DEEPSEEK_MODEL") or "deepseek-flash",
        deepseek_base_url=env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    )
    service = FixedLabelsService(
        run_model=args.run_model,
        max_calls=args.max_calls,
        api_key=env.get("DEEPSEEK_API_KEY") or "preview-no-network",
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        adaptive_thinking=True,
        disable_thinking=True,
    )
    try:
        report = await compare(
            cases,
            service,
            settings,
            output=args.output,
            repeats=args.repeats,
            candidate=args.candidate,
        )
    finally:
        await service.close()
    failed = sum(row["result"]["checks"] == "failed" for row in report["rows"])
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("mode", "complete", "calls", "budget_exceeded")
            }
            | {"failed": failed},
            ensure_ascii=False,
        )
    )
    if not report["complete"] or failed or report["budget_exceeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
