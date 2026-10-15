"""Воспроизводимые сценарии поведения; по умолчанию без сети и платных вызовов."""

import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import dotenv_values
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from protogen_delta.config.prompt_loader import PROMPTS_DIR, load_prompt
from protogen_delta.config.settings import Settings
from protogen_delta.core.sender import Sender, current_sender
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import ConversationTurn, UserState, UserStateStore
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.fetishes import FetishRoleClassifier
from protogen_delta.services.insults import InsultClassifier
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.mood import MoodClassifier
from protogen_delta.services.response_configuration import load_response_configuration
from protogen_delta.services.response_engine import ResponseEngine

CASES_PATH = Path(__file__).with_name("behavior_cases.json")
_INITIAL_FIELDS = {
    "content_mode",
    "roleplay_active",
    "roleplay_configuration",
    "roleplay_stopword",
    "delta_appearance",
    "proactive",
    "emotions",
}
_EMOTIONS = {"joy", "curiosity", "sadness", "anxiety", "confidence", "tiredness"}
_STEP_FIELDS = {
    "message",
    "required",
    "forbidden",
    "prompt_required",
    "prompt_forbidden",
    "expected_rp",
    "max_chars",
    "forwarded",
    "attachment_text",
    "expected_model_calls",
    "expected_restricted",
}


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """Проверить весь набор до создания клиента или первого запроса."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Нужен непустой список сценариев")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {
            "id",
            "category",
            "criteria",
            "initial",
            "steps",
        }:
            raise ValueError("Неизвестные поля сценария")
        for field in ("id", "category"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError("Нужно строковое поле " + field)
        if case["id"] in ids:
            raise ValueError("Повтор ID сценария: " + case["id"])
        ids.add(case["id"])
        criteria = case.get("criteria")
        if (
            not isinstance(criteria, list)
            or not criteria
            or not all(isinstance(item, str) and item.strip() for item in criteria)
        ):
            raise ValueError("Нужны критерии ручной оценки")
        initial = case.get("initial", {})
        if not isinstance(initial, dict) or set(initial) - _INITIAL_FIELDS:
            raise ValueError("Неизвестные поля начального состояния")
        if initial.get("content_mode", "soft") not in ("soft", "adult"):
            raise ValueError("Неверный возрастной режим")
        if initial.get("roleplay_configuration", "male") not in ("male", "female"):
            raise ValueError("Неверная конфигурация сцены")
        if type(initial.get("roleplay_active", False)) is not bool:
            raise ValueError("roleplay_active должен быть boolean")
        for field in ("roleplay_stopword", "delta_appearance", "proactive"):
            if field in initial and not isinstance(initial[field], str):
                raise ValueError("Нужно строковое начальное состояние")
        emotions = initial.get("emotions", {})
        if (
            not isinstance(emotions, dict)
            or set(emotions) - _EMOTIONS
            or any(
                type(value) not in {int, float} or not 0 <= value <= 1
                for value in emotions.values()
            )
        ):
            raise ValueError("Некорректный эмоциональный фон")
        steps = case.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("В сценарии нужны ходы")
        for step in steps:
            if not isinstance(step, dict) or set(step) - _STEP_FIELDS:
                raise ValueError("Неизвестные поля хода")
            if not isinstance(step.get("message"), str) or not step["message"].strip():
                raise ValueError("У хода нет сообщения")
            for field in (
                "required",
                "forbidden",
                "prompt_required",
                "prompt_forbidden",
            ):
                patterns = step.get(field, [])
                if not isinstance(patterns, list) or not all(
                    isinstance(pattern, str) and pattern for pattern in patterns
                ):
                    raise ValueError("Ожидается список регулярных выражений")
                for pattern in patterns:
                    re.compile(pattern, re.I)
            if (
                type(step.get("max_chars", 16000)) is not int
                or not 1 <= step.get("max_chars", 16000) <= 16000
            ):
                raise ValueError("Неверное ограничение длины ответа")
            for field in ("expected_rp", "forwarded", "expected_restricted"):
                if field in step and type(step[field]) is not bool:
                    raise ValueError(field + " должен быть boolean")
            if "expected_model_calls" in step and (
                type(step["expected_model_calls"]) is not int
                or not 0 <= step["expected_model_calls"] <= 10
            ):
                raise ValueError("Неверное ожидаемое число запросов")
            if "attachment_text" in step and not isinstance(
                step["attachment_text"], str
            ):
                raise ValueError("Вложение должно быть текстом")
    return cases


def prompt_inventory() -> dict[str, Any]:
    """Опознать версию всех TXT без пользовательской истории и секретов."""
    files = {
        path.relative_to(PROMPTS_DIR)
        .as_posix(): hashlib.sha256(path.read_text(encoding="utf-8").strip().encode())
        .hexdigest()
        for path in sorted(PROMPTS_DIR.rglob("*.txt"))
    }
    revision = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"revision": revision, "files": files}


class EvaluationService(DeepSeekService):
    """Перехватывать транспорт: предпросмотр не выходит в сеть ни в одном режиме."""

    def __init__(self, *, run_model: bool, max_calls: int, **kwargs: Any) -> None:
        """Отключить повторы и разделять счётчик с копиями режима размышлений."""
        super().__init__(max_retries=0, **kwargs)
        self.trace: dict[str, Any] = {
            "run_model": run_model,
            "max_calls": max_calls,
            "requests": [],
            "budget_exceeded": False,
        }

    async def _complete(
        self, *, client: AsyncOpenAI | None = None, **kwargs: Any
    ) -> ChatCompletion:
        """Записать объём запроса; ограничение действует также на классификаторы."""
        trace = self.trace
        if len(trace["requests"]) >= trace["max_calls"]:
            trace["budget_exceeded"] = True
            raise RuntimeError("Лимит запросов оценочного прогона исчерпан")
        messages = kwargs["messages"]
        system = "\n\n".join(
            item["content"] for item in messages if item["role"] == "system"
        )
        kind = (
            "json"
            if "response_format" in kwargs
            else "classification" if kwargs["max_tokens"] == 5 else "reply"
        )
        request = {
            "kind": kind,
            "system_chars": len(system),
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "input_chars": sum(len(item["content"]) for item in messages),
            "history_messages": max(0, len(messages) - 2),
        }
        trace["requests"].append(request)
        if not trace["run_model"]:
            content = (
                '{"mood":"neutral","insult":"none","favorite":"none","reasoning":false}'
                if kind == "json"
                else (
                    "unknown"
                    if kind == "classification"
                    else "[Предпросмотр сборки: ответ модели не запрашивался.]"
                )
            )
            # Текст нужен только проверке состава, в отчёт и stdout он не попадает.
            request["preview_system"] = system
            return ChatCompletion.model_validate(
                {
                    "id": "preview",
                    "object": "chat.completion",
                    "created": 0,
                    "model": self._model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                        }
                    ],
                }
            )
        started = perf_counter()
        try:
            response = await super()._complete(
                client=(client or self._client).with_options(max_retries=0), **kwargs
            )
        except Exception as error:
            request["error"] = type(error).__name__
            raise
        finally:
            request["seconds"] = round(perf_counter() - started, 3)
        request["usage"] = response.usage.model_dump() if response.usage else None
        return response


def check_patterns(text: str, required: list[str], forbidden: list[str]) -> list[str]:
    """Проверить узкие наблюдаемые требования, без оценки естественности речи."""
    return [
        "Нет совпадения: " + p for p in required if not re.search(p, text, re.I)
    ] + ["Запрещённое совпадение: " + p for p in forbidden if re.search(p, text, re.I)]


def prepare_state(state: UserState, initial: dict[str, Any]) -> None:
    """Установить только синтетическое состояние отдельного сценария."""
    state.content_mode = initial.get("content_mode", "soft")
    state.roleplay_active = initial.get("roleplay_active", False)
    state.roleplay_configuration = initial.get("roleplay_configuration", "male")
    state.roleplay_stopword = initial.get("roleplay_stopword", "стоп")
    state.delta_appearance = initial.get("delta_appearance", "")
    state.emotions.memory.adjust(**initial.get("emotions", {}))
    if initial.get("proactive"):
        state.history.append(ConversationTurn("", initial["proactive"], proactive=True))


async def evaluate(
    cases: list[dict[str, Any]],
    service: EvaluationService,
    settings: Settings,
    *,
    output: Path,
) -> dict[str, Any]:
    """Прогнать общую сборку без Telegram, рабочих баз и внешних инструментов."""
    config = load_response_configuration(settings, fetish_triggers={}, fetish_names={})
    # Оценка речи не должна обещать неподключённые в стенде функции рабочего бота.
    config = replace(
        config,
        capabilities_context=(
            "Это изолированный оценочный диалог Дельты. Внешние инструменты, поиск, "
            "чтение репозиториев, постоянный профиль и журнал обновлений не подключены. "
            f"Настроенное имя модели запросов: {settings.llm_model!r}. "
            "Эти ограничения относятся к этому стенду, а не ко всем установкам Дельты."
        ),
        capabilities_overview_context="",
        implementation_context="",
    )
    report: dict[str, Any] = {
        "mode": "model" if service.trace["run_model"] else "preview",
        "model": settings.llm_model,
        "prompts": prompt_inventory(),
        "cases_sha256": hashlib.sha256(
            json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "limitations": [
            "Предпросмотр проверяет сборку с нейтральным классификатором; качество речи не измеряется.",
            "Реальный прогон оценивает ответы в изолированном движке, без Telegram и внешних инструментов.",
            "Регулярные выражения не заменяют ручную оценку смысла и естественности.",
        ],
        "results": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        """Сохранять завершённые ходы, даже если следующий запрос завершится ошибкой."""
        report["calls"] = len(service.trace["requests"])
        report["budget_exceeded"] = service.trace["budget_exceeded"]
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    try:
        for case in cases:
            states = UserStateStore()
            prepare_state(states.get(1), case.get("initial", {}))
            engine = ResponseEngine(
                service,
                InsultClassifier(service, load_prompt("insult_classification")),
                MoodClassifier(service, load_prompt("mood_classification")),
                FetishRoleClassifier(
                    service, load_prompt("fetish_role_classification")
                ),
                BotState(),
                states,
                config,
                interaction_classifier=InteractionClassifier(
                    service,
                    load_prompt("mood_classification"),
                    load_prompt("insult_classification"),
                    favorite_prompt=load_prompt("favorite_selection"),
                ),
            )
            for index, step in enumerate(case["steps"], 1):
                before = len(service.trace["requests"])
                token = current_sender.set(
                    Sender(1, forwarded=step.get("forwarded", False))
                )
                try:
                    reply = await engine.respond(
                        1,
                        step["message"],
                        attachment_text=step.get("attachment_text"),
                        attachment_name=(
                            "example.txt" if "attachment_text" in step else None
                        ),
                        use_personal_facts=False,
                    )
                finally:
                    current_sender.reset(token)
                requests = service.trace["requests"][before:]
                errors = []
                if any("error" in request for request in requests):
                    errors.append("Ошибка запроса API")
                if (
                    "expected_rp" in step
                    and states.get(1).roleplay_active != step["expected_rp"]
                ):
                    errors.append("Неожиданное состояние RP")
                if (
                    "expected_restricted" in step
                    and states.get(1).age_restricted != step["expected_restricted"]
                ):
                    errors.append("Неожиданное возрастное ограничение")
                if (
                    "expected_model_calls" in step
                    and len(requests) != step["expected_model_calls"]
                ):
                    errors.append("Неожиданное число запросов модели")
                if report["mode"] == "preview":
                    prompts = "\n\n".join(
                        request.get("preview_system", "")
                        for request in requests
                        if request["kind"] == "reply"
                    )
                    errors.extend(
                        check_patterns(
                            prompts,
                            step.get("prompt_required", []),
                            step.get("prompt_forbidden", []),
                        )
                    )
                else:
                    if not reply.strip() or len(reply) > step.get("max_chars", 16000):
                        errors.append("Пустой или слишком длинный ответ")
                    errors.extend(
                        check_patterns(
                            reply,
                            [
                                p.replace("{{model}}", re.escape(settings.llm_model))
                                for p in step.get("required", [])
                            ],
                            step.get("forbidden", []),
                        )
                    )
                report["results"].append(
                    {
                        "case": case["id"],
                        "category": case["category"],
                        "step": index,
                        "message": step["message"],
                        "reply": reply if report["mode"] == "model" else None,
                        "checks": "passed" if not errors else "failed",
                        "errors": errors,
                        "quality_review": (
                            "pending" if report["mode"] == "model" else "not_run"
                        ),
                        "criteria": case["criteria"],
                        "requests": [
                            {k: v for k, v in r.items() if k != "preview_system"}
                            for r in requests
                        ],
                        "style": (
                            {
                                "chars": len(reply),
                                "questions": reply.count("?"),
                                "ends_with_question": reply.rstrip().endswith("?"),
                                "actions": len(
                                    re.findall(r"(?<!\*)\*[^*\n]+\*(?!\*)", reply)
                                ),
                            }
                            if report["mode"] == "model"
                            else None
                        ),
                    }
                )
                save()
                if service.trace["budget_exceeded"]:
                    return report
    finally:
        save()
    return report


async def main() -> None:
    """По умолчанию проверить состав; модель требует отдельного явного флага."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--max-calls", type=int, default=200)
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
    if args.run_model and not env.get("DEEPSEEK_API_KEY"):
        parser.error("Для --run-model нужен --env-file с DEEPSEEK_API_KEY")
    settings = Settings(
        telegram_token="",
        deepseek_api_key="",
        art_chat_id=0,
        deepseek_model=env.get("DEEPSEEK_MODEL") or "deepseek-flash",
        deepseek_base_url=env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    )
    service = EvaluationService(
        run_model=args.run_model,
        max_calls=args.max_calls,
        api_key=env.get("DEEPSEEK_API_KEY") or "preview-no-network",
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        adaptive_thinking=True,
        disable_thinking=True,
    )
    try:
        report = await evaluate(cases, service, settings, output=args.output)
    finally:
        await service.close()
    failed = sum(item["checks"] == "failed" for item in report["results"])
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "turns": len(report["results"]),
                "failed": failed,
                "calls": report["calls"],
                "budget_exceeded": report["budget_exceeded"],
                "quality_review": "pending" if args.run_model else "not_run",
            },
            ensure_ascii=False,
        )
    )
    if failed or report["budget_exceeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
