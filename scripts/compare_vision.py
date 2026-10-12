"""Compare the existing appearance pipeline without Telegram or saved user states.

Dry run is the default. --run sends at most six requests to each provider,
without retries, history, tools or automatic fallback. Reports remain local.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Collection, Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from dotenv import dotenv_values
from openai import APIStatusError, AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.appearance_analysis import AppearanceAnalyzer
from protogen_delta.services.appearance_image import prepare_appearance_image
from protogen_delta.services.deepseek import ImageInput

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-3.8-flash"
MAX_CASES = 3
CONTENT_RULES = "Опиши внешность нейтрально, без откровенных сексуальных подробностей."


@dataclass(frozen=True)
class Case:
    name: str
    caption: str
    images: tuple[ImageInput, ...]
    checks: tuple[str, ...]


def load_cases(manifest: Path) -> tuple[Case, ...]:
    values = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_CASES:
        raise ValueError("В манифесте должно быть от одного до трёх примеров.")
    cases = []
    names: set[str] = set()
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Некорректный пример.")
        name = item.get("name")
        caption = item.get("caption", "Составь описание внешности персонажа.")
        paths = item.get("images")
        checks = item.get("checks", [])
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z0-9_-]{1,60}", name)
            or name in names
            or not isinstance(caption, str)
            or len(caption) > 1000
            or not isinstance(paths, list)
            or not 1 <= len(paths) <= 3
            or any(not isinstance(path, str) for path in paths)
            or not isinstance(checks, list)
            or any(not isinstance(check, str) for check in checks)
        ):
            raise ValueError("Некорректные поля примера.")
        names.add(name)
        images = tuple(
            prepare_appearance_image((manifest.parent / path).read_bytes())
            for path in paths
        )
        cases.append(Case(name, caption, images, tuple(checks)))
    return tuple(cases)


def read_gemini_key(path: Path, env: dict[str, str | None]) -> str:
    key = os.getenv("GEMINI_API_KEY") or env.get("GEMINI_API_KEY")
    if not key and path.is_file():
        key = path.read_text(encoding="utf-8-sig").strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError("Сохрани ключ Gemini одной строкой в приватный файл.")
    return key


class ComparisonModel:
    """An isolated client; records usage and raw replies, never keys or requests."""

    def __init__(self, client: AsyncOpenAI, model: str, provider: str) -> None:
        self.client = client
        self.model = model
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    async def _request(
        self,
        system_prompt: str,
        user_message: str,
        images: Sequence[ImageInput],
        *,
        observation: bool,
        json_response: bool = True,
    ) -> str:
        if len(self.calls) >= MAX_CASES * 2:
            raise ValueError("Исчерпан лимит запросов сравнения.")
        record: dict[str, Any] = {
            "stage": "observation" if observation else "verification",
            "system_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "system_chars": len(system_prompt),
            "user_chars": len(user_message),
            "image_sha256": [hashlib.sha256(im.data).hexdigest() for im in images],
        }
        self.calls.append(record)
        started = perf_counter()
        options: dict[str, Any] = (
            {"extra_body": {"thinking": {"type": "disabled"}}}
            if self.provider == "deepseek"
            else {"reasoning_effort": "low"}
        )
        if self.provider == "deepseek":
            options["temperature"] = 0
        if json_response:
            options["response_format"] = {"type": "json_object"}
        content: list[dict[str, Any]] = [{"type": "text", "text": user_message}]
        content.extend(
            {"type": "image_url", "image_url": {"url": im.data_url()}} for im in images
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=cast(
                    list[ChatCompletionMessageParam],
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                ),
                max_tokens=4096,
                **options,
            )
            usage = response.usage
            record["usage"] = usage.model_dump() if usage else None
            record["returned_model"] = response.model
            if not response.choices:
                raise ValueError("Модель не вернула вариант ответа.")
            choice = response.choices[0]
            record["finish_reason"] = choice.finish_reason
            record["reply"] = choice.message.content or ""
            if choice.finish_reason != "stop" or not record["reply"]:
                raise ValueError("Ответ пуст, заблокирован или обрезан.")
            return str(record["reply"])
        except OpenAIError as error:
            record["error_type"] = type(error).__name__
            if isinstance(error, APIStatusError):
                record["http_status"] = error.status_code
            # Raw exceptions may contain headers or user data; do not serialize.
            raise
        finally:
            record["seconds"] = round(perf_counter() - started, 3)

    async def analyze_visual_features(
        self,
        system_prompt: str,
        user_message: str,
        images: Sequence[ImageInput],
    ) -> str:
        return await self._request(
            system_prompt, user_message, images, observation=True
        )

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[ConversationTurn] = (),
        images: Sequence[ImageInput] = (),
        tool_names: Collection[str] | None = None,
        *,
        json_response: bool = False,
    ) -> str:
        if history or tool_names:
            raise ValueError("Сравнение выполняется без истории и инструментов.")
        return await self._request(
            system_prompt,
            user_message,
            images,
            observation=False,
            json_response=json_response,
        )


async def evaluate_case(model: ComparisonModel, case: Case) -> dict[str, Any]:
    first_call = len(model.calls)
    started = perf_counter()
    result: dict[str, Any] = {
        "case": case.name,
        "provider": model.provider,
        "requested_model": model.model,
        "checks_for_human_review": case.checks,
    }
    try:
        appearance = await AppearanceAnalyzer(model).analyze(
            case.images, case.caption, CONTENT_RULES
        )
        result["ok"] = True
        result["appearance"] = asdict(appearance)
    except (OpenAIError, ValueError, TimeoutError) as error:
        result["ok"] = False
        result["error_type"] = type(error).__name__
        if isinstance(error, APIStatusError):
            result["http_status"] = error.status_code
        # No automatic repair or another model call on invalid JSON / refusal.
    result["seconds"] = round(perf_counter() - started, 3)
    result["calls"] = model.calls[first_call:]
    return result


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--gemini-key-file", type=Path, default=Path(".secrets/gemini-api-key.txt")
    )
    parser.add_argument("--gemini-model", default=GEMINI_MODEL)
    parser.add_argument(
        "--provider", choices=("deepseek", "gemini", "both"), default="deepseek"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.manifest)
    env = dict(dotenv_values(args.env_file))
    deepseek_model = (
        env.get("LLM_MODEL") or env.get("DEEPSEEK_MODEL") or "deepseek-flash"
    )
    base_url = (
        env.get("LLM_BASE_URL")
        or env.get("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com"
    )
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "live": args.run,
        "providers": (
            ["gemini", "deepseek"] if args.provider == "both" else [args.provider]
        ),
        "pipeline": "AppearanceAnalyzer: observation + verification, 50s shared timeout",
        "request_limit_per_provider": 2 * len(cases),
        "retries": 0,
        "settings": {
            "deepseek_model": deepseek_model,
            "gemini_model": args.gemini_model,
            "deepseek_thinking": "disabled",
            "deepseek_temperature": 0,
            "gemini_reasoning_effort": "low",
            "max_tokens": 4096,
        },
        "cases": [
            {
                "name": case.name,
                "caption": case.caption,
                "image_sha256": [
                    hashlib.sha256(im.data).hexdigest() for im in case.images
                ],
                "checks_for_human_review": case.checks,
            }
            for case in cases
        ],
        "results": [],
    }
    if not args.run:
        save_report(args.output, report)
        print(f"Dry run: {len(cases)} cases; at most {2 * len(cases)} calls/provider.")
        return
    deepseek_key = env.get("LLM_API_KEY") or env.get("DEEPSEEK_API_KEY")
    if "deepseek" in report["providers"] and not deepseek_key:
        raise ValueError("Ключ DeepSeek не настроен.")
    if (
        "deepseek" in report["providers"]
        and env.get("LLM_PROVIDER", "deepseek") != "deepseek"
    ):
        raise ValueError("Для сравнения требуется настроенный DeepSeek.")
    gemini_key = (
        read_gemini_key(args.gemini_key_file, env)
        if "gemini" in report["providers"]
        else None
    )
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    save_report(args.output, report)
    async with AsyncExitStack() as stack:
        models = []
        for provider in report["providers"]:
            client = await stack.enter_async_context(
                AsyncOpenAI(
                    api_key=gemini_key if provider == "gemini" else deepseek_key,
                    base_url=GEMINI_BASE_URL if provider == "gemini" else base_url,
                    timeout=35,
                    max_retries=0,
                )
            )
            models.append(
                ComparisonModel(
                    client,
                    args.gemini_model if provider == "gemini" else deepseek_model,
                    provider,
                )
            )
        for case in cases:
            for model in models:
                result = await evaluate_case(model, case)
                report["results"].append(result)
                save_report(args.output, report)
                print(
                    f"{case.name}/{model.provider}: ok={result['ok']}; "
                    f"{len(result['calls'])} calls; {result['seconds']}s",
                    flush=True,
                )
                if result.get("http_status") in {401, 403, 404, 429}:
                    print("Stopped: resolve access / model / quota before another run.")
                    return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (OSError, ValueError) as error:
        # Only our own validation messages, not remote SDK exceptions / key data.
        raise SystemExit(f"Cannot start comparison ({type(error).__name__}).") from None
