"""Найти длинные совпадения с каталогом примеров в сохранённом отчёте без API."""

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from protogen_delta.config.prompt_loader import PROMPTS_DIR

EXAMPLE_FILES = ("voice_examples.txt", "scene_examples.txt")


def words(text: str) -> list[str]:
    """Сравнивать слова без регистра и пунктуации; смысл и перефразирование не оценивать."""
    return re.findall(r"[^\W_]+", text.casefold())


def audit(
    document: dict[str, Any], examples: dict[str, str], *, minimum_words: int = 8
) -> dict[str, Any]:
    """Сохранить исходные оценки; совпадение — сигнал для чтения, не вердикт."""
    if type(minimum_words) is not int or not 4 <= minimum_words <= 100:
        raise ValueError("Порог должен быть целым числом от 4 до 100 слов")
    report = document.get("report", document)
    if not isinstance(report, dict) or report.get("mode") not in {"model", "preview"}:
        raise ValueError("Нужен отчёт evaluate_behavior или compare_example_contracts")
    if set(examples) != set(EXAMPLE_FILES):
        raise ValueError("Нужны оба файла исходных образцов голоса и сцены")
    fingerprints = {
        name: hashlib.sha256(text.strip().encode()).hexdigest()
        for name, text in examples.items()
    }
    samples = [
        (name, number, words(line.partition("→")[2]))
        for name, text in examples.items()
        for number, line in enumerate(text.splitlines(), 1)
        if "→" in line
    ]
    rows = report.get("rows")
    if rows is None:
        results = report.get("results")
        if not isinstance(results, list):
            raise ValueError("В отчёте нет списка результатов")
        rows = [
            {
                "result": result,
                "prompts": report.get("prompts", {}),
                "contract": report.get("example_contract", "unspecified"),
            }
            for result in results
        ]
    if not isinstance(rows, list):
        raise ValueError("Неверный список парных результатов")
    reviewed = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("result"), dict):
            raise ValueError("Неверная строка оценочного отчёта")
        result = row["result"]
        if (
            not isinstance(result.get("case"), str)
            or type(result.get("step")) is not int
            or result["step"] < 1
        ):
            raise ValueError("Нужны ID сценария и положительный номер хода")
        record = {
            "case": result.get("case"),
            "step": result.get("step"),
            "contract": row.get("contract"),
            "repeat": row.get("repeat"),
            "automatic_checks": result.get("checks"),
            "quality_review": result.get("quality_review"),
            "status": "preview_not_assessed",
            "matches": [],
        }
        if report["mode"] == "preview":
            reviewed.append(record)
            continue
        prompts = row.get("prompts")
        if not isinstance(prompts, dict) or not isinstance(prompts.get("files"), dict):
            raise ValueError("В отчёте нет отпечатков исходных промптов")
        recorded = prompts["files"]
        if any(recorded.get(name) != digest for name, digest in fingerprints.items()):
            raise ValueError("Версия образцов не совпадает с сохранённым отчётом")
        reply, message = result.get("reply"), result.get("message")
        if not isinstance(reply, str) or not isinstance(message, str):
            raise ValueError("В реальном результате нужны строки ответа и сообщения")
        reply_words, message_words = words(reply), words(message)
        matches = []
        for name, line, sample_words in samples:
            match = SequenceMatcher(
                None, reply_words, sample_words, autojunk=False
            ).find_longest_match()
            if match.size < minimum_words:
                continue
            start, end = match.a, match.a + match.size
            sequence = reply_words[start:end]
            in_message = (
                SequenceMatcher(None, sequence, message_words, autojunk=False)
                .find_longest_match()
                .size
                == match.size
            )
            matches.append(
                {
                    "source": name,
                    "line": line,
                    "words": match.size,
                    "reply_tokens": [match.a, match.a + match.size],
                    "also_in_current_message": in_message,
                }
            )
        record.update(
            status="signals_only",
            matches=matches,
            reply_sha256=hashlib.sha256(reply.encode()).hexdigest(),
        )
        reviewed.append(record)
    return {
        "audit": "example_catalog_overlap",
        "mode": "offline",
        "minimum_words": minimum_words,
        "example_fingerprints": fingerprints,
        "source_complete": report.get("complete"),
        "source_budget_exceeded": report.get("budget_exceeded"),
        "limitations": [
            "Совпадение с каталогом не доказывает копирование или ошибку: нужный модуль мог не входить в запрос.",
            "Отмечается совпадение и в текущем сообщении; цитирование образца может быть уместным.",
            "Отсутствие совпадения не доказывает достоверность, разнообразие или сохранение характера.",
            "Прежние проверки и ручные оценки не меняются; автоматического выбора варианта нет.",
        ],
        "rows": reviewed,
        "summary": {
            "responses": sum(r["status"] == "signals_only" for r in reviewed),
            "responses_with_matches": sum(bool(r["matches"]) for r in reviewed),
            "preview_skipped": sum(
                r["status"] == "preview_not_assessed" for r in reviewed
            ),
        },
    }


def main() -> None:
    """Читать только локальный отчёт и образцы; ключи и сеть не используются."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, default=PROMPTS_DIR)
    parser.add_argument("--minimum-words", type=int, default=8)
    args = parser.parse_args()
    try:
        if args.report.resolve() == args.output.resolve():
            raise ValueError("Выходной файл не должен перезаписывать исходный отчёт")
        raw = args.report.read_bytes()
        document = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError("Ожидается JSON-объект отчёта")
        examples = {
            name: (args.examples_dir / name).read_text("utf-8")
            for name in EXAMPLE_FILES
        }
        result = audit(document, examples, minimum_words=args.minimum_words)
        result["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
