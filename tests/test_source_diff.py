"""Предложения проверяются по разрешённому снимку, без применения изменений."""

import ast
import hashlib

import pytest

from protogen_delta.services.source_access import SourceFile
from protogen_delta.services.source_diff import (
    annotate_source_diffs,
    preview_source_diff,
    short_review_notice,
)

SOURCE = "value = 42\n# context\nnext_value = 3"
FILE = SourceFile(
    "services/sample.py", SOURCE, hashlib.sha256(SOURCE.encode()).hexdigest()
)
PATCH = (
    "--- a/src/protogen_delta/services/sample.py\n"
    "+++ b/src/protogen_delta/services/sample.py\n"
    "@@ -1,2 +1,2 @@\n-value = 42\n+value = 43\n # context\n"
)


def test_valid_diff_reports_ast_check_without_mutating_source() -> None:
    notice = preview_source_diff(PATCH, [FILE])
    assert FILE.digest in notice and "Контекст строк совпал" in notice
    assert "AST предложения прошёл" in notice and "Код не изменялся" in notice
    assert FILE.text == SOURCE
    text = annotate_source_diffs("Предложение:\n```diff\n" + PATCH + "```\n", [FILE])
    assert PATCH in text and notice in text
    windows = "  ```diff\r\n" + PATCH.replace("\n", "\r\n") + "  ```\r\n"
    assert "Контекст строк совпал" in annotate_source_diffs(windows, [FILE])


@pytest.mark.parametrize(
    "patch",
    [
        PATCH.replace("value = 42", "value = 41"),
        PATCH.replace("@@ -1,2 +1,2 @@", "@@ -1,4 +1,2 @@"),
        PATCH.replace("@@ -1,2 +1,2 @@", "@@ -1,2 +2,2 @@"),
        PATCH.replace("+value = 43", "+value = ("),
        PATCH.replace("+++ b/src/protogen_delta/services/sample.py", "+++ b/new.py"),
        PATCH.replace("services/sample.py", "../.env"),
        PATCH.replace(" # context", "garbage"),
        PATCH + "--- a/main.py\n+++ b/main.py\n",
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+value = 1\n",
        "--- a/services/sample.py\n+++ b/services/sample.py\n@@ -0,0 +1,0 @@\n",
        "x" * 12001,
        "\n" * 201,
        "",
    ],
)
def test_invalid_diff_is_not_presented_as_ready(patch: str) -> None:
    with pytest.raises(ValueError):
        preview_source_diff(patch, [FILE])
    rendered = annotate_source_diffs("```diff\n" + patch + "\n```\n", [FILE])
    assert "сверка не пройдена" in rendered and "не прошло сверку" in rendered


def test_multiple_hunks_and_insert_at_beginning() -> None:
    patch = (
        "diff --git a/services/sample.py b/services/sample.py\nindex 111..222 100644\n"
        "--- a/services/sample.py\n+++ b/services/sample.py\n"
        "@@ -0,0 +1 @@\n+# Added\n"
        "@@ -3 +4 @@\n-next_value = 3\n+next_value = 4\n"
        "\\ No newline at end of file\n"
    )
    assert "AST предложения прошёл" in preview_source_diff(patch, [FILE])


def test_secret_lines_cannot_be_changed_and_masked_structure_is_not_source_error() -> (
    None
):
    hidden = "# [секретный литерал скрыт]"
    with pytest.raises(ValueError, match="секретных"):
        preview_source_diff(PATCH.replace("value = 42", hidden), [FILE])
    masked = SourceFile("services/sample.py", "value = 42\ncall(\n# hidden", "snapshot")
    assert "AST не проверен" in preview_source_diff(
        PATCH.replace("@@ -1,2 +1,2 @@", "@@ -1 +1 @@").replace(" # context\n", ""),
        [masked],
    )


def test_truncated_diff_and_more_than_three_proposals_are_labelled() -> None:
    assert "обрезано" in annotate_source_diffs("Текст\n```diff\n" + PATCH, [FILE])
    assert "обрезано" not in annotate_source_diffs("Обычное замечание", [FILE])
    text = annotate_source_diffs(("```diff\n" + PATCH + "```\n") * 4, [FILE])
    assert text.count("```diff") == 3 and "предел трёх" in text


def test_ast_only_parse_never_executes_proposed_code() -> None:
    patch = PATCH.replace("+value = 43", "+raise RuntimeError('Do not execute')")
    assert "AST предложения прошёл" in preview_source_diff(patch, [FILE])
    assert ast.parse("raise RuntimeError('Do not execute')")


def test_automatic_notice_never_shows_a_partial_diff() -> None:
    report = "Описание. " * 310 + "\n```diff\n" + PATCH + "```\n" + "Проверки. " * 40
    assert len(report) > 3500
    notice = short_review_notice(report)
    assert len(notice) <= 3500 and "```diff" not in notice
    assert "Полный отчёт: /code_review" in notice
    assert "diff доступно в полном отчёте" in notice
    assert short_review_notice("Короткий отчёт") == "Короткий отчёт"
