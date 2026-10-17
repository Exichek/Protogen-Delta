"""Проверка предложения diff по снимку исходника, без записи и исполнения."""

import ast
import re
from collections.abc import Sequence

from protogen_delta.services.source_access import SourceFile

_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?")
_FENCE = re.compile(
    r"^[ ]{0,3}```(?:diff|patch)[ \t]*\n(?P<patch>.*?)(?:(?P<closing>^[ ]{0,3}```[ \t]*(?:\n|$))|\Z)",
    re.M | re.S,
)


def preview_source_diff(patch: str, files: Sequence[SourceFile]) -> str:
    """Вернуть сведения о сверке; исходный и полученный текст остаются в памяти."""
    if len(patch) > 12000 or len(patch.splitlines()) > 200:
        raise ValueError("Предложение слишком большое")
    lines = patch.splitlines()
    if lines and lines[0].startswith("diff --git "):
        lines.pop(0)
        if lines and lines[0].startswith("index "):
            lines.pop(0)
    if (
        len(lines) < 3
        or not lines[0].startswith("--- ")
        or not lines[1].startswith("+++ ")
    ):
        raise ValueError("Нужен unified diff одного существующего Python-файла")
    old_name, new_name = lines[0][4:], lines[1][4:]
    old_name = old_name[2:] if old_name.startswith("a/") else old_name
    new_name = new_name[2:] if new_name.startswith("b/") else new_name
    if old_name != new_name:
        raise ValueError("Создание, удаление и переименование файлов не поддерживаются")
    name = old_name.removeprefix("src/protogen_delta/")
    source = next((item for item in files if item.name == name), None)
    if source is None:
        raise ValueError("Файл отсутствует в разрешённом снимке исходников")
    original = source.text.splitlines()
    output: list[str] = []
    cursor, index, hunks = 0, 2, 0
    while index < len(lines):
        header = _HUNK.fullmatch(lines[index])
        if header is None:
            raise ValueError("Некорректный заголовок или несколько файлов в одном diff")
        old_at, old_count, new_at, new_count = (
            int(header[1]),
            int(header[2] or 1),
            int(header[3]),
            int(header[4] or 1),
        )
        start = old_at - 1 if old_count else old_at
        if not cursor <= start <= len(original) or start + old_count > len(original):
            raise ValueError("Номера исходных строк не совпадают со снимком")
        output.extend(original[cursor:start])
        if new_at != len(output) + bool(new_count):
            raise ValueError("Номера новых строк не соответствуют предложению")
        index += 1
        consumed = produced = 0
        changed = False
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if line == r"\ No newline at end of file":
                index += 1
                continue
            if not line or line[0] not in " +-":
                raise ValueError("Некорректная строка diff")
            marker, value = line[0], line[1:]
            if "[секрет" in value.casefold():
                raise ValueError("Нельзя предлагать изменения скрытых секретных строк")
            if marker in " -":
                if (
                    start + consumed >= len(original)
                    or original[start + consumed] != value
                ):
                    raise ValueError(
                        "Контекст или удаляемая строка не совпадают со снимком"
                    )
                consumed += 1
            if marker in " +":
                output.append(value)
                produced += 1
            changed |= marker in "+-"
            index += 1
        if (consumed, produced) != (old_count, new_count) or not changed:
            raise ValueError("Размеры фрагмента не совпадают или изменений нет")
        cursor = start + consumed
        hunks += 1
    if not hunks:
        raise ValueError("В diff нет изменений")
    output.extend(original[cursor:])
    proposed = "\n".join(output)
    try:
        ast.parse(source.text)
    except SyntaxError, ValueError, RecursionError:
        syntax = "AST не проверен: маскирование изменило структуру снимка."
    else:
        try:
            ast.parse(proposed)
        except SyntaxError, ValueError, RecursionError:
            raise ValueError("Предложенный текст не прошёл разбор AST") from None
        syntax = "Разбор AST предложения прошёл; импорт и тесты не запускались."
    return (
        f"Предложение для {name}; SHA256 снимка {source.digest}. "
        f"Контекст строк совпал с доступным снимком. {syntax} "
        "Код не изменялся. Это предложение для ручного рассмотрения; "
        "сверка снимка не гарантирует применимость к Git или правильность поведения."
    )


def annotate_source_diffs(text: str, files: Sequence[SourceFile]) -> str:
    """Непроверенные/обрезанные блоки не выдаются за готовую правку."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    notes: list[str] = []

    def replace(match: re.Match[str]) -> str:
        if match["closing"] is None:
            return "Предложение diff обрезано и не показано. Код не изменялся.\n"
        if len(notes) >= 3:
            return "Дополнительное предложение diff не показано: предел трёх.\n"
        try:
            notice = preview_source_diff(match["patch"], files)
        except ValueError as error:
            notes.append(
                f"Предложение diff не прошло сверку: {error}. Код не изменялся."
            )
            return "[Diff не показан: сверка не пройдена.]\n"
        notes.append(notice)
        return match[0]

    checked = _FENCE.sub(replace, text)
    return checked + ("\n\n" + "\n\n".join(notes) if notes else "")


def short_review_notice(report: str) -> str:
    """Автоматическое короткое уведомление не содержит обрезанного diff."""
    if len(report) <= 3500:
        return report
    prose = _FENCE.sub("[Предложение diff доступно в полном отчёте.]\n", report)
    excerpt = prose[:3200]
    if len(prose) > len(excerpt):
        boundary = excerpt.rfind(" ")
        if boundary > 0:
            excerpt = excerpt[:boundary]
        excerpt = excerpt.rstrip() + "…"
    return (
        "Фрагмент сохранённого отчёта:\n\n"
        + excerpt
        + "\n\nПолный отчёт: /code_review\nНовый обзор: /code_review new"
    )
