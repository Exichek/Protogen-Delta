"""Несколько явно предложенных ссылок и подтверждённые статусы чтения."""

import json
import re


def review_links(message: str) -> tuple[str, ...]:
    # Содержимое вложения, цитаты и код не создают самостоятельных запросов к сети.
    visible = message.split("<document_content>", 1)[0]
    visible = re.sub(
        r'```[\s\S]*?```|`[^`]*`|«[^»]*»|“[^”]*”|"[^"\n]*"|(?m:^\s*>[^\n]*)',
        " ",
        visible,
    )
    if re.search(
        r"\bне\s+(?:открывай|читай|переходи|проверяй|смотри|ищи)\b|\bбез\s+(?:просмотра|перехода|открытия)\b",
        visible,
        re.I,
    ):
        return ()
    urls = []
    for match in re.finditer(r"https?://[^\s<>]+", visible, re.I):
        url = match.group().rstrip(".,;!?")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        url = url.rstrip("]}")
        if url not in urls:
            urls.append(url)
    remaining = re.sub(r"https?://[^\s<>]+", "", visible).strip(" \t\r\n,;!?.<>()[]")
    requested = not remaining or re.search(
        r"\b(?:посмотри|прочитай|изучи|разбери|проверь|сравни|глянь|просмотри|review|compare|read)\b",
        visible,
        re.I,
    )
    return tuple(urls) if len(urls) >= 2 and requested else ()


def compact_page_result(raw: str) -> tuple[str, str]:
    data = json.loads(raw)
    if not data.get("ok"):
        return raw, "не прочитана"
    payload = data.get("data", {})
    if isinstance(payload, dict) and payload.get("available") is False:
        return raw[:4000], "не прочитана"
    if not isinstance(payload, dict):
        return raw[:4000], "результат получен"
    compact: dict[str, str | bool] = {}
    truncated = bool(payload.get("truncated"))
    for key in (
        "url",
        "title",
        "text",
        "checked_at",
        "content_scope",
        "description",
        "transcript",
        "transcript_language",
        "transcript_status",
        "author",
        "repository",
        "revision",
        "notice",
    ):
        if key in payload:
            value = payload[key]
            if isinstance(value, str):
                maximum = 2000 if key in {"text", "transcript"} else 300
                truncated |= len(value) > maximum
                compact[key] = value[:maximum]
    if type(payload.get("transcript_automatic")) is bool:
        compact["transcript_automatic"] = payload["transcript_automatic"]
    if isinstance(payload.get("entries"), list):
        compact["entries"] = json.dumps(payload["entries"][:20], ensure_ascii=False)[
            :2000
        ]
    compact["truncated"] = truncated
    scope = payload.get("content_scope")
    status = (
        "прочитаны сведения и каталог репозитория, не весь код"
        if scope == "repository_overview"
        else (
            "получены только метаданные ролика"
            if scope == "metadata_only"
            else (
                "получены метаданные и субтитры, не видеоряд"
                if scope == "metadata_and_transcript"
                else (
                    "получен текст страницы"
                    if payload.get("text")
                    else "получена страница без читаемого текста"
                )
            )
        )
    )
    if scope == "repository_file":
        status = "прочитан фрагмент указанного файла"
    if truncated:
        status += " (частично)"
    return json.dumps({"ok": True, "data": compact}, ensure_ascii=False), status
