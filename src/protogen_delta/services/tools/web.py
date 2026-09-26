"""Инструменты поиска в интернете и чтения публичных страниц."""

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

from protogen_delta.services.tools.http import fetch_provider, fetch_public_page

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}


class _TextExtractor(HTMLParser):
    """Извлекать видимый текст HTML без script, style и служебных элементов."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self.title = ""
        self._in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg", "canvas", "template"}:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "canvas", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title:
            self.title = f"{self.title} {clean}".strip()
        else:
            self.parts.append(clean)


async def web_search(
    args: dict[str, str], *, api_key: str, proxy_url: str | None = None
) -> dict[str, Any]:
    """Найти свежие публичные страницы через Brave Search API."""
    query = args["query"].strip()
    if len(query) < 2:
        raise ValueError("Поисковый запрос слишком короткий")
    try:
        count = int(args.get("count", "5"))
    except ValueError as error:
        raise ValueError("Количество результатов должно быть числом") from error
    if not 1 <= count <= 10:
        raise ValueError("Количество результатов должно быть от 1 до 10")
    language = args.get("language", "ru").lower()
    if language not in {"ru", "en"}:
        raise ValueError("Поддерживаются языки ru и en")
    parameters = {
        "q": query,
        "count": str(count),
        "search_lang": language,
        "safesearch": "moderate",
        "spellcheck": "1",
    }
    freshness = args.get("freshness", "").lower()
    if freshness:
        if freshness not in _FRESHNESS:
            raise ValueError("Некорректный период свежести")
        parameters["freshness"] = _FRESHNESS[freshness]
    payload = json.loads(
        await fetch_provider(
            _BRAVE_SEARCH_URL,
            parameters,
            proxy_url=proxy_url,
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
    )
    items = payload.get("web", {}).get("results", [])
    results = []
    for item in items[:count]:
        url = item.get("url")
        title = item.get("title")
        if not isinstance(url, str) or not isinstance(title, str):
            continue
        results.append(
            {
                "title": title[:300],
                "url": url,
                "description": str(item.get("description", ""))[:1000],
                "published": item.get("page_age") or item.get("age"),
            }
        )
    return {
        "query": query,
        "results": results,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": "Brave Search API",
    }


async def fetch_web_page(
    args: dict[str, str], *, proxy_url: str | None = None
) -> dict[str, Any]:
    """Прочитать ограниченный объём текста публичной страницы."""
    raw, final_url, content_type = await fetch_public_page(
        args["url"], proxy_url=proxy_url
    )
    charset_match = re.search(
        rb"charset\s*=\s*['\"]?([a-zA-Z0-9._-]+)", raw[:8192], re.IGNORECASE
    )
    charset = (
        charset_match.group(1).decode("ascii", "ignore") if charset_match else "utf-8"
    )
    try:
        decoded = raw.decode(charset, "replace")
    except LookupError:
        decoded = raw.decode("utf-8", "replace")
    if content_type == "text/plain":
        title = ""
        text = " ".join(decoded.split())
    else:
        extractor = _TextExtractor()
        extractor.feed(decoded)
        title = extractor.title
        text = "\n".join(extractor.parts)
    return {
        "url": final_url,
        "title": title[:300],
        "text": text[:12000],
        "truncated": len(text) > 12000,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
