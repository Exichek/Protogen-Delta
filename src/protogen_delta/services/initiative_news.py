"""Подбирать датированную новость из первоисточника, без выдачи сниппета за факт."""

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from aiohttp import ClientError

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.initiative_topics import TOPIC_LABELS
from protogen_delta.repositories.web_api_usage import WebApiLimitError
from protogen_delta.services.deepseek import DeepSeekService
from protogen_delta.services.web_search import WebSearchService
from protogen_delta.services.web_verification import UNVERIFIED_WEB_REPLY, WebEvidence

MAX_NEWS_CHARS = 700
_QUERIES = {
    "games": "games news (site:news.xbox.com OR site:blog.playstation.com OR site:callofduty.com)",
    "ai": "AI announcements (site:openai.com OR site:anthropic.com OR site:blog.google)",
    "tech": "technology news (site:apple.com/newsroom OR site:blogs.nvidia.com OR site:blogs.windows.com)",
}
_HOSTS = {
    "games": ("news.xbox.com", "blog.playstation.com", "callofduty.com"),
    "ai": ("openai.com", "anthropic.com", "blog.google"),
    "tech": ("apple.com", "blogs.nvidia.com", "blogs.windows.com"),
}


def news_url(value: object) -> str:
    """Сравнивать одну публикацию без рекламных меток и фрагмента."""
    if not isinstance(value, str) or len(value) > 2048:
        return ""
    try:
        parts = urlsplit(value)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.port not in {None, 443}
        ):
            return ""
        query = urlencode(
            sorted(
                (key, item)
                for key, item in parse_qsl(parts.query, keep_blank_values=True)
                if not key.casefold().startswith("utm_")
                and key.casefold() not in {"fbclid", "gclid"}
            )
        )
        return urlunsplit(
            (
                "https",
                parts.hostname.casefold().rstrip("."),
                parts.path or "/",
                query,
                "",
            )
        )
    except ValueError:
        return ""


def _published(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) > 80:
        return None
    try:
        stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
    except ValueError:
        return None


def _primary(url: str, topic: str) -> bool:
    host = urlsplit(url).hostname or ""
    return any(
        host == allowed or host.endswith("." + allowed) for allowed in _HOSTS[topic]
    )


@dataclass(frozen=True, slots=True)
class NewsMessage:
    text: str
    url: str
    topic: str


class InitiativeNewsService:
    """Один поиск, до двух чтений и один запрос модели; общие кеш и квоты провайдера."""

    def __init__(self, web: WebSearchService) -> None:
        self._web = web

    @property
    def enabled(self) -> bool:
        return self._web.enabled

    async def generate(
        self,
        *,
        user_id: int,
        at: float,
        topics: Sequence[str],
        excluded_urls: Sequence[str],
        model: DeepSeekService,
        system_prompt: str,
        alive: Callable[[], Awaitable[bool]],
    ) -> NewsMessage | None:
        if not topics or not self.enabled or not await alive():
            return None
        if any(topic not in TOPIC_LABELS for topic in topics):
            raise ValueError("Неизвестная тема новости")
        now = datetime.fromtimestamp(at, timezone.utc)
        # Идентификатор используется только локально, запрос в поиск содержит общую тему.
        day_msk = now.astimezone(timezone(timedelta(hours=3))).date()
        topic = topics[(user_id + day_msk.toordinal()) % len(topics)]
        search = await self._web.search(
            {
                "query": _QUERIES[topic],
                "language": "en",
                "freshness": "day",
                "count": "5",
            }
        )
        seen = {news_url(url) for url in excluded_urls}
        attempted = set()
        reads = 0
        rows = search.get("results", [])
        if not isinstance(rows, list):
            return None
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            url = news_url(row.get("url"))
            indexed_date = _published(row.get("published"))
            if (
                not url
                or url in seen
                or url in attempted
                or not _primary(url, topic)
                or indexed_date is not None
                and not now - timedelta(days=3) <= indexed_date <= now
            ):
                continue
            if reads >= 2 or not await alive():
                break
            attempted.add(url)
            reads += 1
            try:
                page = await self._web.read({"url": url})
            except WebApiLimitError:
                raise
            except ClientError, TimeoutError, ValueError, RuntimeError:
                continue
            final_url = news_url(page.get("url", url))
            published = _published(page.get("published"))
            if (
                not final_url
                or final_url in seen
                or not _primary(final_url, topic)
                or page.get("final_url_verified") is False
                or published is None
                or not now - timedelta(days=3) <= published <= now
            ):
                continue
            evidence = WebEvidence(used=True, searched=True)
            evidence.observe(
                "fetch_web_page",
                json.dumps({"url": final_url}),
                json.dumps({"ok": True, "data": page}),
            )
            if not evidence.pages:
                continue
            evidence.pages[final_url]["published"] = published.isoformat()
            evidence.pages[final_url]["publication_date_source"] = "page_metadata"
            if not await alive():
                return None
            request = evidence.request(
                "Поделись одной новостью по теме «"
                + TOPIC_LABELS[topic]
                + "» из этой страницы. "
                "Сегодня "
                + now.date().isoformat()
                + ". Дата публикации "
                + published.date().isoformat()
                + ". Коротко скажи, что объявил источник и чем это можно обсудить. "
                "Публикация не доказывает, что само событие произошло сегодня.",
                "",
            )
            raw = await model.chat(
                system_prompt=system_prompt
                + "\n\n"
                + load_prompt("initiative_news")
                + "\n\n"
                + load_prompt("web_verification"),
                user_message=request,
                json_response=True,
                tool_names=(),
            )
            text = evidence.accept(raw)
            link = "[источник](" + final_url + ")"
            if text == UNVERIFIED_WEB_REPLY or link not in text:
                return None
            body = " ".join(text.replace(link, "").split()).strip()
            if not body or len(body) > 420:
                return None
            message = (
                body
                + "\n\nИсточник от "
                + published.strftime("%d.%m.%Y")
                + ": "
                + final_url
            )
            if len(message) > MAX_NEWS_CHARS:
                return None
            return NewsMessage(message, final_url, topic)
        return None
