"""Проверять происхождение веб-ответа по реально полученным страницам."""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from protogen_delta.core.tool_markup import contains_unquoted_tool_markup

UNVERIFIED_WEB_REPLY = (
    "Не удалось достаточно надёжно подтвердить ответ по прочитанным источникам. "
    "Не буду выдавать предположение за проверенный факт."
)


def _normalized(text: str) -> str:
    # Регистр сохраняем: mW и MW — разные единицы, не допустимая правка цитаты.
    return " ".join(unicodedata.normalize("NFKC", text).split())


@dataclass
class WebEvidence:
    """Держать материалы только на время текущего ответа, без записи на диск."""

    used: bool = False
    searched: bool = False
    pages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def observe(self, name: str, arguments: str, result: str) -> None:
        if name not in {"web_search", "fetch_web_page"}:
            return
        self.used = True
        try:
            payload = json.loads(result)
            args = json.loads(arguments)
        except ValueError, TypeError:
            return
        if not isinstance(payload, dict) or not payload.get("ok"):
            return
        data = payload.get("data")
        if not isinstance(data, dict) or data.get("available") is False:
            return
        if name == "web_search":
            self.searched = bool(data.get("results")) or self.searched
            return
        text = data.get("transcript") or data.get("text")
        url = args.get("url") if isinstance(args, dict) else None
        if not isinstance(url, str) or not isinstance(text, str) or not text.strip():
            return
        if len(self.pages) >= 6 or data.get("content_scope") == "metadata_only":
            return
        self.pages[url] = {
            "url": url,
            "title": str(data.get("title", ""))[:300],
            "text": (
                text
                if len(text) <= 8000
                else text[:5000] + "\n[…пропуск…]\n" + text[-3000:]
            ),
            "checked_at": str(data.get("checked_at", ""))[:80],
            "content_scope": str(data.get("content_scope", "page_text"))[:80],
            "notice": str(data.get("notice", ""))[:500],
            "transcript_automatic": bool(data.get("transcript_automatic")),
            "truncated": bool(data.get("truncated")) or len(text) > 8000,
        }

    def request(self, question: str, draft: str) -> str:
        return json.dumps(
            {
                "checked_now_utc": datetime.now(timezone.utc).isoformat(),
                "question": question[:4000],
                "draft": draft[:10000],
                "pages": list(self.pages.values()),
            },
            ensure_ascii=False,
        )

    def accept(self, raw: str) -> str:
        """Не принимать выдуманную ссылку или цитату, которой нет в прочитанном тексте."""
        try:
            result = json.loads(raw)
        except ValueError, TypeError:
            return UNVERIFIED_WEB_REPLY
        if not isinstance(result, dict):
            return UNVERIFIED_WEB_REPLY
        answer, evidence = result.get("answer"), result.get("evidence")
        if (
            not isinstance(answer, str)
            or not 1 <= len(answer.strip()) <= 10000
            or not isinstance(evidence, list)
            or not 1 <= len(evidence) <= 6
        ):
            return UNVERIFIED_WEB_REPLY
        if contains_unquoted_tool_markup(answer):
            return UNVERIFIED_WEB_REPLY
        supported: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict):
                return UNVERIFIED_WEB_REPLY
            url, quote = item.get("url"), item.get("quote")
            if (
                not isinstance(url, str)
                or url not in self.pages
                or not isinstance(quote, str)
                or not 20 <= len(quote.strip()) <= 500
                or _normalized(quote) not in _normalized(self.pages[url]["text"])
            ):
                return UNVERIFIED_WEB_REPLY
            supported.add(url)
        urls = set()
        for value in re.findall(r"https?://[^\s<>]+", answer):
            value = value.rstrip(".,;!?]}")
            while value.endswith(")") and value.count(")") > value.count("("):
                value = value[:-1]
            urls.add(value)
        if not urls or not urls <= supported:
            return UNVERIFIED_WEB_REPLY
        return answer.strip()
