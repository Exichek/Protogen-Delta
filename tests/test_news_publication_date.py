"""Дата посещения и дата изменения не заменяют объявленную дату публикации."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import protogen_delta.services.tools.web as web_module


@pytest.mark.parametrize("attribute", ["property", "name", "itemprop"])
@pytest.mark.parametrize(
    "field",
    [
        "article:published_time",
        "datePublished",
        "pubdate",
        "article:modified_time",
        "dateModified",
        "date",
    ],
)
def test_publication_metadata_is_returned_separately(
    monkeypatch: pytest.MonkeyPatch, field: str, attribute: str
) -> None:
    html = (
        "<html><head><title>News</title><meta "
        + attribute
        + '="'
        + field
        + '" content="2026-10-23T10:00:00Z"></head><body>Official announcement</body></html>'
    ).encode()
    monkeypatch.setattr(
        web_module,
        "fetch_public_page",
        AsyncMock(return_value=(html, "https://news.xbox.com/news", "text/html")),
    )
    result = asyncio.run(
        web_module.fetch_web_page({"url": "https://news.xbox.com/news"})
    )
    if field in {"article:published_time", "datePublished", "pubdate"}:
        assert result["published"] == "2026-10-23T10:00:00Z"
    else:
        assert "published" not in result
    assert result["text"] == "Official announcement"


def test_news_can_be_disabled_without_disabling_discussions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_settings import _disable_dotenv

    from protogen_delta.config.settings import load_settings

    _disable_dotenv(monkeypatch)
    monkeypatch.setenv("TELEGRAM_TOKEN", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setenv("ART_CHAT_ID", "-1")
    monkeypatch.setenv("PROACTIVE_NEWS_ENABLED", "false")
    settings = load_settings()
    assert not settings.proactive_news_enabled
    assert settings.proactive_lively_enabled
