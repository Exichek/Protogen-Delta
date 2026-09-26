"""Контракты внешних инструментов без сети и секретов."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest

import protogen_delta.services.tools.http as http_module
import protogen_delta.services.tools.live_data as live_data_module
import protogen_delta.services.tools.web as web_module
from protogen_delta.services.tools import (
    Tool,
    ToolExecutor,
    ToolRegistry,
    default_registry,
    fetch_web_page,
    get_current_time,
    get_exchange_rate,
    get_weather,
    web_search,
)


def test_registry_and_time() -> None:
    registry = default_registry()
    assert len(registry.schemas()) == 4
    search_registry = default_registry(brave_search_api_key="secret")
    assert len(search_registry.schemas()) == 5
    assert "web_search" in search_registry.tools
    assert all(
        x["function"]["parameters"]["additionalProperties"] is False
        for x in registry.schemas()
    )
    tool = next(iter(registry.tools.values()))
    with pytest.raises(ValueError):
        ToolRegistry([tool, tool])
    result = asyncio.run(get_current_time({}))
    assert result["datetime"].endswith("+03:00")
    assert asyncio.run(get_current_time({"timezone": "UTC"}))["datetime"].endswith(
        "+00:00"
    )
    with pytest.raises(ValueError):
        asyncio.run(get_current_time({"timezone": "unknown"}))


@pytest.mark.parametrize(
    "name,args",
    [
        ("shell", "{}"),
        ("get_current_time", "[]"),
        ("get_current_time", "{"),
        ("get_current_time", '{"extra": "x"}'),
        ("get_current_time", '{"timezone": 3}'),
        ("get_weather", "{}"),
        ("get_current_time", "x" * 4097),
        ("get_weather", json.dumps({"city": "x" * 201})),
    ],
)
def test_invalid_calls_are_not_executed(name: str, args: str) -> None:
    result = asyncio.run(ToolExecutor(default_registry()).execute(name, args))
    assert json.loads(result)["ok"] is False


def test_executor_errors_limits_and_cancellation() -> None:
    handler = AsyncMock(return_value={"result": "ok"})
    executor = ToolExecutor(
        ToolRegistry([Tool("test", "test", {}, (), handler)]), timeout=0.01
    )
    assert json.loads(asyncio.run(executor.execute("test", "{}")))["ok"]
    handler.return_value = {"text": "a" * 16001}
    assert not json.loads(asyncio.run(executor.execute("test", "{}")))["ok"]
    handler.side_effect = OSError("secret-url")
    result = asyncio.run(executor.execute("test", "{}"))
    assert "source_unavailable" in result and "secret-url" not in result

    async def slow(args: dict[str, str]) -> dict[str, str]:
        await asyncio.sleep(1)
        return {}

    handler.side_effect = slow
    assert "source_unavailable" in asyncio.run(executor.execute("test", "{}"))
    handler.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(executor.execute("test", "{}"))


def test_exchange_rate_uses_nominal_date_and_cross_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml = (
        b'<ValCurs Date="25.09.2026"><Valute><CharCode>USD</CharCode>'
        b"<Nominal>1</Nominal><Value>80,00</Value></Valute><Valute>"
        b"<CharCode>JPY</CharCode><Nominal>100</Nominal>"
        b"<Value>50,00</Value></Valute></ValCurs>"
    )
    fetch = AsyncMock(return_value=xml)
    monkeypatch.setattr(live_data_module, "fetch_provider", fetch)
    result = asyncio.run(get_exchange_rate({"base": "USD", "quote": "JPY"}))
    assert result["rate"] == "160.000000"
    assert result["effective_date"] == "25.09.2026"
    assert result["source"].startswith("https://www.cbr.ru/")
    assert asyncio.run(get_exchange_rate({}))["quote"] == "RUB"
    with pytest.raises(ValueError):
        asyncio.run(get_exchange_rate({"base": "https://localhost"}))
    with pytest.raises(KeyError):
        asyncio.run(get_exchange_rate({"base": "ZZZ"}))
    fetch.return_value = b"<!DOCTYPE a><a/>"
    with pytest.raises(ValueError):
        asyncio.run(get_exchange_rate({}))


def test_weather_handles_unknown_ambiguous_and_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = AsyncMock(return_value=b"{}")
    monkeypatch.setattr(live_data_module, "fetch_provider", fetch)
    assert asyncio.run(get_weather({"city": "Nowhere"}))["status"] == "not_found"
    place = {
        "name": "Moscow",
        "country": "Russia",
        "latitude": 55.75,
        "longitude": 37.6,
    }
    fetch.return_value = json.dumps({"results": [place, place]}).encode()
    assert asyncio.run(get_weather({"city": "Moscow"}))["status"] == "ambiguous"
    weather = {
        "timezone": "Europe/Moscow",
        "current": {"temperature_2m": 10, "time": "2026-09-26T12:00"},
        "current_units": {"temperature_2m": "C"},
        "daily": {},
        "daily_units": {},
    }
    fetch.side_effect = [
        json.dumps({"results": [place]}).encode(),
        json.dumps(weather).encode(),
    ]
    result = asyncio.run(get_weather({"city": "Moscow", "country_code": "RU"}))
    assert result["current"]["temperature_2m"] == 10
    assert result["current_units"]["temperature_2m"] == "C"
    assert fetch.await_args_list[-2].args[1]["countryCode"] == "RU"
    for args in ({"city": "x"}, {"city": "Moscow", "country_code": "bad"}):
        with pytest.raises(ValueError):
            asyncio.run(get_weather(args))


@pytest.mark.parametrize("failure", ["", "http", "size"])
def test_provider_fetch_checks_status_size_and_closes_session(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    response = Mock()
    response.status = 503 if failure == "http" else 200

    async def chunks(size: int) -> AsyncIterator[bytes]:
        yield b"x" * 1_048_577 if failure == "size" else b"{}"

    response.content.iter_chunked = chunks
    request = Mock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=False)
    session = Mock()
    session.get.return_value = request
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(http_module, "_session", Mock(return_value=session))
    run = http_module.fetch_provider("https://example.org", {"q": "test"})
    if failure:
        with pytest.raises(ValueError):
            asyncio.run(run)
    else:
        assert asyncio.run(run) == b"{}"
    session.__aexit__.assert_awaited_once()
    assert session.get.call_args.kwargs["allow_redirects"] is False


def test_http_session_uses_optional_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP-сессия должна подключать proxy только при наличии настройки."""
    client_session = Mock()
    connector = Mock()
    from_url = Mock(return_value=connector)
    monkeypatch.setattr(http_module, "ClientSession", client_session)
    monkeypatch.setattr(http_module.ProxyConnector, "from_url", from_url)

    http_module._session(None)
    assert client_session.call_args.kwargs["connector"] is None
    from_url.assert_not_called()

    http_module._session("socks5://proxy.example:1080")
    from_url.assert_called_once_with("socks5://proxy.example:1080")
    assert client_session.call_args.kwargs["connector"] is connector


def test_public_page_follows_validated_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Каждый redirect должен пройти проверку до чтения страницы."""
    redirect = Mock(status=302, headers={"Location": "/final"})
    success = Mock(status=200, headers={"Content-Type": "text/html; charset=utf-8"})

    async def chunks(size: int) -> AsyncIterator[bytes]:
        yield b"<p>ok</p>"

    success.content.iter_chunked = chunks
    redirect_request = Mock()
    redirect_request.__aenter__ = AsyncMock(return_value=redirect)
    redirect_request.__aexit__ = AsyncMock(return_value=False)
    success_request = Mock()
    success_request.__aenter__ = AsyncMock(return_value=success)
    success_request.__aexit__ = AsyncMock(return_value=False)
    session = Mock()
    session.get.side_effect = [redirect_request, success_request]
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    validate = AsyncMock()
    monkeypatch.setattr(http_module, "_session", Mock(return_value=session))
    monkeypatch.setattr(http_module, "_validate_public_url", validate)

    result = asyncio.run(http_module.fetch_public_page(" https://example.org/start "))

    assert result == (b"<p>ok</p>", "https://example.org/final", "text/html")
    assert [call.args[0] for call in validate.await_args_list] == [
        "https://example.org/start",
        "https://example.org/final",
    ]


@pytest.mark.parametrize(
    "status,headers,max_redirects",
    [
        (302, {}, 3),
        (302, {"Location": "/again"}, 0),
        (503, {}, 3),
        (200, {"Content-Type": "application/json"}, 3),
    ],
)
def test_public_page_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
    max_redirects: int,
) -> None:
    """Читалка должна отклонять опасный или неподдерживаемый ответ."""
    response = Mock(status=status, headers=headers)
    request = Mock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=False)
    session = Mock()
    session.get.return_value = request
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(http_module, "_session", Mock(return_value=session))
    monkeypatch.setattr(http_module, "_validate_public_url", AsyncMock())

    with pytest.raises(ValueError):
        asyncio.run(
            http_module.fetch_public_page(
                "https://example.org",
                max_redirects=max_redirects,
            )
        )


def test_web_search_validates_and_normalizes_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Поиск должен вернуть ограниченные результаты со ссылками."""
    payload = {
        "web": {
            "results": [
                {
                    "title": "Новость",
                    "url": "https://example.org/news",
                    "description": "Описание",
                    "page_age": "2026-09-26",
                },
                {"title": 123, "url": None},
            ]
        }
    }
    fetch = AsyncMock(return_value=json.dumps(payload).encode())
    monkeypatch.setattr(web_module, "fetch_provider", fetch)
    result = asyncio.run(
        web_search(
            {
                "query": "новости DeepSeek",
                "count": "3",
                "language": "ru",
                "freshness": "week",
            },
            api_key="secret",
        )
    )
    assert result["results"] == [
        {
            "title": "Новость",
            "url": "https://example.org/news",
            "description": "Описание",
            "published": "2026-09-26",
        }
    ]
    call = fetch.await_args
    assert call is not None
    assert call.args[1]["freshness"] == "pw"
    assert call.kwargs["headers"]["X-Subscription-Token"] == "secret"


@pytest.mark.parametrize(
    "args",
    [
        {"query": ""},
        {"query": "test", "count": "zero"},
        {"query": "test", "count": "11"},
        {"query": "test", "language": "de"},
        {"query": "test", "freshness": "century"},
    ],
)
def test_web_search_rejects_invalid_arguments(args: dict[str, str]) -> None:
    """Некорректные параметры поиска не должны уходить провайдеру."""
    with pytest.raises(ValueError):
        asyncio.run(web_search(args, api_key="secret"))


def test_fetch_web_page_extracts_visible_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Читалка должна убрать script и вернуть канонический URL."""
    html = (
        b"<html><head><title>Test page</title><style>hidden</style></head>"
        b"<body><h1>Hello</h1><script>steal()</script><p>Useful text</p></body></html>"
    )
    fetch = AsyncMock(return_value=(html, "https://example.org/final", "text/html"))
    monkeypatch.setattr(web_module, "fetch_public_page", fetch)
    result = asyncio.run(fetch_web_page({"url": "https://example.org"}))
    assert result["title"] == "Test page"
    assert "Hello" in result["text"]
    assert "Useful text" in result["text"]
    assert "steal" not in result["text"]
    assert result["url"] == "https://example.org/final"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b"plain   text\nnext", "plain text next"),
        (b"<meta charset=unknown-charset>broken", "broken"),
    ],
)
def test_fetch_web_page_handles_plain_text_and_unknown_charset(
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    expected: str,
) -> None:
    """Читалка должна нормализовать текст и переживать неизвестную кодировку."""
    content_type = "text/plain" if raw.startswith(b"plain") else "text/html"
    fetch = AsyncMock(return_value=(raw, "https://example.org", content_type))
    monkeypatch.setattr(web_module, "fetch_public_page", fetch)

    result = asyncio.run(fetch_web_page({"url": "https://example.org"}))

    assert result["text"] == expected
    assert result["title"] == ""


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/admin",
        "http://user:password@example.org/",
        "http://example.org:bad/",
    ],
)
def test_public_page_rejects_unsafe_url(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Читалка не должна обращаться к локальным и некорректным адресам."""
    monkeypatch.setattr(
        http_module.socket,
        "getaddrinfo",
        Mock(return_value=[(2, 1, 6, "", ("127.0.0.1", 80))]),
    )
    with pytest.raises(ValueError):
        asyncio.run(http_module._validate_public_url(url))


def test_public_page_allows_only_global_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS-ответ должен содержать только глобальные IP-адреса."""
    resolver = Mock(return_value=[(2, 1, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(http_module.socket, "getaddrinfo", resolver)
    asyncio.run(http_module._validate_public_url("https://example.org/page"))
    resolver.return_value = [(2, 1, 6, "", ("10.0.0.1", 443))]
    with pytest.raises(ValueError):
        asyncio.run(http_module._validate_public_url("https://example.org/page"))
