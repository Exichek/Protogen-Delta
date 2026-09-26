"""Контракты внешних инструментов без сети и секретов."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest

import protogen_delta.services.tools as module
from protogen_delta.services.tools import (
    Tool,
    ToolExecutor,
    ToolRegistry,
    default_registry,
    get_current_time,
    get_exchange_rate,
    get_weather,
)


def test_registry_and_time() -> None:
    registry = default_registry()
    assert len(registry.schemas()) == 3
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
    monkeypatch.setattr(module, "_fetch", fetch)
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
    monkeypatch.setattr(module, "_fetch", fetch)
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


def test_http_read_limit_and_no_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b"{}"
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(module, "build_opener", Mock(return_value=opener))
    assert asyncio.run(module._fetch("https://example.org", {"q": "a b"})) == b"{}"
    assert opener.open.call_args.args[0].full_url.endswith("q=a+b")
    response.read.return_value = b"x" * 1_048_577
    with pytest.raises(ValueError):
        asyncio.run(module._fetch("https://example.org", {}))
    module.NoRedirect().redirect_request()


@pytest.mark.parametrize("failure", ["", "http", "size"])
def test_proxy_fetch_checks_status_size_and_closes_session(
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
    monkeypatch.setattr(module, "ClientSession", Mock(return_value=session))
    monkeypatch.setattr(module.ProxyConnector, "from_url", Mock())
    run = module._fetch("https://example.org", {}, proxy_url="socks5://localhost:1080")
    if failure:
        with pytest.raises(ValueError):
            asyncio.run(run)
    else:
        assert asyncio.run(run) == b"{}"
    session.__aexit__.assert_awaited_once()
    assert session.get.call_args.kwargs["allow_redirects"] is False
