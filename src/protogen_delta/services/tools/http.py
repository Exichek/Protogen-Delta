"""Ограниченные HTTP-запросы инструментов и защита от SSRF."""

import asyncio
import ipaddress
import socket
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

from aiohttp import ClientSession, ClientTimeout
from aiohttp_socks import ProxyConnector

_MAX_BODY_BYTES = 1_048_576
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _session(proxy_url: str | None) -> ClientSession:
    """Создать короткоживущую HTTP-сессию с необязательным прокси."""
    connector = ProxyConnector.from_url(proxy_url) if proxy_url else None
    return ClientSession(connector=connector, timeout=ClientTimeout(total=8))


async def fetch_provider(
    url: str,
    parameters: Mapping[str, str],
    *,
    proxy_url: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> bytes:
    """Запросить фиксированный адрес провайдера, максимум 1 MiB."""
    request_headers = {"User-Agent": "ProtogenDelta/0.1 (live data)"}
    request_headers.update(headers or {})
    async with _session(proxy_url) as session:
        async with session.get(
            url,
            params=parameters,
            headers=request_headers,
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise ValueError("Источник вернул ошибку")
            return await _read_limited(response)


async def fetch_public_page(
    url: str,
    *,
    proxy_url: str | None = None,
    max_redirects: int = 3,
) -> tuple[bytes, str, str]:
    """Загрузить публичную HTML/text-страницу с проверкой каждого redirect."""
    current = url.strip()
    async with _session(proxy_url) as session:
        for redirect in range(max_redirects + 1):
            await _validate_public_url(current)
            async with session.get(
                current,
                headers={"User-Agent": "ProtogenDelta/0.1 (web reader)"},
                allow_redirects=False,
            ) as response:
                if response.status in _REDIRECT_STATUSES:
                    if redirect == max_redirects:
                        raise ValueError("Слишком много перенаправлений")
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Пустое перенаправление")
                    current = urljoin(current, location)
                    continue
                if response.status != 200:
                    raise ValueError("Страница вернула ошибку")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
                if content_type not in {
                    "text/html",
                    "application/xhtml+xml",
                    "text/plain",
                }:
                    raise ValueError("Неподдерживаемый тип страницы")
                return await _read_limited(response), current, content_type
    raise ValueError("Не удалось загрузить страницу")


async def _read_limited(response: Any) -> bytes:
    """Прочитать HTTP-ответ, не позволяя источнику превысить лимит."""
    data = bytearray()
    async for chunk in response.content.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > _MAX_BODY_BYTES:
            raise ValueError("Ответ слишком большой")
    return bytes(data)


async def _validate_public_url(url: str) -> None:
    """Отклонить credentials, локальные имена и неглобальные IP-адреса."""
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname.casefold() == "localhost"
    ):
        raise ValueError("Разрешены только публичные HTTP(S)-адреса")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise ValueError("Некорректный порт") from error

    def resolve() -> list[tuple[Any, ...]]:
        return socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)

    addresses = await asyncio.to_thread(resolve)
    if not addresses or any(
        not ipaddress.ip_address(item[4][0]).is_global for item in addresses
    ):
        raise ValueError("Локальные и служебные адреса запрещены")
