"""Проверять адрес фактического соединения, а не только предварительный DNS."""

import asyncio
import ipaddress
import socket
from ssl import SSLContext
from typing import Any

from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp_socks import ProxyConnector
from aiohttp_socks.connector import _ResponseHandler
from python_socks.async_.asyncio.v2 import Proxy


class PublicResolver(AbstractResolver):
    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[ResolveResult]:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, port, family=family, type=socket.SOCK_STREAM
        )
        result: list[ResolveResult] = []
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global or getattr(ip, "ipv4_mapped", None) is not None:
                raise ValueError("Локальные и служебные адреса запрещены")
            result.append(
                {
                    "hostname": host,
                    "host": str(ip),
                    "port": port,
                    "family": address[0],
                    "proto": address[2],
                    "flags": socket.AI_NUMERICHOST,
                }
            )
        if not result:
            raise ValueError("Не удалось определить публичный адрес")
        return result

    async def close(self) -> None:
        pass


class PublicProxyConnector(ProxyConnector):
    """Передать SOCKS-прокси проверенный IP, сохранив TLS имя исходного сайта."""

    def __init__(self, url: str) -> None:
        from python_socks import parse_proxy_url

        proxy_type, host, port, username, password = parse_proxy_url(url)
        super().__init__(
            host,
            port,
            proxy_type=proxy_type,
            username=username,
            password=password,
            rdns=False,
        )
        self._public_proxy_url = url

    async def _connect_via_proxy(
        self,
        host: str,
        port: int,
        ssl: SSLContext | None = None,
        timeout: float | None = None,
    ) -> Any:
        addresses = await PublicResolver().resolve(host, port, socket.AF_UNSPEC)
        proxy = Proxy.from_url(self._public_proxy_url, rdns=False)
        stream = await proxy.connect(
            dest_host=addresses[0]["host"], dest_port=port, timeout=timeout
        )
        try:
            if ssl is not None:
                stream = await stream.start_tls(
                    hostname=host, ssl_context=ssl, ssl_handshake_timeout=timeout
                )
        except BaseException:
            await stream.close()
            raise
        transport = stream.writer.transport
        protocol = _ResponseHandler(
            loop=asyncio.get_running_loop(), writer=stream.writer
        )
        transport.set_protocol(protocol)
        protocol.connection_made(transport)
        return transport, protocol
