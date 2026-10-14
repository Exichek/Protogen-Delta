"""Чтение фиксированных биржевых источников с ограничением запросов и кэшем."""

import asyncio
from collections import OrderedDict, deque
from collections.abc import Mapping
from time import monotonic
from urllib.parse import urlsplit

from aiohttp import ClientError

from protogen_delta.services.market_models import MarketDataError
from protogen_delta.services.tools.http import _read_limited, _session


class MarketHTTP:
    """Не передавать ключи другим хостам и не повторять одновременные загрузки."""

    def __init__(
        self,
        key: str | None = None,
        secret: str | None = None,
        *,
        proxy_url: str | None = None,
    ) -> None:
        self._key, self._secret, self._proxy = key, secret, proxy_url
        self._cache: OrderedDict[str, tuple[float, bytes | MarketDataError]] = (
            OrderedDict()
        )
        self._pending: dict[str, asyncio.Task[bytes]] = {}
        self._slots = asyncio.Semaphore(4)
        self._alpaca_requests: deque[float] = deque()

    @property
    def alpaca_enabled(self) -> bool:
        return bool(self._key and self._secret)

    async def get(
        self, url: str, parameters: Mapping[str, str], *, ttl: float = 30
    ) -> bytes:
        """Вернуть ограниченный ответ; ошибка хранится десять секунд, успех — TTL."""
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc
            not in {"data.alpaca.markets", "iss.moex.com", "www.cbr.ru"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Разрешены только фиксированные источники рынка.")
        identity = url + repr(sorted(parameters.items()))
        cached = self._cache.get(identity)
        if cached and cached[0] > monotonic():
            self._cache.move_to_end(identity)
            if isinstance(cached[1], MarketDataError):
                raise MarketDataError(str(cached[1]))
            return cached[1]
        if identity not in self._pending:
            if len(self._pending) >= 64:
                raise MarketDataError("Источники заняты. Попробуй немного позже.")
            self._pending[identity] = asyncio.create_task(
                self._load(identity, url, parameters, ttl)
            )
            self._pending[identity].add_done_callback(self._consume_failure)
        return await asyncio.shield(self._pending[identity])

    @staticmethod
    def _consume_failure(task: asyncio.Task[bytes]) -> None:
        """Отмена клиента не оставляет исключение фоновой загрузки без обработки."""
        if not task.cancelled():
            task.exception()

    async def _load(
        self, identity: str, url: str, parameters: Mapping[str, str], ttl: float
    ) -> bytes:
        headers = {"User-Agent": "ProtogenDelta/0.1 (markets)"}
        alpaca = urlsplit(url).hostname == "data.alpaca.markets"
        if alpaca and self.alpaca_enabled:
            headers.update(
                {
                    "APCA-API-KEY-ID": self._key or "",
                    "APCA-API-SECRET-KEY": self._secret or "",
                }
            )
        try:
            if alpaca:
                now = monotonic()
                while self._alpaca_requests and now - self._alpaca_requests[0] >= 60:
                    self._alpaca_requests.popleft()
                if len(self._alpaca_requests) >= 180:
                    raise MarketDataError(
                        "Лимит источника Alpaca: попробуй через минуту."
                    )
                self._alpaca_requests.append(now)
            async with self._slots, _session(self._proxy) as session:
                async with session.get(
                    url, params=parameters, headers=headers, allow_redirects=False
                ) as response:
                    if response.status != 200:
                        notice = {
                            401: "Alpaca: проверь серверные ключи доступа.",
                            403: "Alpaca: тариф не разрешает эти данные.",
                            429: "Источник ограничил частоту запросов. Попробуй позже.",
                        }.get(response.status, "Источник рынка временно недоступен.")
                        raise MarketDataError(notice)
                    result = await _read_limited(response)
            self._remember(identity, result, ttl)
            return result
        except MarketDataError as error:
            self._remember(identity, MarketDataError(str(error)), 10)
            raise
        except ClientError, TimeoutError, ValueError:
            # Тело ошибки, заголовки авторизации и адрес прокси не попадают в ответ.
            unavailable = MarketDataError("Источник рынка временно недоступен.")
            self._remember(identity, unavailable, 10)
            raise unavailable from None
        finally:
            self._pending.pop(identity, None)

    def _remember(
        self, identity: str, value: bytes | MarketDataError, ttl: float
    ) -> None:
        self._cache[identity] = (monotonic() + ttl, value)
        self._cache.move_to_end(identity)
        while len(self._cache) > 128:
            self._cache.popitem(last=False)

    async def close(self) -> None:
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._cache.clear()
