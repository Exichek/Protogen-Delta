"""DeepSeek Search с резервом Brave и платным Fetch после обычного чтения."""

import asyncio
import ipaddress
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from aiohttp import ClientError

from protogen_delta.repositories.web_api_usage import (
    WebApiLimitError,
    WebApiUsageRepository,
)
from protogen_delta.services.tools.http import (
    ProviderHTTPError,
    UnsafePublicURLError,
    _read_limited,
    _session,
    _validate_public_url,
)
from protogen_delta.services.tools.web import fetch_web_page, web_search
from protogen_delta.services.tools.youtube import youtube_video_id

logger = logging.getLogger(__name__)
_FRESH_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}


def _public_result_url(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 2048:
        return False
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if (
            parsed.scheme not in {"https", "http"}
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or not 0 < (parsed.port or 443) < 65536
        ):
            return False
        if host.casefold().rstrip(".") == "localhost" or host.casefold().endswith(
            (".localhost", ".local")
        ):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    except ValueError:
        return False


class WebSearchService:
    def __init__(
        self,
        usage: WebApiUsageRepository,
        *,
        deepseek_key: str | None = None,
        brave_key: str | None = None,
        proxy_url: str | None = None,
        clock: Callable[[], float] = monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._usage, self._deepseek_key, self._brave_key = (
            usage,
            deepseek_key,
            brave_key,
        )
        self._proxy, self._clock, self._now = proxy_url, clock, now
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._pending: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._slots = asyncio.Semaphore(2)
        self._pace = {
            name: asyncio.Lock()
            for name in ("deepseek_search", "deepseek_fetch", "brave")
        }
        self._next: dict[str, float] = {}
        self._blocked: dict[str, float] = {}
        self._closed = False

    @property
    def enabled(self) -> bool:
        return bool(self._deepseek_key or self._brave_key)

    async def _cached(
        self, key: str, load: Callable[[], Awaitable[dict[str, Any]]]
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Веб-инструменты остановлены")
        cached = self._cache.get(key)
        if cached and cached[0] > self._clock():
            self._cache.move_to_end(key)
            return deepcopy(cached[1])
        if key not in self._pending:
            if len(self._pending) >= 32:
                raise RuntimeError("Веб-инструменты заняты")
            self._pending[key] = asyncio.create_task(self._load(key, load))
            self._pending[key].add_done_callback(self._consume)
        return deepcopy(await asyncio.shield(self._pending[key]))

    @staticmethod
    def _consume(task: asyncio.Task[dict[str, Any]]) -> None:
        if not task.cancelled():
            task.exception()

    async def _load(
        self, key: str, load: Callable[[], Awaitable[dict[str, Any]]]
    ) -> dict[str, Any]:
        try:
            async with self._slots:
                result = await load()
            # Кеш только в памяти: максимум 128 ограниченных результатов на пять минут.
            self._cache[key] = self._clock() + 300, result
            self._cache.move_to_end(key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
            return result
        finally:
            self._pending.pop(key, None)

    async def _reserve(self, provider: str, operation: str) -> None:
        name = "deepseek_" + operation if provider == "deepseek" else "brave"
        if self._blocked.get(name, 0) > self._clock():
            raise RuntimeError("Источник временно приостановлен")
        async with self._pace[name]:
            delay = self._next.get(name, 0) - self._clock()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next[name] = self._clock() + (1.0 if provider == "brave" else 0.3)
        await self._usage.reserve(provider, operation)

    def _failed(self, name: str, error: Exception) -> None:
        status = error.status if isinstance(error, ProviderHTTPError) else 0
        delay = 3600 if status in {401, 402, 403} else 60 if status == 429 else 30
        self._blocked[name] = self._clock() + delay
        logger.info("Web API provider=%s outcome=unavailable status=%d", name, status)

    async def _post(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Ключ отправляется только на фиксированный официальный адрес; без redirect/retry.
        async def request() -> dict[str, Any]:
            async with _session(self._proxy) as session:
                async with session.post(
                    "https://api.deepseek.com/v1/tools/" + operation,
                    json=payload,
                    headers={
                        "Authorization": "Bearer " + (self._deepseek_key or ""),
                        "Content-Type": "application/json",
                    },
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        raise ProviderHTTPError(response.status)
                    result = json.loads(await _read_limited(response))
                    if not isinstance(result, dict):
                        raise ValueError("Некорректный ответ веб-API")
                    return result

        return await asyncio.wait_for(request(), timeout=4.5)

    async def search(self, args: dict[str, str]) -> dict[str, Any]:
        query = args["query"].strip()
        count = int(args.get("count", "5"))
        language, freshness = (
            args.get("language", "ru").lower(),
            args.get("freshness", "").lower(),
        )
        if (
            not 2 <= len(query) <= 100
            or not 1 <= count <= 10
            or language not in {"ru", "en"}
            or freshness
            and freshness not in _FRESH_DAYS
        ):
            raise ValueError("Некорректные параметры поиска")
        normalized = {
            "query": query,
            "count": str(count),
            "language": language,
            "freshness": freshness,
        }
        return await self._cached(
            "search:"
            + self._now().date().isoformat()
            + repr(sorted(normalized.items())),
            lambda: self._search(normalized),
        )

    async def _search(self, args: dict[str, str]) -> dict[str, Any]:
        count = int(args["count"])
        for provider, key in (
            ("deepseek", self._deepseek_key),
            ("brave", self._brave_key),
        ):
            name = "deepseek_search" if provider == "deepseek" else "brave"
            if not key or self._blocked.get(name, 0) > self._clock():
                continue
            try:
                await self._reserve(provider, "search")
                if provider == "deepseek":
                    query = args["query"] + " lang:" + args["language"]
                    if args["freshness"]:
                        after = (
                            self._now() - timedelta(days=_FRESH_DAYS[args["freshness"]])
                        ).date()
                        query += " after:" + after.isoformat()
                    payload = await self._post(
                        "search",
                        {"query": query, "count": count, "need_content": False},
                    )
                    rows = payload.get("results")
                else:
                    payload = await asyncio.wait_for(
                        web_search(args, api_key=key, proxy_url=self._proxy),
                        timeout=4.5,
                    )
                    rows = payload.get("results")
                if not isinstance(rows, list):
                    raise ValueError("Некорректная выдача поиска")
                results: list[dict[str, Any]] = []
                size = 0
                for row in rows[:count]:
                    if (
                        not isinstance(row, dict)
                        or not isinstance(row.get("title"), str)
                        or not _public_result_url(row.get("url"))
                    ):
                        continue
                    description = (
                        row.get("summary")
                        or row.get("snippet")
                        or row.get("description")
                        or ""
                    )
                    published = row.get("date") or row.get("published")
                    item = {
                        "title": row["title"][:300],
                        "url": row["url"],
                        "description": (
                            description[:700] if isinstance(description, str) else ""
                        ),
                        "published": (
                            published[:80] if isinstance(published, str) else None
                        ),
                    }
                    size += len(json.dumps(item, ensure_ascii=False))
                    if size > 12000:
                        break
                    results.append(item)
                if rows and not results:
                    raise ValueError("В выдаче нет пригодных публичных ссылок")
                return {
                    "query": args["query"],
                    "results": results,
                    "checked_at": self._now().isoformat(),
                    "notice": "Даты и описания взяты из поисковой выдачи; чтение страницы подтверждается отдельно.",
                    "source": (
                        "DeepSeek Search API"
                        if provider == "deepseek"
                        else "Brave Search API"
                    ),
                }
            except WebApiLimitError:
                raise
            except (ClientError, TimeoutError, ValueError, RuntimeError) as error:
                self._failed(
                    "deepseek_search" if provider == "deepseek" else "brave", error
                )
        raise RuntimeError("Поисковые источники временно недоступны")

    async def read(self, args: dict[str, str]) -> dict[str, Any]:
        url = args["url"].strip()
        if len(url) > 2048:
            raise ValueError("Ссылка слишком длинная")
        await _validate_public_url(url)
        return await self._cached("page:" + url, lambda: self._read({"url": url}))

    async def _read(self, args: dict[str, str]) -> dict[str, Any]:
        try:
            result = await asyncio.wait_for(
                fetch_web_page(args, proxy_url=self._proxy), timeout=5
            )
            if result.get("text") or youtube_video_id(args["url"]):
                return result
        except UnsafePublicURLError:
            raise
        except ClientError, TimeoutError, ValueError:
            if youtube_video_id(args["url"]):
                raise
        if (
            not self._deepseek_key
            or self._blocked.get("deepseek_fetch", 0) > self._clock()
        ):
            raise RuntimeError("Страница временно недоступна")
        try:
            await self._reserve("deepseek", "fetch")
            page = await self._post("fetch", {"url": args["url"]})
            content, title = page.get("content"), page.get("title", "")
            if (
                not isinstance(content, str)
                or not content.strip()
                or not isinstance(title, str)
            ):
                raise ValueError("Источник не вернул текст страницы")
            return {
                "url": args["url"],
                "title": title[:300],
                "text": content[:12000],
                "truncated": len(content) > 12000,
                "checked_at": self._now().isoformat(),
                "source": "DeepSeek Fetch API",
                "final_url_verified": False,
            }
        except WebApiLimitError:
            raise
        except (ClientError, TimeoutError, ValueError, RuntimeError) as error:
            self._failed("deepseek_fetch", error)
            raise RuntimeError("Страница временно недоступна") from None

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._cache.clear()
