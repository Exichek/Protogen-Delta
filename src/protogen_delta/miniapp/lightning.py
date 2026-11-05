"""Наблюдения Foreca: ключ остаётся на сервере, доступ выключен без ключа."""

import asyncio
import json
import math
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any

from aiohttp import web

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.miniapp.auth import MiniAppUser
from protogen_delta.services.tools.http import fetch_provider


class LightningRoutes:
    """Показывать только полученные разряды, отдельно от прогноза грозы."""

    def __init__(
        self,
        authenticate: Callable[[web.Request], MiniAppUser],
        key: str | None,
        proxy_url: str | None,
    ) -> None:
        self._authenticate = authenticate
        self._key = key
        self._proxy_url = proxy_url
        self._limiter = UserRateLimiter(cooldown_seconds=2)
        self._slot = asyncio.Semaphore(1)
        self._cache: OrderedDict[tuple[float, float], tuple[float, dict[str, Any]]] = (
            OrderedDict()
        )
        self._day = ""
        self._attempts = 0
        self._retry_at = 0.0

    def routes(self) -> list[web.RouteDef]:
        return [web.get("/api/weather/lightning", self.observations)]

    async def observations(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        if set(request.query) != {"latitude", "longitude"} or any(
            len(request.query.getall(k)) != 1 for k in request.query
        ):
            raise web.HTTPBadRequest(text="Нужны только широта и долгота города.")
        try:
            lat, lon = (float(request.query[k]) for k in ("latitude", "longitude"))
        except ValueError:
            raise web.HTTPBadRequest(text="Некорректные координаты.") from None
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not (-90 <= lat <= 90 and -180 <= lon <= 180)
        ):
            raise web.HTTPBadRequest(text="Некорректные координаты.")
        if not self._limiter.allow(user.id):
            raise web.HTTPTooManyRequests(text="Подожди пару секунд.")
        if not self._key:
            return self._response({"status": "disabled"})
        if not (-60 <= lat <= 75 and (lon <= 100 or lon >= 160)):
            return self._response({"status": "outside_coverage"})
        position = (round(lat, 2), round(lon, 2))
        cached = self._cache.get(position)
        if cached and monotonic() - cached[0] < 600:
            self._cache.move_to_end(position)
            return self._response(cached[1])
        day = datetime.now(timezone(timedelta(hours=3))).date().isoformat()
        if day != self._day:
            self._day, self._attempts = day, 0
        if self._attempts >= 144:
            raise web.HTTPTooManyRequests(text="Лимит наблюдений на сегодня исчерпан.")
        if monotonic() < self._retry_at or self._slot.locked():
            raise web.HTTPServiceUnavailable(text="Наблюдения пока недоступны.")
        self._attempts += 1
        try:
            async with self._slot:
                raw = await fetch_provider(
                    f"https://weatherapi.foreca.net/api/v1/lightning/"
                    f"{position[1]:.2f},{position[0]:.2f}",
                    {"radius": "100", "limit": "1000"},
                    headers={"Authorization": f"Bearer {self._key}"},
                    proxy_url=self._proxy_url,
                )
                result = self._parse(json.loads(raw))
        except Exception:
            self._retry_at = monotonic() + 60
            raise web.HTTPBadGateway(
                text="Источник молний недоступен. Попробуй позже."
            ) from None
        self._cache[position] = (monotonic(), result)
        while len(self._cache) > 128:
            self._cache.popitem(last=False)
        return self._response(result)

    @staticmethod
    def _parse(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict) or not isinstance(data.get("strikes"), list):
            raise ValueError("Некорректные наблюдения")
        if len(data["strikes"]) > 1000:
            raise ValueError("Слишком много наблюдений")
        now = datetime.now(timezone.utc)
        strikes = []
        for item in data["strikes"]:
            lat, lon = item["lat"], item["lon"]
            at = datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
            if (
                type(lat) not in (int, float)
                or type(lon) not in (int, float)
                or not math.isfinite(lat)
                or not math.isfinite(lon)
                or not (-90 <= lat <= 90 and -180 <= lon <= 180)
                or at.tzinfo is None
                or not timedelta(0) <= now - at <= timedelta(minutes=30)
            ):
                raise ValueError("Некорректный разряд")
            strikes.append({"latitude": lat, "longitude": lon, "time": at.isoformat()})
        return {
            "status": "ok",
            "strikes": strikes,
            "as_of": now.isoformat(),
            "radius_km": 100,
            "minutes": 30,
            "possibly_truncated": len(strikes) == 1000,
            "attribution": "Foreca",
        }

    @staticmethod
    def _response(data: dict[str, Any]) -> web.Response:
        return web.json_response(
            data,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
