"""Защищённый прогноз миниаппа без обращения к языковой модели."""

import asyncio
import re
from collections import OrderedDict
from collections.abc import Callable
from importlib.resources import files
from time import monotonic
from typing import Any

from aiohttp import web

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.miniapp.auth import MiniAppUser
from protogen_delta.services.tools.live_data import get_weather


class WeatherRoutes:
    """Ограничить запросы и кешировать только публичные погодные данные."""

    def __init__(
        self, authenticate: Callable[[web.Request], MiniAppUser], proxy_url: str | None
    ) -> None:
        self._authenticate = authenticate
        self._proxy_url = proxy_url
        self._limiter = UserRateLimiter(cooldown_seconds=1)
        self._slots = asyncio.Semaphore(4)
        self._cache: OrderedDict[
            tuple[str, str, str, str], tuple[float, dict[str, Any]]
        ] = OrderedDict()

    def routes(self) -> list[web.RouteDef]:
        return [
            web.get("/api/weather", self.forecast),
            web.get("/static/weather.js", self.javascript),
            web.get(
                "/static/{asset:weather-map.js|weather-time.js|fullscreen.js|"
                "maplibre.js|maplibre.css|weather-map-style.json|maplibre-LICENSE.txt}",
                self.asset,
            ),
        ]

    async def javascript(self, request: web.Request) -> web.Response:
        return web.Response(
            text=files("protogen_delta.miniapp.static")
            .joinpath("weather.js")
            .read_text("utf-8"),
            content_type="application/javascript",
            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    async def asset(self, request: web.Request) -> web.Response:
        """Отдать только явно перечисленные ресурсы карты."""
        name = request.match_info["asset"]
        allowed = {
            "weather-map.js",
            "weather-time.js",
            "fullscreen.js",
            "maplibre.js",
            "maplibre.css",
            "weather-map-style.json",
            "maplibre-LICENSE.txt",
        }
        if name not in allowed:
            raise web.HTTPNotFound()
        kinds = {
            "css": "text/css",
            "json": "application/json",
            "txt": "text/plain",
            "js": "application/javascript",
        }
        return web.Response(
            body=files("protogen_delta.miniapp.static").joinpath(name).read_bytes(),
            content_type=kinds[name.rsplit(".", 1)[-1]],
            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    async def forecast(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        if set(request.query) - {
            "city",
            "country_code",
            "region",
            "location_id",
        } or any(len(request.query.getall(key)) != 1 for key in request.query):
            raise web.HTTPBadRequest(
                text="Неизвестные или повторяющиеся параметры погоды."
            )
        args = {key: value.strip() for key, value in request.query.items()}
        if not 2 <= len(args.get("city", "")) <= 100 or any(
            len(v) > 100 for v in args.values()
        ):
            raise web.HTTPBadRequest(text="Укажи город: от 2 до 100 символов.")
        country = args.get("country_code", "")
        if country and not re.fullmatch("[A-Za-z]{2}", country):
            raise web.HTTPBadRequest(text="Нужен двухбуквенный код страны.")
        if args.get("location_id") and not re.fullmatch(
            r"[0-9]{1,20}", args["location_id"]
        ):
            raise web.HTTPBadRequest(text="Некорректный ID города.")
        if not self._limiter.allow(user.id):
            raise web.HTTPTooManyRequests(
                text="Подожди секунду перед следующим запросом."
            )
        key = (
            args["city"].casefold(),
            args.get("country_code", "").upper(),
            args.get("region", "").casefold(),
            args.get("location_id", ""),
        )
        cached = self._cache.get(key)
        if cached is not None and monotonic() - cached[0] < 300:
            payload = cached[1]
            self._cache.move_to_end(key)
        else:
            if self._slots.locked():
                raise web.HTTPServiceUnavailable(
                    text="Погода занята. Попробуй чуть позже."
                )
            try:
                async with asyncio.timeout(12), self._slots:
                    payload = await get_weather(
                        args, proxy_url=self._proxy_url, forecast_days=10
                    )
            except ValueError, KeyError, TypeError:
                raise web.HTTPBadGateway(
                    text="Источник погоды вернул некорректные данные."
                ) from None
            except Exception:
                raise web.HTTPBadGateway(
                    text="Не удалось получить погоду. Попробуй позже."
                ) from None
            self._cache[key] = (monotonic(), payload)
            self._cache.move_to_end(key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return web.json_response(
            payload,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
