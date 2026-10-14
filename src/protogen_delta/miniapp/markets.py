"""Защищённые маршруты отдельного раздела «Рынки» Telegram Mini App."""

from collections.abc import Callable
from typing import Any

from aiohttp import web

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.miniapp.auth import MiniAppUser
from protogen_delta.services.deepseek import DeepSeekError
from protogen_delta.services.market_advisor import MarketAdvisor
from protogen_delta.services.market_models import MarketDataError


class MarketRoutes:
    """Проверить подпись до источников и модели; не принимать ID и историю из JSON."""

    def __init__(
        self,
        authenticate: Callable[[web.Request], MiniAppUser],
        advisor: MarketAdvisor | None,
    ) -> None:
        self._authenticate, self._advisor = authenticate, advisor
        self._limits = {
            kind: UserRateLimiter(cooldown_seconds=seconds)
            for kind, seconds in {
                "quotes": 0.5,
                "chart": 0.5,
                "news": 2,
                "rates": 0.5,
                "convert": 0.5,
                "consult": 3,
                "history": 0.2,
            }.items()
        }

    def routes(self) -> list[web.RouteDef]:
        return [
            web.get("/api/markets/{kind:quotes|chart|news|rates}", self.read),
            web.post("/api/markets/convert", self.convert),
            web.post("/api/markets/consult", self.consult),
            web.delete("/api/markets/history", self.clear),
        ]

    def _access(self, request: web.Request, kind: str) -> tuple[int, MarketAdvisor]:
        user = self._authenticate(request)
        if self._advisor is None:
            raise web.HTTPServiceUnavailable(text="Раздел «Рынки» сейчас недоступен.")
        if not self._limits[kind].allow(user.id):
            raise web.HTTPTooManyRequests(
                text="Подожди немного перед следующим запросом."
            )
        return user.id, self._advisor

    @staticmethod
    def _response(payload: dict[str, Any]) -> web.Response:
        return web.json_response(
            payload,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @staticmethod
    async def _payload(request: web.Request, allowed: set[str]) -> dict[str, str]:
        try:
            payload = await request.json()
            if (
                not isinstance(payload, dict)
                or set(payload) - allowed
                or not all(isinstance(value, str) for value in payload.values())
            ):
                raise ValueError
            return payload
        except ValueError, TypeError, RecursionError:
            raise web.HTTPBadRequest(
                text="Нужен JSON с текстовыми полями запроса; чужие ID и историю передавать нельзя."
            ) from None

    async def read(self, request: web.Request) -> web.Response:
        kind = request.match_info["kind"]
        _, advisor = self._access(request, kind)
        if set(request.query) - {"market", "symbols", "symbol", "period"}:
            raise web.HTTPBadRequest(text="Неизвестные параметры рынка.")
        try:
            if kind == "quotes":
                payload = await advisor.data.quotes(
                    request.query.get("market", "us"), request.query.get("symbols", "")
                )
            elif kind == "chart":
                payload = await advisor.data.chart(
                    request.query.get("market", "us"),
                    request.query.get("symbol", "SPY"),
                    request.query.get("period", "1m"),
                )
            elif kind == "news":
                payload = await advisor.data.news()
            else:
                payload = (await advisor.data.rates()).payload()
        except MarketDataError as error:
            raise web.HTTPServiceUnavailable(text=str(error)) from None
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from None
        return self._response(payload)

    async def convert(self, request: web.Request) -> web.Response:
        _, advisor = self._access(request, "convert")
        payload = await self._payload(request, {"amount", "from", "to"})
        try:
            result = (await advisor.data.rates()).convert(
                payload.get("amount", ""),
                payload.get("from", ""),
                payload.get("to", ""),
            )
        except MarketDataError as error:
            raise web.HTTPServiceUnavailable(text=str(error)) from None
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from None
        return self._response(result)

    async def consult(self, request: web.Request) -> web.Response:
        user_id, advisor = self._access(request, "consult")
        payload = await self._payload(request, {"text", "market", "symbols", "period"})
        try:
            result = await advisor.answer(
                user_id,
                payload.get("text", ""),
                payload.get("market", "us"),
                payload.get("symbols", ""),
                payload.get("period", "1m"),
            )
        except MarketDataError, DeepSeekError:
            raise web.HTTPServiceUnavailable(
                text="Консультант временно недоступен. История не дополнена; попробуй позже."
            ) from None
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from None
        return self._response(result)

    async def clear(self, request: web.Request) -> web.Response:
        user_id, advisor = self._access(request, "history")
        await advisor.clear(user_id)
        return self._response({"cleared": True})
