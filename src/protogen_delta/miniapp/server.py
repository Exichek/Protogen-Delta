"""Встроенный HTTP-сервер панели настроек Telegram Mini App."""

import asyncio
from importlib.resources import files
from typing import Any, cast

from aiohttp import web

from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.core.user_state import ContentMode, UserState, UserStateStore
from protogen_delta.miniapp.auth import (
    MiniAppAuthError,
    MiniAppUser,
    validate_init_data,
)
from protogen_delta.services.appearance_image import (
    MAX_APPEARANCE_BYTES,
    prepare_appearance_image,
)
from protogen_delta.services.response_engine import (
    AppearanceAnalysisError,
    ResponseBusyError,
    ResponseEngine,
)

_MAX_PROFILE_FIELD_CHARS = 1000


def _profile(
    state: UserState, user: MiniAppUser, *, appearance_upload_enabled: bool = False
) -> dict[str, Any]:
    """Собрать публичное представление собственного профиля пользователя."""
    return {
        "user": {
            "id": user.id,
            "first_name": user.first_name,
            "username": user.username,
        },
        "content_mode": state.content_mode,
        "roleplay_active": state.roleplay_active,
        "roleplay_configuration": state.roleplay_configuration,
        "roleplay_character": state.roleplay_character,
        "delta_appearance": state.delta_appearance,
        "appearance_upload_enabled": appearance_upload_enabled,
        "roleplay_fetishes": list(state.roleplay_fetishes),
        "roleplay_preferences": state.roleplay_preferences,
        "roleplay_boundaries": state.roleplay_boundaries,
    }


class MiniAppServer:
    """Обслуживать статическую панель и API только с проверенным initData."""

    def __init__(
        self,
        bot_token: str,
        user_states: UserStateStore,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        auth_max_age_seconds: int = 3600,
        response_engine: ResponseEngine | None = None,
    ) -> None:
        if not host.strip():
            raise ValueError("host не может быть пустым")
        if not 1 <= port <= 65535:
            raise ValueError("port должен быть от 1 до 65535")
        if auth_max_age_seconds <= 0:
            raise ValueError("auth_max_age_seconds должен быть больше нуля")
        self._bot_token = bot_token
        self._user_states = user_states
        self._host = host
        self._port = port
        self._auth_max_age_seconds = auth_max_age_seconds
        self._response_engine = response_engine
        self._appearance_pending: set[int] = set()
        self._appearance_slots = asyncio.Semaphore(2)
        self._appearance_limiter = UserRateLimiter(cooldown_seconds=10)
        self._runner: web.AppRunner | None = None

    def application(self) -> web.Application:
        """Создать aiohttp-приложение для запуска и тестирования."""
        application = web.Application(client_max_size=64 * 1024)
        application.add_routes(
            [
                web.get("/", self._index),
                web.get("/api/profile", self._get_profile),
                web.patch("/api/profile", self._update_profile),
                web.delete("/api/profile/roleplay", self._delete_roleplay_profile),
                web.post("/api/profile/appearance", self._upload_appearance),
                web.delete("/api/profile/appearance", self._reset_appearance),
                web.get("/health", self._health),
            ]
        )
        return application

    async def start(self) -> None:
        """Запустить HTTP-сервер."""
        if self._runner is not None:
            return
        runner = web.AppRunner(self.application())
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        try:
            await site.start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner

    async def close(self) -> None:
        """Остановить HTTP-сервер и освободить порт."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _index(self, request: web.Request) -> web.Response:
        del request
        html = (
            files("protogen_delta.miniapp.static")
            .joinpath("index.html")
            .read_text(encoding="utf-8")
        )
        return web.Response(
            text=html,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' https://telegram.org "
                    "'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                    "connect-src 'self'; img-src 'self' data: blob: https:"
                ),
            },
        )

    async def _health(self, request: web.Request) -> web.Response:
        del request
        return web.json_response({"ok": True})

    def _authenticate(self, request: web.Request) -> MiniAppUser:
        init_data = request.headers.get("X-Telegram-Init-Data", "")
        try:
            return validate_init_data(
                init_data,
                self._bot_token,
                max_age_seconds=self._auth_max_age_seconds,
            )
        except MiniAppAuthError as error:
            raise web.HTTPUnauthorized(
                text=str(error),
                headers={"Cache-Control": "no-store"},
            ) from error

    async def _get_profile(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        async with self._user_states.use(user.id) as state:
            payload = self._profile(state, user)
        return web.json_response(payload, headers={"Cache-Control": "no-store"})

    async def _update_profile(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        try:
            payload = await request.json()
        except (ValueError, TypeError) as error:
            raise web.HTTPBadRequest(text="Нужен JSON-объект") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="Нужен JSON-объект")

        allowed = {
            "content_mode",
            "roleplay_active",
            "roleplay_configuration",
            "roleplay_character",
            "roleplay_preferences",
            "roleplay_boundaries",
        }
        if set(payload) - allowed:
            raise web.HTTPBadRequest(text="Переданы неизвестные настройки")
        content_mode = payload.get("content_mode")
        if content_mode is not None and (
            not isinstance(content_mode, str) or content_mode not in {"soft", "adult"}
        ):
            raise web.HTTPBadRequest(text="Некорректный возрастной режим")
        roleplay_active = payload.get("roleplay_active")
        if roleplay_active is not None and type(roleplay_active) is not bool:
            raise web.HTTPBadRequest(text="Некорректный RP-режим")
        configuration = payload.get("roleplay_configuration")
        if configuration is not None and (
            not isinstance(configuration, str)
            or configuration not in {"male", "female"}
        ):
            raise web.HTTPBadRequest(text="Некорректная конфигурация Дельты")
        character = payload.get("roleplay_character")
        if character is not None and (
            not isinstance(character, str) or len(character) > _MAX_PROFILE_FIELD_CHARS
        ):
            raise web.HTTPBadRequest(text="Описание персонажа слишком длинное")
        preferences = payload.get("roleplay_preferences")
        if preferences is not None and (
            not isinstance(preferences, str)
            or len(preferences) > _MAX_PROFILE_FIELD_CHARS
        ):
            raise web.HTTPBadRequest(text="Предпочтения слишком длинные")
        boundaries = payload.get("roleplay_boundaries")
        if boundaries is not None and (
            not isinstance(boundaries, str)
            or len(boundaries) > _MAX_PROFILE_FIELD_CHARS
        ):
            raise web.HTTPBadRequest(text="Границы слишком длинные")

        async with self._user_states.use(user.id) as state:
            if content_mode is not None:
                state.content_mode = cast(ContentMode, content_mode)
            if roleplay_active is not None:
                state.roleplay_active = roleplay_active
            if configuration is not None:
                state.roleplay_configuration = configuration
            if character is not None:
                state.roleplay_character = character.strip()
            if preferences is not None:
                state.roleplay_preferences = preferences.strip()
            if boundaries is not None:
                state.roleplay_boundaries = boundaries.strip()
            response = self._profile(state, user)
        return web.json_response(response, headers={"Cache-Control": "no-store"})

    async def _delete_roleplay_profile(self, request: web.Request) -> web.Response:
        """Удалить RP-профиль пользователя, сохранив прочую память и настройки."""
        user = self._authenticate(request)
        async with self._user_states.use(user.id) as state:
            state.roleplay_active = False
            state.roleplay_configuration = "male"
            state.roleplay_character = ""
            state.roleplay_fetishes = ()
            state.roleplay_preferences = ""
            state.roleplay_boundaries = ""
            state.emotions.arousal = 0.0
            response = self._profile(state, user)
        return web.json_response(response, headers={"Cache-Control": "no-store"})

    def _profile(self, state: UserState, user: MiniAppUser) -> dict[str, Any]:
        return _profile(
            state, user, appearance_upload_enabled=self._response_engine is not None
        )

    async def _upload_appearance(self, request: web.Request) -> web.Response:
        """Принять один файл только от подписанного владельца профиля."""
        user = self._authenticate(request)
        if self._response_engine is None:
            raise web.HTTPServiceUnavailable(text="Анализ облика сейчас недоступен.")
        if (
            request.content_length is not None
            and request.content_length > MAX_APPEARANCE_BYTES
        ):
            raise web.HTTPRequestEntityTooLarge(
                max_size=MAX_APPEARANCE_BYTES,
                actual_size=request.content_length,
                text="Картинка должна быть до 20 МБ.",
            )
        if user.id in self._appearance_pending:
            raise web.HTTPConflict(text="Картинка уже обрабатывается. Подожди немного.")
        self._appearance_pending.add(user.id)
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._appearance_slots.acquire(), timeout=0.1)
                acquired = True
            except TimeoutError as error:
                raise web.HTTPTooManyRequests(
                    text="Сейчас обрабатываю другие картинки. Попробуй чуть позже."
                ) from error
            async with asyncio.timeout(60):
                data = await request.clone(
                    client_max_size=MAX_APPEARANCE_BYTES + 1
                ).read()
                try:
                    image = await asyncio.to_thread(prepare_appearance_image, data)
                except ValueError as error:
                    raise web.HTTPBadRequest(text=str(error)) from error
                if not self._appearance_limiter.allow(user.id):
                    raise web.HTTPTooManyRequests(
                        text="Подожди 10 секунд перед следующей картинкой.",
                        headers={"Retry-After": "10"},
                    )
                await self._response_engine.set_delta_appearance_from_image(
                    user.id, image
                )
                async with self._user_states.use(user.id) as state:
                    payload = self._profile(state, user)
                return web.json_response(payload, headers={"Cache-Control": "no-store"})
        except ResponseBusyError as error:
            raise web.HTTPConflict(
                text="Бот ещё отвечает в чате. Дождись ответа и попробуй снова."
            ) from error
        except AppearanceAnalysisError as error:
            raise web.HTTPBadGateway(
                text="Не получилось разобрать облик. Прежний облик сохранён; попробуй ещё раз."
            ) from error
        except TimeoutError as error:
            raise web.HTTPGatewayTimeout(
                text="Обработка заняла слишком много времени. Обнови карточку облика и попробуй снова."
            ) from error
        finally:
            self._appearance_pending.discard(user.id)
            if acquired:
                self._appearance_slots.release()

    async def _reset_appearance(self, request: web.Request) -> web.Response:
        """Вернуть базовую внешность, сохранив сцену, персонажа и настройки."""
        user = self._authenticate(request)
        if user.id in self._appearance_pending:
            raise web.HTTPConflict(
                text="Дождись обработки картинки перед сбросом облика."
            )
        async with self._user_states.use(user.id) as state:
            state.delta_appearance = ""
            payload = self._profile(state, user)
        return web.json_response(payload, headers={"Cache-Control": "no-store"})
