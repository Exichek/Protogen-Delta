"""Встроенный HTTP-сервер панели настроек Telegram Mini App."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from importlib.resources import files
from typing import Any, Literal, cast
from urllib.parse import quote, unquote

from aiogram.exceptions import TelegramAPIError
from aiohttp import web

from protogen_delta.core.appearance_profile import AppearanceProfile
from protogen_delta.core.appearance_species import AppearanceSpecies
from protogen_delta.core.conversation_safety import (
    reference_is_child,
    validate_stopword,
)
from protogen_delta.core.llm_budget import LlmBudgetError, llm_user
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.core.runtime_health import RuntimeHealth
from protogen_delta.core.telegram_commands import ModeChange
from protogen_delta.core.user_state import ContentMode, UserState, UserStateStore
from protogen_delta.miniapp.auth import (
    MiniAppAuthError,
    MiniAppUser,
    validate_init_data,
)
from protogen_delta.miniapp.tools import MiniAppTools, ToolsBusyError
from protogen_delta.repositories.llm_usage import LlmUsageRepository
from protogen_delta.services.appearance_analysis import (
    validate_reference_notes,
    validate_species_hint,
)
from protogen_delta.services.appearance_description import parse_description_file
from protogen_delta.services.appearance_image import (
    MAX_APPEARANCE_BYTES,
    prepare_appearance_upload,
)
from protogen_delta.services.blocking_work import WorkRunner
from protogen_delta.services.fetishes import FetishTriggers, detect_fetishes
from protogen_delta.services.media_conversion import (
    MIME_TYPES,
    ConversionBusyError,
    ConversionError,
    MediaConversionService,
)
from protogen_delta.services.media_conversion_worker import MAX_INPUT, validate_options
from protogen_delta.services.native_work import NativeWorkPool
from protogen_delta.services.response_engine import (
    AppearanceAnalysisError,
    ResponseBusyError,
    ResponseEngine,
)
from protogen_delta.services.translation import (
    MAX_TRANSLATION_OUTPUT_CHARS,
    TranslationError,
    TranslationService,
)
from protogen_delta.services.translation_document import make_translated_docx

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
        "age_restricted": state.age_restricted,
        "delta_reference_restricted": state.delta_reference_restricted,
        "roleplay_stopword": state.roleplay_stopword,
        "roleplay_active": state.roleplay_active,
        "roleplay_configuration": state.roleplay_configuration,
        "roleplay_character": state.roleplay_character,
        "roleplay_character_thumbnail": (
            state.roleplay_character_thumbnail if state.roleplay_character else ""
        ),
        "roleplay_character_restricted": state.roleplay_character_restricted,
        "delta_appearance_profile": (
            asdict(
                AppearanceProfile.decode(state.delta_appearance_profile)
                or AppearanceProfile()
            )
            if state.delta_appearance
            else asdict(AppearanceProfile())
        ),
        "delta_appearance": state.delta_appearance,
        "delta_appearance_thumbnail": (
            state.delta_appearance_thumbnail if state.delta_appearance else ""
        ),
        "delta_species": (
            asdict(species)
            if (species := AppearanceSpecies.decode(state.delta_species))
            else None
        ),
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
        on_mode_change: ModeChange | None = None,
        tools: MiniAppTools | None = None,
        native_work: WorkRunner | None = None,
        llm_usage: LlmUsageRepository | None = None,
        runtime_health: RuntimeHealth | None = None,
        fetish_triggers: FetishTriggers | None = None,
        fetish_names: dict[str, str] | None = None,
        media_converter: MediaConversionService | None = None,
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
        self._on_mode_change = on_mode_change
        self._tools = tools
        self._native_work = native_work or NativeWorkPool(2)
        self._runtime_health = runtime_health
        self._llm_usage = llm_usage
        self._fetish_triggers = fetish_triggers or {}
        self._fetish_names = fetish_names or {}
        self._appearance_pending: set[int] = set()
        self._appearance_slots = asyncio.Semaphore(2)
        self._appearance_limiter = UserRateLimiter(cooldown_seconds=10)
        self._translation_export_slots = asyncio.Semaphore(2)
        self._translation_export_limiter = UserRateLimiter(cooldown_seconds=2)
        self._runner: web.AppRunner | None = None
        self._media_converter = media_converter or MediaConversionService()
        if self._runtime_health is not None:
            self._runtime_health.add_diagnostics(
                "converter", self._media_converter.snapshot
            )

    def application(self) -> web.Application:
        """Создать aiohttp-приложение для запуска и тестирования."""
        application = web.Application(
            client_max_size=64 * 1024, middlewares=[self._llm_scope]
        )
        application.add_routes(
            [
                web.get("/", self._index),
                web.get("/static/app.js", self._javascript),
                web.get("/api/profile", self._get_profile),
                web.get("/api/usage", self._get_usage),
                web.patch("/api/profile", self._update_profile),
                web.delete("/api/profile/roleplay", self._delete_roleplay_profile),
                web.post("/api/profile/appearance", self._upload_appearance),
                web.put("/api/profile/appearance", self._set_text_appearance),
                web.patch("/api/profile/appearance", self._set_text_appearance),
                web.patch(
                    "/api/profile/appearance/species", self._set_appearance_species
                ),
                web.delete("/api/profile/appearance", self._reset_appearance),
                web.patch(
                    "/api/profile/appearance/behavior", self._set_appearance_profile
                ),
                web.post(
                    "/api/profile/character/appearance",
                    self._upload_character_appearance,
                ),
                web.get("/health", self._health),
                web.post("/api/tools/{tool}", self._run_tool),
                web.post("/api/tools/translate/file", self._translate_file),
                web.post("/api/tools/translate/export", self._export_translation),
                web.post("/api/tools/convert/file", self._convert_media),
                web.delete("/api/tools/convert/{token}", self._delete_conversion),
                web.get("/converted-media/{token}/{kind}", self._conversion_file),
            ]
        )
        application.on_startup.append(self._start_converter)
        application.on_cleanup.append(self._close_converter)
        return application

    async def _start_converter(self, application: web.Application) -> None:
        await self._media_converter.start()

    async def _close_converter(self, application: web.Application) -> None:
        await self._media_converter.close()

    async def _convert_media(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        try:
            raw = request.headers.get("X-Conversion-Options", "")
            if len(raw) > 3000:
                raise ValueError("Слишком длинные параметры.")
            options = json.loads(unquote(raw, errors="strict"))
            if not isinstance(options, dict) or set(options) - {
                "mode",
                "format",
                "resolution",
                "bitrate",
                "file_name",
            }:
                raise ValueError("Некорректные параметры.")
            mode, extension = options.get("mode"), options.get("format")
            resolution, bitrate = options.get("resolution", "720"), options.get(
                "bitrate", 192
            )
            name = options.get("file_name", "файл")
            if (
                not all(
                    isinstance(value, str)
                    for value in (mode, extension, resolution, name)
                )
                or len(name) > 160
            ):
                raise ValueError("Проверь параметры файла.")
            mode, extension = cast(str, mode), cast(str, extension)
            validate_options(mode, extension, resolution, bitrate)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise web.HTTPBadRequest(
                text="Выбери файл и поддерживаемый формат."
            ) from error
        if request.content_length is not None and request.content_length > MAX_INPUT:
            raise web.HTTPRequestEntityTooLarge(
                max_size=MAX_INPUT,
                actual_size=request.content_length,
                text="Файл должен быть до 100 МБ.",
            )

        async def read_file() -> bytes:
            return await request.clone(client_max_size=MAX_INPUT + 1).read()

        try:
            job = await self._media_converter.convert(
                user.id,
                read_file,
                name=name,
                mode=mode,
                extension=extension,
                resolution=resolution,
                bitrate=bitrate,
            )
        except ConversionBusyError as error:
            raise web.HTTPTooManyRequests(text=str(error)) from error
        except (ConversionError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        prefix = "/converted-media/" + job.token
        return web.json_response(
            {
                "token": job.token,
                "name": job.name,
                "format": job.extension,
                "mode": mode,
                "bytes": job.size,
                "duration": job.duration,
                "expires_in": 900,
                "download_url": prefix + "/download",
                "preview_url": prefix + "/preview",
            },
            headers={"Cache-Control": "no-store"},
        )

    async def _delete_conversion(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        if not await self._media_converter.remove(request.match_info["token"], user.id):
            raise web.HTTPNotFound(text="Результат уже удалён.")
        return web.json_response(
            {"deleted": True}, headers={"Cache-Control": "no-store"}
        )

    async def _conversion_file(self, request: web.Request) -> web.FileResponse:
        job = self._media_converter.get(request.match_info["token"])
        kind = request.match_info["kind"]
        if job is None or kind not in {"download", "preview"}:
            raise web.HTTPNotFound(
                text="Результат устарел. Преобразуй файл ещё раз.",
                headers={"Cache-Control": "no-store"},
            )
        path = job.path if kind == "download" else job.preview
        disposition = (
            "attachment; filename*=UTF-8''" + quote(job.name)
            if kind == "download"
            else "inline"
        )
        return web.FileResponse(
            path,
            headers={
                "Content-Type": MIME_TYPES[path.suffix[1:]],
                "Content-Disposition": disposition,
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": "default-src 'none'; sandbox",
            },
        )

    @web.middleware
    async def _llm_scope(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        user_id = (
            self._authenticate(request).id if request.path.startswith("/api/") else None
        )
        with llm_user(user_id):
            try:
                return await handler(request)
            except LlmBudgetError as error:
                raise web.HTTPTooManyRequests(
                    text=str(error), headers={"Cache-Control": "no-store"}
                ) from error

    async def _get_usage(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        if self._llm_usage is None:
            raise web.HTTPServiceUnavailable(text="Учёт запросов сейчас недоступен")
        return web.json_response(
            await self._llm_usage.get(user.id), headers={"Cache-Control": "no-store"}
        )

    async def start(self) -> None:
        """Запустить HTTP-сервер."""
        if self._runner is not None:
            return
        runner = web.AppRunner(self.application(), access_log=None)
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
                    "default-src 'self'; script-src 'self' https://telegram.org; "
                    "style-src 'self' 'unsafe-inline'; "
                    "connect-src 'self'; img-src 'self' data: blob: https:; "
                    "object-src 'none'; base-uri 'none'"
                ),
            },
        )

    async def _javascript(self, request: web.Request) -> web.Response:
        del request
        return web.Response(
            text=files("protogen_delta.miniapp.static")
            .joinpath("app.js")
            .read_text(encoding="utf-8"),
            content_type="application/javascript",
            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    async def _health(self, request: web.Request) -> web.Response:
        del request
        if self._runtime_health is None:
            return web.json_response({"ok": True})
        state = self._runtime_health.snapshot()
        return web.json_response(state, status=200 if state["ok"] else 503)

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
            "expected_character",
            "roleplay_preferences",
            "roleplay_boundaries",
            "roleplay_stopword",
        }
        if set(payload) - allowed:
            raise web.HTTPBadRequest(text="Переданы неизвестные настройки")
        content_mode = payload.get("content_mode")
        stopword = payload.get("roleplay_stopword")
        if stopword is not None:
            try:
                if not isinstance(stopword, str):
                    raise ValueError("Стоп-слово должно быть текстом.")
                stopword = validate_stopword(stopword)
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error)) from error
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
            not isinstance(character, str) or len(character) > 2000
        ):
            raise web.HTTPBadRequest(text="Описание персонажа слишком длинное")
        expected_character = payload.get("expected_character")
        if expected_character is not None and (
            not isinstance(expected_character, str) or len(expected_character) > 2000
        ):
            raise web.HTTPBadRequest(text="Некорректная исходная версия персонажа")
        if character is not None and user.id in self._appearance_pending:
            raise web.HTTPConflict(
                text="Дождись обработки картинки перед изменением персонажа."
            )
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
            if content_mode == "adult" and state.age_restricted:
                raise web.HTTPForbidden(
                    text="Ты сообщил, что тебе нет 18. Доступен обычный режим общения."
                )
            if character is not None and (
                (
                    expected_character is not None
                    and expected_character != state.roleplay_character
                )
                or (state.roleplay_character_thumbnail and expected_character is None)
            ):
                raise web.HTTPConflict(
                    text="Персонаж уже изменился. Открой панель заново."
                )
            previous_mode = state.content_mode
            if stopword is not None:
                state.roleplay_stopword = stopword
            if content_mode is not None:
                state.content_mode = cast(ContentMode, content_mode)
                if content_mode == "soft":
                    state.stop_roleplay()
            if roleplay_active is not None:
                if roleplay_active:
                    state.roleplay_active = True
                else:
                    state.stop_roleplay()
            if configuration is not None:
                state.roleplay_configuration = configuration
            if character is not None:
                state.roleplay_character = character.strip()
                if not state.roleplay_character:
                    state.roleplay_character_thumbnail = ""
                    state.roleplay_character_restricted = False
                elif reference_is_child(state.roleplay_character):
                    state.roleplay_character_restricted = True
                if state.roleplay_character_restricted or reference_is_child(
                    state.roleplay_character
                ):
                    state.stop_roleplay()
            if preferences is not None:
                state.roleplay_preferences = preferences.strip()
            if boundaries is not None:
                state.roleplay_boundaries = boundaries.strip()
            response = self._profile(state, user)
            mode = state.content_mode
        if mode != previous_mode and self._on_mode_change is not None:
            await self._on_mode_change(user.id, mode)
        return web.json_response(response, headers={"Cache-Control": "no-store"})

    async def _delete_roleplay_profile(self, request: web.Request) -> web.Response:
        """Удалить RP-профиль пользователя, сохранив прочую память и настройки."""
        user = self._authenticate(request)
        async with self._user_states.use(user.id) as state:
            state.stop_roleplay()
            state.roleplay_configuration = "male"
            state.roleplay_character = ""
            state.roleplay_character_thumbnail = ""
            state.roleplay_character_restricted = False
            state.roleplay_fetishes = ()
            state.roleplay_preferences = ""
            state.roleplay_boundaries = ""
            state.emotions.arousal = 0.0
            response = self._profile(state, user)
        return web.json_response(response, headers={"Cache-Control": "no-store"})

    def _profile(self, state: UserState, user: MiniAppUser) -> dict[str, Any]:
        response = _profile(
            state, user, appearance_upload_enabled=self._response_engine is not None
        )
        appearance_profile = cast(dict[str, str], response["delta_appearance_profile"])
        response["delta_appearance_topics"] = [
            {"key": key, "name": self._fetish_names.get(key, key)}
            for key in detect_fetishes(
                appearance_profile["fetishes"], self._fetish_triggers
            )
        ]
        return response

    async def _upload_character_appearance(self, request: web.Request) -> web.Response:
        return await self._upload_appearance(request, target="user")

    async def _upload_appearance(
        self, request: web.Request, *, target: Literal["delta", "user"] = "delta"
    ) -> web.Response:
        """Принять один файл только от подписанного владельца профиля."""
        user = self._authenticate(request)
        if self._response_engine is None:
            raise web.HTTPServiceUnavailable(text="Анализ облика сейчас недоступен.")
        options: dict[str, Any] = {}
        raw_options = request.headers.get("X-Appearance-Options", "")
        if raw_options:
            try:
                if len(raw_options) > 7000:
                    raise ValueError("Слишком длинные параметры облика.")
                options = json.loads(unquote(raw_options, errors="strict"))
                if not isinstance(options, dict) or set(options) - {
                    "species",
                    "update_existing",
                    "notes",
                }:
                    raise ValueError("Некорректные параметры облика.")
                if (
                    not isinstance(options.get("species", ""), str)
                    or type(options.get("update_existing", False)) is not bool
                ):
                    raise ValueError("Некорректные параметры облика.")
                options["species"] = validate_species_hint(options.get("species", ""))
                options["notes"] = validate_reference_notes(options.get("notes", ""))
            except (ValueError, UnicodeError, RecursionError) as error:
                raise web.HTTPBadRequest(
                    text="Проверь вид, комментарий (до 400 символов) и режим обновления."
                ) from error
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
            async with asyncio.timeout(150):
                data = await request.clone(
                    client_max_size=MAX_APPEARANCE_BYTES + 1
                ).read()
                try:
                    image, thumbnail = await self._native_work.run(
                        prepare_appearance_upload, data
                    )
                except ValueError as error:
                    raise web.HTTPBadRequest(text=str(error)) from error
                if not self._appearance_limiter.allow(user.id):
                    raise web.HTTPTooManyRequests(
                        text="Подожди 10 секунд перед следующей картинкой.",
                        headers={"Retry-After": "10"},
                    )
                if target == "user":
                    await self._response_engine.set_roleplay_character_from_image(
                        user.id,
                        image,
                        thumbnail=thumbnail,
                        species_hint=options.get("species", ""),
                        reference_notes=options.get("notes", ""),
                    )
                elif options:
                    await self._response_engine.set_delta_appearance_from_image(
                        user.id,
                        image,
                        thumbnail=thumbnail,
                        species_hint=options.get("species", ""),
                        update_existing=options.get("update_existing", False),
                        reference_notes=options.get("notes", ""),
                    )
                else:
                    await self._response_engine.set_delta_appearance_from_image(
                        user.id, image, thumbnail=thumbnail
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
            state.delta_species = ""
            state.delta_appearance_thumbnail = ""
            state.delta_appearance_profile = ""
            state.delta_reference_restricted = False
            payload = self._profile(state, user)
        return web.json_response(payload, headers={"Cache-Control": "no-store"})

    async def _set_text_appearance(self, request: web.Request) -> web.Response:
        """Применить описание или TXT/JSON только к собственному облику Дельты."""
        user = self._authenticate(request)
        if self._response_engine is None:
            raise web.HTTPServiceUnavailable(text="Смена облика сейчас недоступна.")
        if user.id in self._appearance_pending:
            raise web.HTTPConflict(text="Облик уже обрабатывается. Подожди немного.")
        self._appearance_pending.add(user.id)
        try:
            try:
                if request.method == "PATCH":
                    payload = await request.json()
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != {"appearance", "expected_appearance"}
                        or not isinstance(payload["appearance"], str)
                        or not isinstance(payload["expected_appearance"], str)
                        or not payload["expected_appearance"]
                        or len(payload["expected_appearance"]) > 2000
                    ):
                        raise ValueError("Нужны описание и его исходная версия.")
                    description = payload["appearance"]
                elif request.content_type == "application/json":
                    description = parse_description_file(
                        await request.read(), json_file=True
                    )
                elif request.content_type == "text/plain":
                    description = parse_description_file(
                        await request.read(), json_file=False
                    )
                else:
                    raise ValueError("Нужен текст, TXT или JSON.")
                await self._response_engine.set_delta_appearance_from_text(
                    user.id,
                    description,
                    **(
                        {"expected_appearance": payload["expected_appearance"]}
                        if request.method == "PATCH"
                        else {}
                    ),
                )
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error)) from error
            except ResponseBusyError as error:
                raise web.HTTPConflict(
                    text="Бот ещё отвечает в чате. Попробуй после ответа."
                ) from error
            async with self._user_states.use(user.id) as state:
                payload = self._profile(state, user)
            return web.json_response(payload, headers={"Cache-Control": "no-store"})
        finally:
            self._appearance_pending.discard(user.id)

    async def _set_appearance_profile(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        if self._response_engine is None:
            raise web.HTTPServiceUnavailable(text="Смена профиля сейчас недоступна.")
        if user.id in self._appearance_pending:
            raise web.HTTPConflict(text="Облик уже обрабатывается. Подожди немного.")
        self._appearance_pending.add(user.id)
        try:
            try:
                payload = await request.json()
                if (
                    not isinstance(payload, dict)
                    or set(payload)
                    != {"profile", "expected_appearance", "expected_profile"}
                    or not isinstance(payload["expected_appearance"], str)
                    or not 1 <= len(payload["expected_appearance"]) <= 2000
                ):
                    raise ValueError("Нужны профиль и исходная версия облика.")
                await self._response_engine.set_delta_appearance_profile(
                    user.id,
                    AppearanceProfile.from_payload(payload["profile"]),
                    expected_appearance=payload["expected_appearance"],
                    expected_profile=AppearanceProfile.from_payload(
                        payload["expected_profile"]
                    ),
                )
            except (ValueError, RecursionError) as error:
                raise web.HTTPBadRequest(text=str(error)) from error
            except ResponseBusyError as error:
                raise web.HTTPConflict(
                    text="Бот ещё отвечает. Попробуй после ответа."
                ) from error
            async with self._user_states.use(user.id) as state:
                result = self._profile(state, user)
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        finally:
            self._appearance_pending.discard(user.id)

    async def _set_appearance_species(self, request: web.Request) -> web.Response:
        """Применить вид к собственному сохранённому облику, без новой картинки."""
        user = self._authenticate(request)
        if self._response_engine is None:
            raise web.HTTPServiceUnavailable(text="Смена облика сейчас недоступна.")
        if user.id in self._appearance_pending:
            raise web.HTTPConflict(text="Облик уже обрабатывается. Подожди немного.")
        self._appearance_pending.add(user.id)
        try:
            try:
                payload = await request.json()
                if (
                    not isinstance(payload, dict)
                    or set(payload)
                    != {"species", "expected_appearance", "expected_species"}
                    or not isinstance(payload["expected_appearance"], str)
                    or not 1 <= len(payload["expected_appearance"]) <= 2000
                ):
                    raise ValueError("Нужны вид и исходная версия облика.")
                hint = validate_species_hint(payload["species"])
                expected = payload["expected_species"]
                previous = None
                if expected is not None:
                    if not isinstance(expected, dict):
                        raise ValueError("Некорректная исходная версия вида.")
                    previous = AppearanceSpecies.decode(json.dumps(expected))
                    if previous is None:
                        raise ValueError("Некорректная исходная версия вида.")
                await self._response_engine.set_delta_species(
                    user.id,
                    hint,
                    expected_appearance=payload["expected_appearance"],
                    expected_species=previous,
                )
            except (ValueError, RecursionError) as error:
                raise web.HTTPBadRequest(text=str(error)) from error
            except ResponseBusyError as error:
                raise web.HTTPConflict(
                    text="Бот ещё отвечает в чате. Попробуй после ответа."
                ) from error
            async with self._user_states.use(user.id) as state:
                result = self._profile(state, user)
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        finally:
            self._appearance_pending.discard(user.id)

    async def _translate_file(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        if self._tools is None:
            raise web.HTTPServiceUnavailable(text="Переводчик сейчас недоступен.")
        try:
            raw = request.headers.get("X-Translation-Options", "")
            if len(raw) > 3000:
                raise ValueError("Слишком длинные параметры перевода.")
            options = json.loads(unquote(raw, errors="strict"))
            if not isinstance(options, dict) or set(options) - {
                "mode",
                "source",
                "target",
                "file_name",
            }:
                raise ValueError("Некорректные параметры перевода.")
            mode = options.get("mode")
            source = options.get("source", "auto")
            target = options.get("target", "en")
            name = options.get("file_name", "document")
            if not all(
                isinstance(v, str) for v in (mode, source, target, name)
            ) or mode not in {"image", "document"}:
                raise ValueError("Выбери изображение или документ и языки перевода.")
            if len(name) > 160:
                raise ValueError("Имя файла слишком длинное.")
            TranslationService.validate_languages(source, target)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise web.HTTPBadRequest(text="Проверь файл и языки перевода.") from error
        if (
            request.content_length is not None
            and request.content_length > MAX_APPEARANCE_BYTES
        ):
            raise web.HTTPRequestEntityTooLarge(
                max_size=MAX_APPEARANCE_BYTES,
                actual_size=request.content_length,
                text="Файл должен быть до 20 МБ.",
            )

        async def read_file() -> bytes:
            async with asyncio.timeout(90):
                data = await request.clone(
                    client_max_size=MAX_APPEARANCE_BYTES + 1
                ).read()
            if len(data) > MAX_APPEARANCE_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_APPEARANCE_BYTES,
                    actual_size=len(data),
                    text="Файл должен быть до 20 МБ.",
                )
            return data

        try:
            result = await self._tools.translate_file(
                user.id, read_file, name, request.content_type, mode, source, target
            )
        except ToolsBusyError as error:
            raise web.HTTPTooManyRequests(text=str(error)) from error
        except TranslationError as error:
            raise web.HTTPBadGateway(text=str(error)) from error
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except TimeoutError as error:
            raise web.HTTPGatewayTimeout(
                text="Перевод занял слишком много времени. Раздели документ на части."
            ) from error
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    async def _export_translation(self, request: web.Request) -> web.Response:
        user = self._authenticate(request)
        if not self._translation_export_limiter.allow(user.id):
            raise web.HTTPTooManyRequests(
                text="Подожди пару секунд перед следующим скачиванием."
            )
        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._translation_export_slots.acquire(), timeout=0.1
                )
                acquired = True
            except TimeoutError as error:
                raise web.HTTPTooManyRequests(
                    text="Создание файла занято. Попробуй позже."
                ) from error
            async with asyncio.timeout(90):
                payload = await request.clone(
                    client_max_size=MAX_TRANSLATION_OUTPUT_CHARS * 12 + 1024
                ).json()
                if not isinstance(payload, dict) or set(payload) != {"text"}:
                    raise ValueError("Нужен текст перевода.")
                text = payload["text"]
                if (
                    not isinstance(text, str)
                    or not text.strip()
                    or len(text) > MAX_TRANSLATION_OUTPUT_CHARS
                ):
                    raise ValueError("Некорректный размер перевода.")
                data = await self._native_work.run(
                    make_translated_docx, text.encode("utf-8")
                )
            return web.Response(
                body=data,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={
                    "Cache-Control": "no-store",
                    "Content-Disposition": 'attachment; filename="translation.docx"',
                },
            )
        except (ValueError, UnicodeError, RecursionError) as error:
            raise web.HTTPBadRequest(
                text="Не удалось создать DOCX из этого текста."
            ) from error
        except TimeoutError as error:
            raise web.HTTPGatewayTimeout(
                text="Не удалось создать файл вовремя. Попробуй ещё раз."
            ) from error
        finally:
            if acquired:
                self._translation_export_slots.release()

    async def _run_tool(self, request: web.Request) -> web.Response:
        """Авторизовать владельца до разбора тела и запуска инструмента."""
        user = self._authenticate(request)
        if self._tools is None:
            raise web.HTTPServiceUnavailable(text="Инструменты сейчас недоступны.")
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Нужен JSON-объект.")
            tool = request.match_info["tool"]
            if tool == "id":
                target = payload.get("target", "self")
                if not isinstance(target, str):
                    raise ValueError("Некорректный выбор ID.")
                result = await self._tools.lookup_id(user.id, target)
            elif tool == "download":
                url = payload.get("url")
                if not isinstance(url, str):
                    raise ValueError("Нужна ссылка на видео.")
                result = await self._tools.download(user.id, url.strip())
            elif tool == "translate":
                text = payload.get("text")
                source = payload.get("source", "auto")
                target = payload.get("target", "en")
                if (
                    not isinstance(text, str)
                    or not isinstance(source, str)
                    or not isinstance(target, str)
                ):
                    raise ValueError("Укажи текст и языки перевода.")
                result = await self._tools.translate(user.id, text, source, target)
            else:
                raise web.HTTPNotFound(text="Такого инструмента нет.")
        except ToolsBusyError as error:
            raise web.HTTPTooManyRequests(text=str(error)) from error
        except TranslationError as error:
            raise web.HTTPBadGateway(text=str(error)) from error
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except TelegramAPIError as error:
            raise web.HTTPBadGateway(
                text="Telegram не смог определить ID. Для приватных чатов используй выбор в Telegram."
            ) from error
        return web.json_response(result, headers={"Cache-Control": "no-store"})
