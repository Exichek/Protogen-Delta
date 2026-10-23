"""Воспроизведения ревизии: сброс, команды, инициатива, графики и длинное видео."""

import asyncio
import hashlib
import io
import json
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import av
import pytest
from aiogram import Bot, Dispatcher
from aiogram.enums import MessageOriginType
from aiogram.types import (
    Chat,
    Message,
    MessageEntity,
    MessageOriginUser,
    Update,
    User,
)
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request
from test_miniapp import TOKEN, _signed_init_data
from test_prompt_assembly import make_engine
from test_response_engine import _create_engine

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.config.settings import Settings
from protogen_delta.core.input_operations import InputOperations
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.creator import create_creator_router
from protogen_delta.handlers.memory import create_memory_router
from protogen_delta.handlers.rp import create_rp_router
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.initiative_followups import (
    InitiativeFollowupsRepository,
)
from protogen_delta.repositories.memories import MemoriesRepository
from protogen_delta.repositories.user_facts import FactChange, UserFactsRepository
from protogen_delta.repositories.user_state import UserStateRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.animation_frames import extract_animation_frames
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.market_data import MarketDataService
from protogen_delta.services.market_http import MarketHTTP
from protogen_delta.services.market_models import MarketDataError
from protogen_delta.services.memory import MemoryService
from protogen_delta.services.response_configuration import load_response_configuration
from protogen_delta.services.user_facts import UserFactsService


@pytest.mark.parametrize("target", ["delta", "user"])
def test_reset_cancels_appearance_upload_before_it_can_restore_profile(
    tmp_path: Path, target: str
) -> None:
    async def run() -> None:
        states = UserStateStore(persistence=UserStateRepository(tmp_path))
        engine, _, model, *_ = _create_engine()
        engine._user_states = states
        operations = InputOperations()
        started, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def prepare(*args: Any) -> tuple[ImageInput, str]:
            started.set()
            try:
                await release.wait()
            finally:
                cleaned.set()
            return ImageInput(b"synthetic", "image/jpeg"), "synthetic-thumbnail"

        work = AsyncMock()
        work.run.side_effect = prepare
        setter = AsyncMock()
        setattr(
            engine,
            (
                "set_delta_appearance_from_image"
                if target == "delta"
                else "set_roleplay_character_from_image"
            ),
            setter,
        )
        server = MiniAppServer(
            TOKEN,
            states,
            response_engine=engine,
            native_work=work,
            input_operations=operations,
        )
        route = (
            "/api/profile/appearance"
            if target == "delta"
            else "/api/profile/character/appearance"
        )
        async with TestClient(TestServer(server.application())) as client:

            async def upload() -> Any:
                return await client.post(
                    route,
                    headers={"X-Telegram-Init-Data": _signed_init_data()},
                    data=b"synthetic",
                )

            task = asyncio.create_task(upload())
            try:
                await asyncio.wait_for(started.wait(), 2)

                async def cleanup() -> None:
                    async with operations.resetting(42):
                        pass

                await asyncio.wait_for(engine.reset_user(42, cleanup=cleanup), 2)
            finally:
                release.set()
            response = await asyncio.wait_for(task, 2)
            assert response.status == 409
            assert "очищена" in await response.text()
            setter.assert_not_awaited()
            model.chat.assert_not_awaited()
            assert cleaned.is_set() and not server._appearance_pending
            assert not operations._users
            restored = UserStateStore(persistence=UserStateRepository(tmp_path))
            async with restored.use(42) as state:
                assert not state.delta_appearance
                assert not state.delta_appearance_thumbnail
                assert not state.roleplay_character

    asyncio.run(run())


@pytest.mark.parametrize("field", ["city", None])
def test_missing_field_forget_keeps_unrelated_plan_but_full_forget_removes_it(
    tmp_path: Path, field: str | None
) -> None:
    async def run() -> None:
        episodes = MemoriesRepository(tmp_path)
        facts = UserFactsRepository(tmp_path)
        followups = InitiativeFollowupsRepository(tmp_path)
        now = datetime(2026, 10, 19, 10, tzinfo=timezone.utc).timestamp()
        await episodes.note_user_activity(42, now)
        await followups.observe(42, ("Завтра у меня экзамен",), now)
        memory = MemoryService(
            episodes,
            facts=UserFactsService(facts, AsyncMock()),
            followups=followups,
        )
        result = await memory.forget_facts(42, field)
        assert result.changed == 0
        with closing(followups._connect()) as db:
            count = db.execute(
                "SELECT COUNT(*) FROM initiative_followups WHERE user_id=42"
            ).fetchone()[0]
        assert count == (1 if field is not None else 0)

    asyncio.run(run())


@pytest.mark.parametrize("command", ["memory", "rp", "message"])
@pytest.mark.parametrize("forwarded", [False, True])
def test_state_changing_commands_require_own_message(
    tmp_path: Path, command: str, forwarded: bool
) -> None:
    async def run() -> None:
        facts = UserFactsRepository(tmp_path)
        await facts.apply(
            42, [FactChange("city", "add", "Тестоград", "Я живу в Тестограде")], 100
        )
        engine, *_ = _create_engine()
        states = engine._user_states
        states.get(42).roleplay_active = True
        bot = Bot(TOKEN)
        bot.session.make_request = AsyncMock()  # type: ignore[method-assign]
        dispatcher = Dispatcher()
        memory = MemoryService(
            MemoriesRepository(tmp_path), facts=UserFactsService(facts, AsyncMock())
        )
        if command == "memory":
            router = create_memory_router(memory, states)
            text = "/memory forget city"
        elif command == "rp":
            router = create_rp_router(engine)
            text = "/rp off"
        else:
            router = create_creator_router(
                bot=bot, users_repository=UsersRepository(tmp_path), creator_id=42
            )
            text = "/message 99 Синтетическое сообщение"
        dispatcher.include_router(router)
        now = datetime.now(timezone.utc)
        message = Message(
            message_id=1,
            date=now,
            chat=Chat(id=42, type="private"),
            from_user=User(id=42, is_bot=False, first_name="Synthetic"),
            text=text,
            entities=[
                MessageEntity(type="bot_command", offset=0, length=len(command) + 1)
            ],
            forward_origin=(
                MessageOriginUser(
                    type=MessageOriginType.USER,
                    date=now,
                    sender_user=User(id=7, is_bot=False, first_name="Forwarded"),
                )
                if forwarded
                else None
            ),
        )
        try:
            await dispatcher.feed_update(bot, Update(update_id=1, message=message))
            if command == "memory":
                assert bool(await facts.all(42)) is forwarded
            elif command == "rp":
                assert states.get(42).roleplay_active is forwarded
            else:
                recipients = [
                    call.args[1].chat_id
                    for call in bot.session.make_request.await_args_list
                ]
                assert (99 in recipients) is not forwarded
        finally:
            await bot.session.close()

    asyncio.run(run())


@pytest.mark.parametrize("lively", [False, True])
def test_initiative_prompt_describes_active_scheduler(lively: bool) -> None:
    settings = Settings("synthetic", "synthetic", -1, proactive_lively_enabled=lively)
    config = load_response_configuration(settings, fetish_triggers={}, fetish_names={})
    prompt = config.system_prompt
    if lively:
        assert "минимум 4 часа" in prompt
        assert "максимум 2" in prompt
        assert "без ответа повторов нет" in prompt
        assert "через 20 часов" not in prompt
    else:
        assert "через 20 часов" in prompt
        assert "одно последнее повторное сообщение" in prompt


def test_prompt_update_preserves_all_text_outside_initiative_description() -> None:
    """Сохранить исторические SHA: меняется только технический блок расписания."""
    folder = Path(__file__).parent / "fixtures"
    old = json.loads((folder / "behavior_prompt_baseline.json").read_text("utf-8"))
    current = json.loads((folder / "behavior_prompt_current.json").read_text("utf-8"))
    adult_previous = json.loads(
        (folder / "adult_prompt_update_baseline.json").read_text("utf-8")
    )["sections"]
    assert [case["arguments"] for case in old["cases"]] == [
        case["arguments"] for case in current["cases"]
    ]
    settings = Settings("", "", 0)
    old_section = (
        (folder / "initiative_system_before_audit.txt")
        .read_text("utf-8")
        .strip()
        .format(
            minimum_hours=settings.proactive_idle_seconds / 3600,
            maximum_hours=settings.proactive_idle_max_seconds / 3600,
            followup_hours=settings.proactive_cooldown_seconds / 3600,
        )
    )
    engine = make_engine()
    prefix = load_prompt("personality/core.txt") + "\n\n"
    suffix = "\n\n" + load_prompt("emotional_system")
    core = engine._config.system_prompt
    assert core.startswith(prefix) and core.endswith(suffix)
    start, end = len(prefix), len(core) - len(suffix)
    new_section = core[start:end]
    for case in old["cases"]:
        prompt = engine._build_prompt(**case["arguments"])
        assert prompt.count(new_section) == 1
        reconstructed = prompt.replace(new_section, old_section, 1)
        # Пользователь отдельно заменил эти два файла. Для исторического SHA
        # вернуть именно прежние секции, не переписывая исходный эталон.
        for name, previous in adult_previous.items():
            section = load_prompt(name)
            if section in reconstructed:
                assert reconstructed.count(section) == 1
                reconstructed = reconstructed.replace(section, previous, 1)
        assert len(reconstructed) == case["chars"]
        assert hashlib.sha256(reconstructed.encode()).hexdigest() == case["sha256"]


@pytest.mark.parametrize(
    "market,symbol", [("us", "SPY"), ("ru", "SBER"), ("crypto", "BTC/USD")]
)
def test_chart_reuses_same_snapshot_and_refreshes_after_ttl(
    monkeypatch: pytest.MonkeyPatch, market: str, symbol: str
) -> None:
    calls: list[dict[str, str]] = []
    now = [100.0]

    class Content:
        async def iter_chunked(self, size: int) -> Any:
            yield json.dumps({"bars": {symbol: []}}).encode()

    class Response:
        status = 200
        content = Content()

        async def __aenter__(self) -> "Response":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    class Session:
        async def __aenter__(self) -> "Session":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        def get(self, url: str, **kwargs: Any) -> Response:
            calls.append(kwargs["params"])
            return Response()

    monkeypatch.setattr(
        "protogen_delta.services.market_http._session", lambda _: Session()
    )
    monkeypatch.setattr(
        "protogen_delta.services.market_data.monotonic", lambda: now[0], raising=False
    )
    monkeypatch.setattr("protogen_delta.services.market_http.monotonic", lambda: now[0])

    async def run() -> None:
        http = MarketHTTP("synthetic", "synthetic")
        data = MarketDataService(http)
        try:
            first = await data.chart(market, symbol)
            snapshots = await asyncio.gather(
                *(data.chart(market, symbol) for _ in range(5))
            )
            assert len(calls) == 1
            assert all(item == first for item in snapshots)
            first["candles"].append({"corrupted": True})
            assert not (await data.chart(market, symbol))["candles"]
            now[0] += 61
            refreshed = await data.chart(market, symbol)
            assert len(calls) == 2
            assert refreshed["checked_at"] != snapshots[0]["checked_at"]
        finally:
            await data.close()

    asyncio.run(run())


def test_long_video_sampling_reaches_late_action() -> None:
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(6000):
            frame = av.VideoFrame(16, 16, "rgb24")
            frame.planes[0].update(b"\x00" * frame.planes[0].buffer_size)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    frames = extract_animation_frames(output.getvalue(), label="длинное видео")
    assert len(frames) == 4
    assert "150.00 с" in frames[-1].label


def test_chart_shares_inflight_load_and_preserves_it_when_one_client_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def load(*args: Any) -> dict[str, Any]:
            started.set()
            await release.wait()
            return {"candles": [], "checked_at": "source-time"}

        loader = AsyncMock(side_effect=load)
        monkeypatch.setattr("protogen_delta.services.market_data.read_chart", loader)
        data = MarketDataService(MarketHTTP())
        first = asyncio.create_task(data.chart("us", "spy"))
        await started.wait()
        second = asyncio.create_task(data.chart("us", "SPY"))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert (await second)["checked_at"] == "source-time"
        loader.assert_awaited_once()
        assert not data._chart_pending
        await data.close()
        with pytest.raises(MarketDataError):
            await data.chart("us", "SPY")

    asyncio.run(run())


def test_chart_retries_failed_load_and_bounds_cached_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        loader = AsyncMock(side_effect=MarketDataError("synthetic"))
        monkeypatch.setattr("protogen_delta.services.market_data.read_chart", loader)
        data = MarketDataService(MarketHTTP())
        with pytest.raises(MarketDataError):
            await data.chart("us", "SPY")
        assert not data._chart_pending and not data._charts
        loader.side_effect = None
        loader.return_value = {"candles": []}
        for number in range(128):
            await data.chart("us", f"S{number}")
        await data.chart("us", "S0")  # Сохранить недавно просмотренный снимок.
        await data.chart("us", "SPY")
        assert len(data._charts) == 128
        assert ("us", "S0", "1m", "1d") in data._charts
        assert ("us", "S1", "1m", "1d") not in data._charts
        await data.close()
        assert not data._charts

    asyncio.run(run())


def test_chart_limits_distinct_inflight_requests_and_closes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        waiting = asyncio.Event()

        async def load(*args: Any) -> dict[str, Any]:
            await waiting.wait()
            return {}

        monkeypatch.setattr("protogen_delta.services.market_data.read_chart", load)
        data = MarketDataService(MarketHTTP())
        tasks = [asyncio.create_task(data.chart("us", f"S{i}")) for i in range(32)]
        await asyncio.sleep(0)
        with pytest.raises(MarketDataError, match="заняты"):
            await data.chart("us", "SPY")
        await data.close()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert not data._chart_pending and not data._charts

    asyncio.run(run())


def test_appearance_shutdown_cancellation_is_not_misreported_as_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        operations = InputOperations()
        server = MiniAppServer(TOKEN, UserStateStore(), input_operations=operations)
        started = asyncio.Event()

        async def upload(*args: Any) -> Any:
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            server, "_read_appearance_upload", AsyncMock(side_effect=upload)
        )
        request = make_mocked_request(
            "POST",
            "/api/profile/appearance",
            headers={"X-Telegram-Init-Data": _signed_init_data()},
        )
        task = asyncio.create_task(server._upload_appearance(request, target="delta"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not operations._users

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode", ["unknown", "partial", "seek-failure", "seek-partial", "no-video"]
)
def test_video_limits_decode_work_and_marks_incomplete_sampling(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from fractions import Fraction

    from protogen_delta.services import animation_frames as frames

    decoded = [0]
    opened = [0]
    seeks = [0]
    stream = SimpleNamespace(
        codec_context=SimpleNamespace(thread_count=0),
        duration=None if mode == "unknown" else 200,
        time_base=Fraction(1),
        frames=0 if mode == "unknown" else 6000,
        average_rate=30,
        start_time=0,
    )

    class Container:
        streams = SimpleNamespace(video=[] if mode == "no-video" else [stream])
        duration = None
        format = SimpleNamespace(name="mp4" if mode.startswith("seek-") else "gif")

        def __enter__(self) -> "Container":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def decode(self, *args: Any) -> Any:
            for index in range(6000):
                decoded[0] += 1
                yield SimpleNamespace(time=index / 30)

        def seek(self, *args: Any, **kwargs: Any) -> None:
            seeks[0] += 1
            if mode == "seek-failure" and seeks[0] == 2:
                raise ValueError("synthetic unseekable stream")

    def open_container(*args: Any, **kwargs: Any) -> Any:
        opened[0] += 1
        return Container()

    monkeypatch.setattr(frames.av, "open", open_container)
    monkeypatch.setattr(frames, "MAX_DECODED_FRAMES", 8)
    monkeypatch.setattr(frames, "_encode_png", lambda _: b"synthetic-png")
    result = frames.extract_animation_frames(b"synthetic", label="видео")
    assert decoded[0] <= 8
    if mode == "no-video":
        assert not result and decoded[0] == 0
    else:
        assert result
        assert all(
            ("Длительность неизвестна" if mode == "unknown" else "Выборка неполная")
            in frame.label
            for frame in result
        )
    assert opened[0] == (2 if mode == "seek-failure" else 1)
