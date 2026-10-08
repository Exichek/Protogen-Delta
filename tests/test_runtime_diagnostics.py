"""Метрики показывают настоящую занятость и не запускают API-проверки."""

import asyncio
import json
import threading
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import Message
from test_admin_handlers import _call_handler, _create_message_mock
from test_audio_understanding import wav_data
from test_deepseek_service import _create_response, _create_service

from protogen_delta.core.operation_metrics import OperationMetrics
from protogen_delta.core.runtime_health import RuntimeHealth
from protogen_delta.core.state import BotState
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.handlers.admin import create_admin_router
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.repositories.images import ImagesRepository
from protogen_delta.repositories.users import UsersRepository
from protogen_delta.services.audio_understanding import AudioUnderstandingService
from protogen_delta.services.blocking_work import BlockingWorkPool, WorkRunner
from protogen_delta.services.documents import extract_document
from protogen_delta.services.native_work import NativeWorkError, NativeWorkPool
from protogen_delta.services.speech import SpeechTranscriber


def test_metrics_bound_samples_measure_queue_and_keep_last_success_age() -> None:
    now = [0.0]
    metrics = OperationMetrics(capacity=2, sample_limit=2, clock=lambda: now[0])
    assert metrics.snapshot()["last_result"] == "unknown"
    assert metrics.snapshot()["recent_p95_ms"] is None
    with metrics.measure(queued=True) as task:
        assert metrics.snapshot()["waiting"] == 1
        now[0] = 1
        task.start()
        task.start()
        assert metrics.snapshot()["active"] == 1
        now[0] = 3
    with pytest.raises(RuntimeError):
        with metrics.measure():
            now[0] = 4
            raise RuntimeError("SECRET_MESSAGE")
    cancelled = metrics.measure(queued=True)
    now[0] = 5
    cancelled.finish(asyncio.CancelledError())
    cancelled.finish()
    cancelled.start()
    data = metrics.snapshot()
    assert data["completed"] == 3
    assert data["active"] == data["waiting"] == 0
    assert data["failed"] == data["cancelled"] == 1
    assert data["recent_samples"] == 2
    assert data["recent_p50_ms"] == data["recent_p95_ms"] == 1000
    assert data["queue_p95_ms"] == 1000
    assert data["last_success_age_seconds"] == 2
    assert data["last_finished_age_seconds"] == 0
    assert "SECRET_MESSAGE" not in json.dumps(data)


@pytest.mark.parametrize("option", ["capacity", "sample_limit"])
def test_metrics_reject_nonpositive_limits(option: str) -> None:
    with pytest.raises(ValueError):
        if option == "capacity":
            OperationMetrics(capacity=0)
        else:
            OperationMetrics(sample_limit=0)


@pytest.mark.parametrize(
    "method", ["chat", "classify", "classify_interaction", "extract_user_facts"]
)
def test_every_llm_request_is_counted_without_additional_requests(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    service, _, create, _ = _create_service(monkeypatch)
    create.return_value = _create_response("Ответ")
    result = asyncio.run(getattr(service, method)("SECRET_PROMPT", "SECRET_USER"))
    assert result
    for _ in range(3):
        snapshot = service.snapshot()
        assert snapshot["completed"] == 1
        assert snapshot["active"] == snapshot["failed"] == 0
        assert snapshot["last_result"] == "ok"
        assert "SECRET" not in json.dumps(snapshot)
    create.assert_awaited_once()


def test_llm_failure_and_cancellation_release_active_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        service, _, create, _ = _create_service(monkeypatch)
        create.side_effect = RuntimeError("SECRET_PROVIDER_ERROR")
        with pytest.raises(RuntimeError):
            await service.chat("system", "message")
        entered, release = asyncio.Event(), asyncio.Event()

        async def request(**kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return _create_response("Ответ")

        create.side_effect = request
        task = asyncio.create_task(service.chat("system", "message"))
        await entered.wait()
        assert service.snapshot()["active"] == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        data = service.snapshot()
        assert data["active"] == 0
        assert data["failed"] == data["cancelled"] == 1
        assert "SECRET_PROVIDER_ERROR" not in json.dumps(data)

    asyncio.run(scenario())


def test_cancelled_thread_waiter_keeps_actual_worker_busy() -> None:
    async def scenario() -> None:
        work = BlockingWorkPool()
        started, release = threading.Event(), threading.Event()

        def blocking() -> str:
            started.set()
            assert release.wait(5)
            return "finished"

        first = asyncio.create_task(work.run(blocking))
        await asyncio.to_thread(started.wait, 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert work.snapshot()["active"] == 1
        later = asyncio.create_task(work.run(lambda: "queued"))
        await asyncio.sleep(0)
        assert work.snapshot()["waiting"] == 1
        later.cancel()
        with pytest.raises(asyncio.CancelledError):
            await later
        assert work.snapshot()["active"] == 1
        assert work.snapshot()["waiting"] == 0
        release.set()
        assert await work.run(lambda: "next") == "next"
        data = work.snapshot()
        assert data["completed"] == 3 and data["cancelled"] == 1
        assert data["active"] == data["waiting"] == 0
        assert data["capacity"] == 1

    asyncio.run(scenario())


def test_native_queue_cancellation_and_failure_restore_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pool = NativeWorkPool(1)
        entered, release = asyncio.Event(), asyncio.Event()

        async def execute(*args: Any) -> Any:
            entered.set()
            await release.wait()
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(pool, "_execute", execute)
        first = asyncio.create_task(
            pool.run(extract_document, b"text", "a.txt", "text/plain")
        )
        await entered.wait()
        later = asyncio.create_task(
            pool.run(extract_document, b"other", "b.txt", "text/plain")
        )
        await asyncio.sleep(0)
        assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 1
        later.cancel()
        with pytest.raises(asyncio.CancelledError):
            await later
        release.set()
        with pytest.raises(RuntimeError):
            await first
        data = pool.snapshot()
        assert data["active"] == data["waiting"] == 0
        assert data["failed"] == data["cancelled"] == 1

    asyncio.run(scenario())


def test_native_timeout_counts_failure_without_marking_provider_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pool = NativeWorkPool(1, timeout=0.02)

        async def execute(*args: Any) -> Any:
            await asyncio.Event().wait()

        monkeypatch.setattr(pool, "_execute", execute)
        with pytest.raises(NativeWorkError):
            await pool.run(extract_document, b"text", "a.txt", "text/plain")
        assert pool.snapshot()["failed"] == 1
        assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 0

    asyncio.run(scenario())


def test_audio_queue_and_cancellation_are_visible_without_network_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        service = AudioUnderstandingService(
            cast(Any, Mock()), "synthetic", native_work=cast(WorkRunner, AsyncMock())
        )
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def analyze(data: bytes) -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
            await release.wait()
            return "sound"

        monkeypatch.setattr(service, "_analyze", analyze)
        tasks = [asyncio.create_task(service.analyze(wav_data())) for _ in range(3)]
        await entered.wait()
        await asyncio.sleep(0)
        assert service.snapshot()["active"] == 2
        assert service.snapshot()["waiting"] == 1
        tasks[2].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[2]
        release.set()
        assert await asyncio.gather(*tasks[:2]) == ["sound", "sound"]
        assert service.snapshot()["completed"] == 3
        assert service.snapshot()["cancelled"] == 1
        assert service.snapshot()["active"] == service.snapshot()["waiting"] == 0

    asyncio.run(scenario())


def test_whisper_exposes_shared_worker_metrics() -> None:
    transcriber = SpeechTranscriber()
    assert transcriber.snapshot()["capacity"] == 1
    assert transcriber.snapshot()["active"] == 0


def test_health_is_polling_readiness_even_when_llm_last_request_failed() -> None:
    async def scenario() -> None:
        metrics = OperationMetrics()
        with pytest.raises(RuntimeError):
            with metrics.measure():
                raise RuntimeError("SECRET_PROMPT")
        health = RuntimeHealth()
        health.add_diagnostics("llm", metrics.snapshot)
        server = MiniAppServer("synthetic", UserStateStore(), runtime_health=health)
        assert (await server._health(cast(Any, Mock()))).status == 503
        health.polling_succeeded()
        response = await server._health(cast(Any, Mock()))
        assert response.status == 200
        assert response.text is not None
        assert json.loads(response.text)["diagnostics"]["llm"]["failed"] == 1
        assert "SECRET_PROMPT" not in response.text
        with pytest.raises(ValueError):
            health.add_diagnostics("SECRET_USER", metrics.snapshot)

    asyncio.run(scenario())


@pytest.mark.parametrize("user_id", [123, 999])
def test_admin_status_renders_diagnostics_only_for_authorized_user(
    user_id: int,
) -> None:
    async def scenario() -> None:
        health = RuntimeHealth()
        metrics = OperationMetrics(capacity=2)
        with metrics.measure():
            pass
        health.add_diagnostics("native", metrics.snapshot)
        health.add_diagnostics("llm", OperationMetrics().snapshot)
        users = Mock(spec=UsersRepository)
        users.count.return_value = 10
        router = create_admin_router(
            cast(ImagesRepository, Mock()),
            cast(UsersRepository, users),
            BotState(),
            frozenset({123}),
            runtime_health=health,
        )
        message, _, answer, _ = _create_message_mock(text="/status", user_id=user_id)
        await _call_handler(router, "status", cast(Message, message))
        assert answer.await_args is not None
        text = answer.await_args.args[0]
        if user_id == 123:
            assert "Декодеры: выполняется 0/2, очередь 0" in text
            assert "ещё нет запросов" in text
            assert "p95" in text
            assert "нет свежего подтверждения" in text
        else:
            assert "нет доступа" in text and "Декодеры" not in text

    asyncio.run(scenario())
