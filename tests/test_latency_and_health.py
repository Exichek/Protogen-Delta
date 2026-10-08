"""Число LLM-запросов, параллельность аудио и здоровье Telegram polling."""

import asyncio
import io
import wave
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.client.session.middlewares.base import NextRequestMiddlewareType
from aiogram.methods import GetUpdates, SendMessage
from aiohttp.test_utils import TestClient, TestServer
from test_audio_understanding import wav_data
from test_deepseek_service import _create_response, _create_service
from test_response_engine import _create_engine

from protogen_delta.core.runtime_health import PollingHealthMiddleware, RuntimeHealth
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.services.audio_analysis import AudioAnalysis, analyze_audio
from protogen_delta.services.audio_pipeline import AudioPipeline
from protogen_delta.services.audio_understanding import (
    AudioUnderstandingService,
    audio_excerpt,
)
from protogen_delta.services.blocking_work import WorkRunner
from protogen_delta.services.deepseek import DeepSeekError, DeepSeekService
from protogen_delta.services.interaction_classification import InteractionClassifier
from protogen_delta.services.menu_sync import synchronize_menus
from protogen_delta.services.music import MusicRecognitionService, audio_tags
from protogen_delta.services.native_work import NativeWorkPool
from protogen_delta.services.speech import (
    SpeechRecognitionError,
    SpeechTranscriber,
    Transcript,
)
from protogen_delta.services.speech_audio import prepare_speech_audio


def test_one_classification_replaces_two_legacy_calls() -> None:
    engine, _, model, insult, mood, _ = _create_engine()
    model.classify_interaction.return_value = '{"mood":"sweet","insult":"none"}'
    engine._interaction_classifier = InteractionClassifier(
        cast(DeepSeekService, model), "mood rules", "insult rules"
    )
    asyncio.run(engine.respond(1, "Спасибо за помощь"))
    assert model.classify_interaction.await_count == model.chat.await_count == 1
    insult.classify.assert_not_awaited()
    mood.classify.assert_not_awaited()
    assert engine._user_states.get(1).mood == "sweet"


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        "{}",
        '{"mood":"angry","insult":"direct","instruction":"x"}',
        '{"mood":"new","insult":"none"}',
        '{"mood":"neutral","insult":1}',
        "x" * 4097,
    ],
)
def test_bad_classification_preserves_mood_without_extra_calls(response: str) -> None:
    engine, _, model, insult, mood, _ = _create_engine()
    model.classify_interaction.return_value = response
    engine._interaction_classifier = InteractionClassifier(
        cast(DeepSeekService, model), "mood rules", "insult rules"
    )
    engine._user_states.get(1).mood = "playful"
    asyncio.run(engine.respond(1, "Привет"))
    assert engine._user_states.get(1).mood == "playful"
    assert model.classify_interaction.await_count == 1
    insult.classify.assert_not_awaited()
    mood.classify.assert_not_awaited()


def test_classification_failure_still_allows_chat() -> None:
    engine, _, model, _, _, _ = _create_engine()
    model.classify_interaction.side_effect = DeepSeekError("offline")
    engine._interaction_classifier = InteractionClassifier(
        cast(DeepSeekService, model), "mood", "insult"
    )
    assert asyncio.run(engine.respond(1, "Привет")) == "Ответ"


def test_combined_classifier_sdk_request_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, create, _ = _create_service(monkeypatch)
    create.return_value = _create_response('{"mood":"neutral","insult":"none"}')
    asyncio.run(service.classify_interaction("rules", "message"))
    assert create.await_args and create.await_count == 1
    kwargs = create.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] == 80 and kwargs["temperature"] == 0
    assert len(kwargs["messages"]) == 2 and "tools" not in kwargs


def test_audio_stages_enter_together() -> None:
    async def scenario() -> None:
        entered: set[str] = set()
        ready = asyncio.Event()

        async def barrier(name: str, value: Any) -> Any:
            entered.add(name)
            if len(entered) == 5:
                ready.set()
            await ready.wait()
            return value

        class Work:
            async def run(self, function: Any, *args: Any, **kwargs: Any) -> Any:
                return await barrier(
                    function.__name__,
                    (
                        {"title": "track"}
                        if function is audio_tags
                        else AudioAnalysis(1, 16000, 1, -10, -1, 0, 8, 1500)
                    ),
                )

        speech = AsyncMock(spec=SpeechTranscriber)

        async def transcribe(data: bytes) -> Transcript:
            return cast(Transcript, await barrier("speech", Transcript("words")))

        speech.transcribe.side_effect = transcribe
        understanding = AsyncMock(spec=AudioUnderstandingService)

        async def describe(data: bytes) -> str:
            return cast(str, await barrier("understanding", "music"))

        understanding.analyze.side_effect = describe
        recognition = AsyncMock(spec=MusicRecognitionService)

        async def recognize(data: bytes) -> str:
            return cast(str, await barrier("recognition", "identified"))

        recognition.recognize.side_effect = recognize
        pipeline = AudioPipeline(
            cast(SpeechTranscriber, speech),
            cast(WorkRunner, Work()),
            cast(AudioUnderstandingService, understanding),
            cast(MusicRecognitionService, recognition),
            timeout=2,
        )
        result = await pipeline.collect(b"audio", uploaded=True)
        assert len(entered) == 5 and result.metadata == {"title": "track"}
        assert result.transcript and result.transcript.text == "words"
        assert (
            result.analysis
            and result.semantic_report == "music"
            and result.music_report == "identified"
        )

    asyncio.run(scenario())


def test_audio_budget_keeps_completed_results_and_cancels_pending() -> None:
    async def scenario() -> None:
        cancelled = asyncio.Event()
        speech = AsyncMock(spec=SpeechTranscriber)

        async def blocked(data: bytes) -> Transcript:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return Transcript("never")

        speech.transcribe.side_effect = blocked

        class Work:
            async def run(self, function: Any, *args: Any, **kwargs: Any) -> Any:
                if function is audio_tags:
                    return {"title": "known"}
                raise ValueError("bad measurements")

        model = AsyncMock(spec=AudioUnderstandingService)
        model.analyze.return_value = "guitar"
        pipeline = AudioPipeline(
            cast(SpeechTranscriber, speech),
            cast(WorkRunner, Work()),
            cast(AudioUnderstandingService, model),
            timeout=0.03,
        )
        result = await pipeline.collect(b"audio", uploaded=True)
        assert (
            result.metadata == {"title": "known"} and result.semantic_report == "guitar"
        )
        assert (
            result.transcript is None and result.analysis is None and cancelled.is_set()
        )

    asyncio.run(scenario())


def test_voice_skips_music_and_speech_timeout_is_bounded() -> None:
    async def scenario() -> None:
        speech = AsyncMock(spec=SpeechTranscriber)
        speech.transcribe.side_effect = SpeechRecognitionError("no speech")
        work = AsyncMock()
        model = AsyncMock(spec=AudioUnderstandingService)
        recognition = AsyncMock(spec=MusicRecognitionService)
        pipeline = AudioPipeline(
            cast(SpeechTranscriber, speech),
            cast(WorkRunner, work),
            cast(AudioUnderstandingService, model),
            cast(MusicRecognitionService, recognition),
            speech_timeout=0.02,
        )
        assert (await pipeline.collect(b"audio", uploaded=False)).transcript is None
        work.run.assert_not_awaited()
        model.analyze.assert_not_awaited()
        recognition.recognize.assert_not_awaited()

        async def blocked(data: bytes) -> Transcript:
            await asyncio.Event().wait()
            return Transcript("never")

        speech.transcribe.side_effect = blocked
        assert (await pipeline.collect(b"audio", uploaded=False)).transcript is None

    asyncio.run(scenario())


def test_audio_workers_return_metrics_tags_and_valid_pcm() -> None:
    async def scenario() -> None:
        pool = NativeWorkPool()
        data = wav_data()
        assert await pool.run(audio_tags, data) == {}
        metrics = await pool.run(analyze_audio, data)
        assert metrics.duration_seconds > 0 and metrics.source_sample_rate > 0
        for function in (audio_excerpt, prepare_speech_audio):
            pcm = await pool.run(function, data)
            with wave.open(io.BytesIO(pcm), "rb") as wav:
                assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (
                    1,
                    16000,
                    2,
                )

    asyncio.run(scenario())


def test_polling_health_uses_getupdates_not_other_requests() -> None:
    async def scenario() -> None:
        clock = [100.0]
        health = RuntimeHealth(clock=lambda: clock[0])
        middleware = PollingHealthMiddleware(health)
        request = AsyncMock(return_value=[])
        bot = cast(Bot, AsyncMock(spec=Bot))
        assert health.snapshot()["ok"] is False
        await middleware(
            cast(NextRequestMiddlewareType[Any], request),
            bot,
            SendMessage(chat_id=1, text="private"),
        )
        assert health.snapshot()["ok"] is False
        await middleware(
            cast(NextRequestMiddlewareType[Any], request), bot, GetUpdates()
        )
        assert health.snapshot()["ok"] is True
        request.side_effect = RuntimeError("offline")
        with pytest.raises(RuntimeError):
            await middleware(
                cast(NextRequestMiddlewareType[Any], request), bot, GetUpdates()
            )
        assert health.snapshot()["polling_errors"] == 1
        clock[0] = 191
        assert health.snapshot()["ok"] is False
        health.polling_succeeded()
        assert health.snapshot()["ok"] is True
        health.stop()
        assert health.snapshot()["ok"] is False

    asyncio.run(scenario())


def test_http_health_reports_starting_stale_and_ready() -> None:
    async def scenario() -> None:
        clock = [0.0]
        health = RuntimeHealth(clock=lambda: clock[0])
        server = MiniAppServer("token", UserStateStore(), runtime_health=health)
        async with TestClient(TestServer(server.application())) as client:
            assert (await client.get("/health")).status == 503
            health.polling_succeeded()
            response = await client.get("/health")
            assert response.status == 200 and (await response.json())["polling"] is True
            clock[0] = 90
            assert (await client.get("/health")).status == 503

    asyncio.run(scenario())


def test_menu_sync_is_bounded_and_one_failure_does_not_stop_others() -> None:
    async def scenario() -> None:
        active = maximum = 0
        visited: list[int] = []

        async def update(user: int) -> None:
            nonlocal active, maximum
            visited.append(user)
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.001)
                if user == 2:
                    raise RuntimeError("one failed")
            finally:
                active -= 1

        await synchronize_menus(range(20), update, limit=3)
        assert sorted(visited) == list(range(20)) and maximum == 3 and active == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_runtime_budgets_validate_input(value: float) -> None:
    with pytest.raises(ValueError):
        RuntimeHealth(stale_seconds=value)
    with pytest.raises(ValueError):
        AudioPipeline(
            cast(SpeechTranscriber, AsyncMock()),
            cast(WorkRunner, AsyncMock()),
            timeout=value,
        )


def test_empty_classifier_rules_and_menu_limit_are_rejected() -> None:
    with pytest.raises(ValueError):
        InteractionClassifier(cast(DeepSeekService, AsyncMock()), "", "rules")
    with pytest.raises(ValueError):
        asyncio.run(synchronize_menus([], AsyncMock(), limit=0))


def test_speech_receives_prepared_pcm_before_model_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        data = prepare_speech_audio(wav_data())
        work = AsyncMock()
        work.run.return_value = data
        transcriber = SpeechTranscriber(native_work=cast(WorkRunner, work))

        class Model:
            def transcribe(self, source: io.BytesIO, **kwargs: Any) -> Any:
                with wave.open(source, "rb") as wav:
                    assert wav.getframerate() == 16000 and wav.getnchannels() == 1
                from types import SimpleNamespace

                return iter([SimpleNamespace(text="Recognized")]), SimpleNamespace(
                    language="en"
                )

        monkeypatch.setattr(transcriber, "_load_model", lambda: Model())
        result = await transcriber.transcribe(b"compressed input")
        assert result.text == "Recognized"
        work.run.assert_awaited_once_with(prepare_speech_audio, b"compressed input")
        work.run.side_effect = ValueError("bad file")
        with pytest.raises(SpeechRecognitionError):
            await transcriber.transcribe(b"broken")

    asyncio.run(scenario())


def test_pcm_preparation_rejects_empty_streams() -> None:
    from protogen_delta.services.native_work import NativeWorkError

    with pytest.raises(Exception):
        prepare_speech_audio(b"not audio")
    with pytest.raises(NativeWorkError):
        NativeWorkPool._decode("speech", {"speech_data": b"RIFFinvalid"})


def test_cancelling_audio_pipeline_cancels_its_stages() -> None:
    async def scenario() -> None:
        started, stopped = asyncio.Event(), asyncio.Event()
        speech = AsyncMock(spec=SpeechTranscriber)

        async def transcribe(data: bytes) -> Transcript:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return Transcript("never")

        speech.transcribe.side_effect = transcribe
        pipeline = AudioPipeline(
            cast(SpeechTranscriber, speech), cast(WorkRunner, AsyncMock())
        )
        task = asyncio.create_task(pipeline.collect(b"audio", uploaded=False))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(scenario())


def test_cancelling_menu_sync_stops_inflight_updates() -> None:
    async def scenario() -> None:
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def update(user: int) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        task = asyncio.create_task(synchronize_menus(range(10000), update, limit=1))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation,value",
    [
        ("audio", {"analysis": {"duration_seconds": "bad"}}),
        ("tags", {"tags": {"instruction": "bad"}}),
        ("clip", {"clip": "not base64"}),
    ],
)
def test_audio_worker_protocol_validates_results(
    operation: str, value: dict[str, Any]
) -> None:
    from protogen_delta.services.native_work import NativeWorkError

    with pytest.raises(NativeWorkError):
        NativeWorkPool._decode(operation, value)
