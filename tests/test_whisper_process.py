"""Настоящие процессы с подменённой моделью: reuse, reap, очередь и JSON."""

import asyncio
import io
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import protogen_delta.services.whisper_process as process_module
import protogen_delta.services.whisper_worker as worker_module
from protogen_delta.services.speech import SpeechRecognitionError, SpeechTranscriber
from protogen_delta.services.whisper_process import WhisperProcess

FAKE_WORKER = """
import os
from types import SimpleNamespace
from protogen_delta.services import whisper_worker as worker
class Model:
    def __init__(self): self.calls=0
    def transcribe(self,source,**kwargs):
        self.calls+=1
        print('NOISE_FROM_MODEL')
        return iter([SimpleNamespace(text=f'pid={os.getpid()} call={self.calls}')]),SimpleNamespace(language='en')
def load(config):
    assert not any(key in os.environ for key in ('TELEGRAM_TOKEN','DEEPSEEK_API_KEY','HTTPS_PROXY','HF_TOKEN'))
    return Model()
worker.load_model=load
worker.main()
"""


def pcm() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)
    return output.getvalue()


def fake_spawn(
    monkeypatch: pytest.MonkeyPatch, code: str = FAKE_WORKER
) -> list[asyncio.subprocess.Process]:
    real = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        if len(args) > 2 and args[2] == "protogen_delta.services.whisper_worker":
            result = await real(sys.executable, "-c", code, *args[3:], **kwargs)
            processes.append(result)
            return result
        return await real(*args, **kwargs)

    monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", spawn)
    return processes


def test_real_worker_reuses_model_and_removes_audio_after_each_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("TELEGRAM_TOKEN", "DEEPSEEK_API_KEY", "HTTPS_PROXY", "HF_TOKEN"):
        monkeypatch.setenv(key, "synthetic-secret")
    processes = fake_spawn(monkeypatch)

    async def scenario() -> None:
        client = WhisperProcess(timeout=15)
        first = await client.transcribe(pcm())
        second = await client.transcribe(pcm())
        assert first.text.endswith(" call=1")
        assert second.text == first.text.replace(" call=1", " call=2")
        assert len(processes) == 1
        assert client._directory is not None
        directory = Path(client._directory.name)
        assert not (directory / "audio.wav").exists()
        assert client.snapshot()["completed"] == 2
        await client.close()
        await client.close()
        assert processes[0].returncode is not None
        assert not directory.exists()
        with pytest.raises(SpeechRecognitionError, match="остановлено"):
            await client.transcribe(pcm())

    asyncio.run(scenario())


def test_no_speech_keeps_persistent_process_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = FAKE_WORKER.replace(
        "        self.calls+=1",
        '        self.calls+=1\n        if self.calls == 1: return iter([]),SimpleNamespace(language="en")',
    )
    processes = fake_spawn(monkeypatch, code)

    async def scenario() -> None:
        client = WhisperProcess(timeout=15)
        with pytest.raises(SpeechRecognitionError, match="речь"):
            await client.transcribe(pcm())
        assert processes[0].returncode is None
        result = await client.transcribe(pcm())
        assert result.text.endswith(" call=2") and len(processes) == 1
        await client.close()

    asyncio.run(scenario())


def test_dead_cached_worker_is_replaced_on_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = fake_spawn(monkeypatch)

    async def scenario() -> None:
        client = WhisperProcess(timeout=15)
        first = await client.transcribe(pcm())
        await process_module._terminate(processes[0])
        second = await client.transcribe(pcm())
        assert len(processes) == 2 and first.text != second.text
        assert first.text.endswith(" call=1") and second.text.endswith(" call=1")
        await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_hung_inference_is_reaped_before_next_request(
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    async def scenario() -> None:
        real = asyncio.create_subprocess_exec
        entered = asyncio.Event()
        processes: list[asyncio.subprocess.Process] = []

        async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            if len(args) < 3 or args[2] != "protogen_delta.services.whisper_worker":
                return await real(*args, **kwargs)
            code = "import time; time.sleep(30)" if not processes else FAKE_WORKER
            result = await real(sys.executable, "-c", code, *args[3:], **kwargs)
            processes.append(result)
            entered.set()
            return result

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", spawn)
        client = WhisperProcess(timeout=0.3)
        task = asyncio.create_task(client.transcribe(pcm()))
        await entered.wait()
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(SpeechRecognitionError, match="слишком много времени"):
                await task
        assert processes[0].returncode is not None
        assert client.snapshot()["active"] == 0
        client._timeout = 15
        result = await client.transcribe(pcm())
        assert result.text.endswith(" call=1") and len(processes) == 2
        await client.close()
        assert all(process.returncode is not None for process in processes)

    asyncio.run(scenario())


def test_cancellation_during_spawn_reaps_late_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        real = asyncio.create_subprocess_exec
        entered, release = asyncio.Event(), asyncio.Event()
        processes: list[asyncio.subprocess.Process] = []

        async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            if len(args) < 3 or args[2] != "protogen_delta.services.whisper_worker":
                return await real(*args, **kwargs)
            result = await real(
                sys.executable, "-c", "import time; time.sleep(30)", **kwargs
            )
            processes.append(result)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", spawn)
        client = WhisperProcess()
        task = asyncio.create_task(client.transcribe(pcm()))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None
        assert client._directory is None
        await client.close()

    asyncio.run(scenario())


def test_cancellation_is_preserved_if_late_spawn_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def spawn(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            raise OSError("synthetic spawn failure")

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", spawn)
        client = WhisperProcess()
        task = asyncio.create_task(client.transcribe(pcm()))
        await entered.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client._directory is None
        assert client.snapshot()["cancelled"] == 1
        await client.close()

    asyncio.run(scenario())


def test_waiting_timeout_does_not_stop_another_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        client = WhisperProcess(timeout=10)

        async def exchange(data: bytes) -> Any:
            entered.set()
            await release.wait()
            from protogen_delta.services.speech import Transcript

            return Transcript("ok")

        shutdown = Mock(wraps=client._shutdown)
        monkeypatch.setattr(client, "_exchange", exchange)
        monkeypatch.setattr(client, "_shutdown", shutdown)
        first = asyncio.create_task(client.transcribe(pcm()))
        await entered.wait()
        client._timeout = 0.02
        with pytest.raises(SpeechRecognitionError):
            await client.transcribe(pcm())
        assert not first.done() and client.snapshot()["active"] == 1
        assert client.snapshot()["waiting"] == 0
        shutdown.assert_not_called()
        release.set()
        await first
        await client.close()

    asyncio.run(scenario())


def test_close_stops_active_and_queued_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        real = asyncio.create_subprocess_exec
        entered = asyncio.Event()
        processes: list[asyncio.subprocess.Process] = []

        async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            if len(args) < 3 or args[2] != "protogen_delta.services.whisper_worker":
                return await real(*args, **kwargs)
            result = await real(
                sys.executable, "-c", "import time; time.sleep(30)", **kwargs
            )
            processes.append(result)
            entered.set()
            return result

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", spawn)
        client = WhisperProcess()
        active = asyncio.create_task(client.transcribe(pcm()))
        await entered.wait()
        queued = asyncio.create_task(client.transcribe(pcm()))
        await asyncio.sleep(0)
        assert client.snapshot()["waiting"] == 1
        await asyncio.wait_for(client.close(), 5)
        with pytest.raises(asyncio.CancelledError):
            await active
        with pytest.raises(SpeechRecognitionError, match="остановлено"):
            await queued
        assert processes[0].returncode is not None
        assert client.snapshot()["active"] == client.snapshot()["waiting"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"sequence": True},
        {"sequence": 2, "error": "load"},
        {"sequence": 1, "error": []},
        {"sequence": 1, "error": "secret"},
        {
            "sequence": 1,
            "text": "",
            "language": None,
            "truncated": False,
            "uncertain": False,
        },
        {
            "sequence": 1,
            "text": "x",
            "language": {},
            "truncated": False,
            "uncertain": False,
        },
        {
            "sequence": 1,
            "text": "x",
            "language": None,
            "truncated": 0,
            "uncertain": False,
        },
    ],
)
def test_protocol_rejects_untrusted_or_stale_metadata(payload: Any) -> None:
    with pytest.raises(SpeechRecognitionError):
        WhisperProcess._decode(json.dumps(payload).encode(), 1)


@pytest.mark.parametrize(
    "options", [{"timeout": 0}, {"timeout": float("inf")}, {"model_size": ""}]
)
def test_invalid_worker_configuration_is_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        WhisperProcess(**options)


def test_worker_errors_are_bounded_and_no_speech_does_not_drop_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    model.transcribe.side_effect = [
        (iter([]), SimpleNamespace(language="en")),
        (iter([SimpleNamespace(text="ok")]), SimpleNamespace(language="en")),
    ]
    load = Mock(return_value=model)
    monkeypatch.setattr(worker_module, "load_model", load)
    worker = worker_module.WhisperWorker(
        {"model_size": "tiny", "device": "cpu", "compute_type": "int8"}
    )
    assert worker.reply(pcm(), 1) == {"sequence": 1, "error": "no_speech"}
    assert worker.reply(pcm(), 2)["text"] == "ok"
    assert worker.reply(b"broken", 3) == {"sequence": 3, "error": "recognition"}
    load.assert_called_once()
    monkeypatch.setattr(
        worker_module, "load_model", Mock(side_effect=RuntimeError("SECRET_KEY"))
    )
    broken = worker_module.WhisperWorker(
        {"model_size": "tiny", "device": "cpu", "compute_type": "int8"}
    )
    assert broken.reply(pcm(), 1) == {"sequence": 1, "error": "load"}


def test_process_creation_error_cleans_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        async def spawn(*args: Any, **kwargs: Any) -> Any:
            raise OSError("synthetic")

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", spawn)
        client = WhisperProcess()
        with pytest.raises(SpeechRecognitionError):
            await client.transcribe(pcm())
        assert client._process is client._directory is None
        await client.close()

    asyncio.run(scenario())


def test_transcriber_close_delegates_to_inference() -> None:
    from typing import cast
    from unittest.mock import AsyncMock

    from protogen_delta.services.speech import SpeechInference

    inference = AsyncMock(spec=SpeechInference)
    transcriber = SpeechTranscriber(inference=cast(SpeechInference, inference))
    asyncio.run(transcriber.close())
    inference.close.assert_awaited_once()
