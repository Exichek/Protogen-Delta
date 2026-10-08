"""Реальные дочерние процессы: таймаут, отмена, восстановление слота и JSON."""

import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from test_tgs_frames import _tgs

import protogen_delta.services.native_work as native
import protogen_delta.services.native_worker as worker
from protogen_delta.services.animation_frames import extract_animation_frames
from protogen_delta.services.appearance_image import prepare_appearance_image
from protogen_delta.services.documents import (
    DocumentReadError,
    DocumentTooLargeError,
    UnsupportedDocumentError,
    extract_document,
)
from protogen_delta.services.native_work import (
    NativeWorkError,
    NativeWorkPool,
    worker_environment,
)
from protogen_delta.services.tgs_frames import extract_tgs_frames


def _image(format: str = "PNG") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(output, format)
    return output.getvalue()


def test_real_worker_reads_document_and_images() -> None:
    async def scenario() -> None:
        pool = NativeWorkPool()
        doc = await pool.run(
            extract_document, b"Hello world", "readme.txt", "text/plain"
        )
        assert doc.text == "Hello world" and doc.kind == "TXT"
        image = await pool.run(prepare_appearance_image, _image())
        assert image.mime_type == "image/jpeg" and image.data.startswith(b"\xff\xd8")
        frames = await pool.run(extract_animation_frames, _image("GIF"), label="GIF")
        assert frames and frames[0].data.startswith(b"\x89PNG")
        tgs = await pool.run(extract_tgs_frames, _tgs())
        assert len(tgs) == 4 and all(frame.mime_type == "image/png" for frame in tgs)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_hung_worker_is_reaped_and_next_request_runs(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    async def scenario() -> None:
        start = asyncio.create_subprocess_exec
        spawned: list[asyncio.subprocess.Process] = []
        entered = asyncio.Event()

        async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            if (
                len(args) > 2
                and args[2] == "protogen_delta.services.native_worker"
                and not spawned
            ):
                process = await start(
                    sys.executable, "-c", "import time; time.sleep(30)", **kwargs
                )
                spawned.append(process)
                entered.set()
                return process
            return await start(*args, **kwargs)

        monkeypatch.setattr(native.asyncio, "create_subprocess_exec", spawn)
        pool = NativeWorkPool(limit=1, timeout=0.3)
        task = asyncio.create_task(pool.run(extract_document, b"text", "x.txt", None))
        await entered.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(NativeWorkError, match="слишком много времени"):
                await task
        assert spawned[0].returncode is not None
        pool._timeout = 10
        result = await pool.run(extract_document, b"next", "x.txt", None)
        assert result.text == "next"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload", ["broken", "[]", '{"error":"unsupported"}', "x" * 300]
)
def test_bad_worker_result_is_bounded_and_slot_is_released(
    monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    async def scenario() -> None:
        start = asyncio.create_subprocess_exec

        async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            if len(args) > 2 and args[2] == "protogen_delta.services.native_worker":
                code = "import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])"
                return await start(
                    sys.executable, "-c", code, args[4], payload, **kwargs
                )
            return await start(*args, **kwargs)

        monkeypatch.setattr(native.asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(native, "MAX_NATIVE_OUTPUT", 200)
        pool = NativeWorkPool(limit=1)
        for _ in range(2):
            with pytest.raises((NativeWorkError, UnsupportedDocumentError)):
                await pool.run(extract_document, b"test", "x.txt", None)

    asyncio.run(scenario())


def test_worker_does_not_inherit_secrets_or_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "TELEGRAM_TOKEN",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "HTTPS_PROXY",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(key, "synthetic-secret")
    environment = worker_environment()
    assert "synthetic-secret" not in environment.values()
    assert environment["OMP_NUM_THREADS"] == "1"


@pytest.mark.parametrize(
    "limit,timeout", [(0, 1), (1, 0), (1, float("inf")), (1, float("nan"))]
)
def test_worker_limits_are_validated(limit: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        NativeWorkPool(limit, timeout)


def test_unknown_callable_and_large_input_cannot_launch_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        launch = AsyncMock()
        monkeypatch.setattr(native.asyncio, "create_subprocess_exec", launch)
        with pytest.raises(ValueError):
            await NativeWorkPool().run(lambda data: data, b"test")
        with pytest.raises(DocumentTooLargeError):
            await NativeWorkPool().run(
                extract_document, b"x" * (native.MAX_NATIVE_INPUT + 1), "x.txt", None
            )
        with pytest.raises(NativeWorkError):
            await NativeWorkPool().run(prepare_appearance_image, b"")
        launch.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation,error,expected",
    [
        ("document", "too_large", DocumentTooLargeError),
        ("document", "unsupported", UnsupportedDocumentError),
        ("document", "read_error", DocumentReadError),
        ("tgs", "failed", NativeWorkError),
    ],
)
def test_domain_errors_survive_worker_protocol(
    operation: str, error: str, expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        NativeWorkPool._decode(operation, {"error": error})


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"images": [{"data": "!", "mime_type": "image/png", "label": "x"}]},
        {"images": [{"data": "eA==", "mime_type": "text/html", "label": "x"}]},
    ],
)
def test_invalid_image_protocol_is_rejected(value: dict[str, Any]) -> None:
    with pytest.raises(NativeWorkError):
        NativeWorkPool._decode("appearance", value)


@pytest.mark.parametrize(
    "operation,data,args,kwargs",
    [
        ("document", b"hi", ["x.txt", None], {}),
        ("appearance", _image(), [], {}),
        ("animation", _image("GIF"), [], {"label": "GIF"}),
        ("tgs", _tgs(), [], {}),
    ],
)
def test_worker_serializes_real_results(
    operation: str, data: bytes, args: list[Any], kwargs: dict[str, Any]
) -> None:
    value = worker.perform(
        data, {"operation": operation, "args": args, "kwargs": kwargs}
    )
    if operation == "document":
        assert value["text"] == "hi"
    else:
        assert value["images"]
        assert base64.b64decode(value["images"][0]["data"])
    assert NativeWorkPool._decode(operation, value)


@pytest.mark.parametrize(
    "data,name,error",
    [
        (b"text", "x.zip", "unsupported"),
        (b"broken", "x.pdf", "read_error"),
        (b"", "x.txt", "read_error"),
    ],
)
def test_worker_reports_known_document_failures(
    data: bytes, name: str, error: str
) -> None:
    value = worker.perform(
        data, {"operation": "document", "args": [name, None], "kwargs": {}}
    )
    assert value == {"error": error}


def test_worker_main_writes_only_bounded_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target, request = (
        tmp_path / name for name in ("source", "result", "request")
    )
    source.write_bytes(b"test")
    request.write_text(
        json.dumps({"operation": "document", "args": ["x.txt", None], "kwargs": {}}),
        "utf-8",
    )
    monkeypatch.setattr(worker, "apply_limits", lambda: None)
    monkeypatch.setattr(
        worker.sys, "argv", ["worker", str(source), str(target), str(request)]
    )
    worker.main()
    assert json.loads(target.read_text("utf-8"))["text"] == "test"
    monkeypatch.setattr(worker, "MAX_OUTPUT", 1)
    worker.main()
    assert json.loads(target.read_text("utf-8")) == {"error": "too_large"}
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="input_size"):
        worker.main()


def test_posix_resource_limits_and_process_group_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits: list[tuple[int, tuple[int, int]]] = []
    resource = SimpleNamespace(
        RLIMIT_AS=1,
        RLIMIT_CPU=2,
        RLIMIT_FSIZE=3,
        RLIMIT_CORE=4,
        setrlimit=lambda key, value: limits.append((key, value)),
    )
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(worker, "sys", SimpleNamespace(platform="linux"))
    worker.apply_limits()
    assert len(limits) == 4 and limits[-1][1] == (0, 0)
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="linux"))
    groups: list[int] = []
    monkeypatch.setattr(
        native.os, "killpg", lambda pid, sig: groups.append(pid), raising=False
    )
    monkeypatch.setattr(native.signal, "SIGKILL", 9, raising=False)
    process = SimpleNamespace(pid=123, wait=AsyncMock())
    asyncio.run(native._terminate(cast(asyncio.subprocess.Process, process)))
    assert groups == [123]
    process.wait.assert_awaited_once()


@pytest.mark.parametrize(
    "reference",
    [
        {"p": "/app/data/private.png"},
        {"u": "https://example.org/image.png"},
        {"fPath": "/private/font.ttf"},
    ],
)
def test_tgs_cannot_load_external_paths(reference: dict[str, str]) -> None:
    from protogen_delta.services.tgs_frames import _valid_lottie

    assert not _valid_lottie({"w": 32, "h": 32, "op": 10, "layers": [reference]})


def test_playlist_cannot_open_network_or_local_file() -> None:
    for reference in ("http://127.0.0.1:1/secret.ts", "file:///app/data/secret.ts"):
        playlist = (
            "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n"
            + reference
            + "\n#EXT-X-ENDLIST\n"
        ).encode()
        assert extract_animation_frames(playlist, label="playlist") == ()
