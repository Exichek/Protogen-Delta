"""Настоящие кодеки, подпись Mini App, предпросмотр, сохранение и очистка файлов."""

import asyncio
import json
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote

import av
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_download_video_format import _video
from test_miniapp import TOKEN, _signed_init_data

from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.services.media_conversion import (
    ConversionBusyError,
    ConversionError,
    MediaConversionService,
)
from protogen_delta.services.media_conversion_worker import (
    convert_media,
    run,
    validate_options,
)


@pytest.mark.parametrize(
    "extension,codec",
    [("mp4", "h264"), ("avi", "h264"), ("mov", "h264"), ("mkv", "h264")],
)
@pytest.mark.parametrize("sound", [False, True])
def test_real_video_conversion_preserves_movement_sound_and_duration(
    tmp_path: Path, extension: str, codec: str, sound: bool
) -> None:
    source = tmp_path / "input.mp4"
    _video(source, sound=sound)
    target = tmp_path / ("result." + extension)
    metadata = convert_media(source, target, mode="video", extension=extension)
    assert metadata["bytes"] == target.stat().st_size
    with av.open(str(target)) as result:
        assert (
            result.duration is not None
            and 0.95 <= result.duration / av.time_base <= 1.1
        )
        assert result.streams.video[0].codec_context.name == codec
        assert bool(result.streams.audio) is sound
        frames = list(result.decode(video=0))
        assert len(frames) == 24 and frames[0].format.name == "yuv420p"
        assert np.max(frames[-1].to_ndarray()) > np.max(frames[0].to_ndarray())
    if sound:
        with av.open(str(target)) as result:
            assert result.streams.audio[0].codec_context.name == (
                "mp3float" if extension == "avi" else "aac"
            )
            samples = np.concatenate(
                [frame.to_ndarray().reshape(-1) for frame in result.decode(audio=0)]
            )
            assert samples.size >= 90000 and np.max(np.abs(samples)) > 0.1


@pytest.mark.parametrize(
    "extension,codec", [("mp3", "mp3float"), ("wav", "pcm_s24le"), ("flac", "flac")]
)
def test_real_audio_is_extracted_from_video(
    tmp_path: Path, extension: str, codec: str
) -> None:
    source = tmp_path / "input.mp4"
    _video(source)
    target = tmp_path / ("result." + extension)
    convert_media(source, target, mode="audio", extension=extension, bitrate=320)
    with av.open(str(target)) as result:
        assert not result.streams.video
        assert result.streams.audio[0].codec_context.name == codec
        assert sum(frame.samples for frame in result.decode(audio=0)) >= 47000


@pytest.mark.parametrize(
    "mode,extension",
    [
        ("video", "mp4"),
        ("video", "avi"),
        ("video", "mov"),
        ("video", "mkv"),
        ("audio", "mp3"),
        ("audio", "wav"),
        ("audio", "flac"),
    ],
)
def test_real_worker_result_and_browser_preview(
    tmp_path: Path, mode: str, extension: str
) -> None:
    _video(tmp_path / "input", container_format="mp4")
    run(
        tmp_path,
        {"mode": mode, "format": extension, "resolution": "720", "bitrate": 192},
    )
    assert json.loads((tmp_path / "result.json").read_text("utf-8"))["bytes"] > 0
    target = tmp_path / ("result." + extension)
    preview = tmp_path / ("preview.mp4" if mode == "video" else "preview.mp3")
    path = preview if preview.exists() else target
    with av.open(str(path)) as result:
        if mode == "video":
            assert "mp4" in result.format.name.split(",")
            assert result.streams.video[0].codec_context.name == "h264"
            assert next(result.decode(video=0)).width == 64
        else:
            assert result.streams.audio
            assert next(result.decode(audio=0)).samples > 0


@pytest.mark.parametrize(
    "mode,extension,resolution,bitrate",
    [
        ("bad", "mp4", "720", 192),
        ("audio", "mp4", "720", 192),
        ("video", "mp3", "720", 192),
        ("video", "mp4", "99999", 192),
        ("audio", "mp3", "720", True),
        ("audio", "mp3", "720", 500),
    ],
)
def test_conversion_rejects_unbounded_options(
    mode: str, extension: str, resolution: str, bitrate: int
) -> None:
    with pytest.raises(ValueError):
        validate_options(mode, extension, resolution, bitrate)


def test_conversion_rejects_empty_input_missing_audio_and_indirect_formats(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="непустым"):
        convert_media(source, tmp_path / "out.mp4", mode="video", extension="mp4")
    _video(source, container_format="mp4", sound=False)
    with pytest.raises(ValueError, match="нет звука"):
        convert_media(source, tmp_path / "out.mp3", mode="audio", extension="mp3")
    source.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile:///etc/passwd\n", "utf-8"
    )
    with pytest.raises(av.error.FFmpegError):
        convert_media(source, tmp_path / "out.mp4", mode="video", extension="mp4")


def test_service_keeps_private_results_then_expires_and_cleans_up(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    _video(source)
    now = [100.0]
    service = MediaConversionService(clock=lambda: now[0])

    async def scenario() -> None:
        async def read_file() -> bytes:
            return source.read_bytes()

        job = await service.convert(
            42, read_file, name="../../мой файл.mp4", mode="video", extension="mp4"
        )
        assert job.name == "мой файл.mp4" and job.path.exists()
        assert service.get(job.token) is job
        assert not await service.remove(job.token, 99)
        assert not await service.remove("unknown", 42)
        now[0] += 901
        assert service.get(job.token) is None
        await service.prune()
        assert not job.path.exists()
        await service.close()

    asyncio.run(scenario())


def test_service_limits_concurrency_before_reading_second_file(tmp_path: Path) -> None:
    service = MediaConversionService()
    second = Mock()

    async def scenario() -> None:
        entered = asyncio.Event()

        async def read_file() -> bytes:
            entered.set()
            await asyncio.Event().wait()
            return b"unreachable"

        task = asyncio.create_task(
            service.convert(
                42, read_file, name="file.mp4", mode="video", extension="mp4"
            )
        )
        await entered.wait()
        with pytest.raises(ConversionBusyError, match="Предыдущий"):
            await service.convert(
                42, second, name="file.mp4", mode="video", extension="mp4"
            )
        with pytest.raises(ConversionBusyError, match="занят"):
            await service.convert(
                99, second, name="file.mp4", mode="video", extension="mp4"
            )
        second.assert_not_called()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not service._pending and not service._active
        await service.close()

    asyncio.run(scenario())


def test_service_timeout_and_cooldown(tmp_path: Path) -> None:
    service = MediaConversionService(timeout=0.01)

    async def scenario() -> None:
        async def blocked() -> bytes:
            await asyncio.Event().wait()
            return b"unreachable"

        with pytest.raises(ConversionError, match="четырёх"):
            await service.convert(
                42, blocked, name="x.mp4", mode="video", extension="mp4"
            )
        with pytest.raises(ConversionBusyError, match="10 секунд"):
            await service.convert(
                42, blocked, name="x.mp4", mode="video", extension="mp4"
            )
        await service.close()

    asyncio.run(scenario())


def test_authenticated_api_download_preview_ranges_and_owner_deletion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    _video(source)

    async def scenario() -> None:
        server = MiniAppServer(TOKEN, UserStateStore())
        async with TestClient(TestServer(server.application())) as client:
            options = quote(
                json.dumps(
                    {"mode": "video", "format": "mp4", "file_name": "пример.mp4"}
                )
            )
            headers = {
                "X-Telegram-Init-Data": _signed_init_data(),
                "X-Conversion-Options": options,
                "Content-Type": "video/mp4",
            }
            denied = await client.post(
                "/api/tools/convert/file",
                data=source.read_bytes(),
                headers={"X-Conversion-Options": options},
            )
            assert denied.status == 401
            response = await client.post(
                "/api/tools/convert/file", data=source.read_bytes(), headers=headers
            )
            assert response.status == 200
            result = await response.json()
            assert result["name"] == "пример.mp4" and result["duration"] > 0
            download = await client.get(result["download_url"])
            assert download.status == 200 and download.content_type == "video/mp4"
            assert "attachment" in download.headers["Content-Disposition"]
            assert (await download.read())[4:8] == b"ftyp"
            preview = await client.get(
                result["preview_url"], headers={"Range": "bytes=0-31"}
            )
            assert preview.status == 206 and len(await preview.read()) == 32
            denied = await client.delete(
                "/api/tools/convert/" + result["token"],
                headers={"X-Telegram-Init-Data": _signed_init_data(user_id=99)},
            )
            assert denied.status == 404
            assert (await client.get("/converted-media/unknown/download")).status == 404
            assert (
                await client.get("/converted-media/" + result["token"] + "/unknown")
            ).status == 404
            deleted = await client.delete(
                "/api/tools/convert/" + result["token"],
                headers={"X-Telegram-Init-Data": _signed_init_data()},
            )
            assert deleted.status == 200
            assert (await client.get(result["download_url"])).status == 404

    asyncio.run(scenario())


@pytest.mark.parametrize("extension", ["wav", "flac"])
def test_lossless_audio_retains_samples_and_original_sample_rate(
    tmp_path: Path, extension: str
) -> None:
    source = tmp_path / "input.wav"
    values = np.arange(96000, dtype=np.int16)[None, :]
    with av.open(str(source), "w", format="wav") as output:
        stream = output.add_stream("pcm_s16le", rate=96000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(values, format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 96000, 0, Fraction(1, 96000)
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    target = tmp_path / ("output." + extension)
    convert_media(source, target, mode="audio", extension=extension)
    with av.open(str(target)) as result:
        assert result.streams.audio[0].codec_context.sample_rate == 96000
        actual = np.concatenate(
            [frame.to_ndarray().reshape(-1) for frame in result.decode(audio=0)]
        )
        np.testing.assert_array_equal(actual, values.reshape(-1))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"mode": "video", "format": "exe"},
        {"mode": True, "format": "mp4"},
        {"mode": "video", "format": "mp4", "user_id": 99},
        {"mode": "audio", "format": "mp3", "bitrate": True},
    ],
)
def test_api_rejects_invalid_options_before_reading_file(
    options: dict[str, Any],
) -> None:
    async def scenario() -> None:
        server = MiniAppServer(TOKEN, UserStateStore())
        async with TestClient(TestServer(server.application())) as client:
            response = await client.post(
                "/api/tools/convert/file",
                data=b"invalid",
                headers={
                    "X-Telegram-Init-Data": _signed_init_data(),
                    "X-Conversion-Options": quote(json.dumps(options)),
                },
            )
            assert response.status == 400
            assert not server._media_converter._pending

    asyncio.run(scenario())


def test_corrupt_file_and_empty_upload_fail_without_retained_data() -> None:
    async def scenario() -> None:
        service = MediaConversionService()

        async def empty() -> bytes:
            return b""

        with pytest.raises(ConversionError, match="непустым"):
            await service.convert(42, empty, name="x", mode="audio", extension="mp3")

        async def corrupt() -> bytes:
            return b"PRIVATE_SECRET_NOT_MEDIA"

        with pytest.raises(ConversionError, match="Не удалось") as error:
            await service.convert(99, corrupt, name="x", mode="video", extension="mp4")
        assert (
            "PRIVATE" not in str(error.value)
            and not service._jobs
            and not service._pending
        )
        await service.close()

    asyncio.run(scenario())


def test_service_shutdown_cancels_read_and_cleanup_loop() -> None:
    async def scenario() -> None:
        service = MediaConversionService()
        await service.start()
        await service.start()
        entered = asyncio.Event()

        async def blocked() -> bytes:
            entered.set()
            await asyncio.Event().wait()
            return b"unreachable"

        task = asyncio.create_task(
            service.convert(42, blocked, name="x", mode="audio", extension="mp3")
        )
        await entered.wait()
        await service.close()
        assert (
            task.cancelled() and service._cleanup_task is None and not service._active
        )

    asyncio.run(scenario())


def test_new_conversion_replaces_previous_result_and_close_removes_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    _video(source)

    async def scenario() -> None:
        service = MediaConversionService()
        monkeypatch.setattr(service._limiter, "allow", Mock(return_value=True))

        async def read_file() -> bytes:
            return source.read_bytes()

        first = await service.convert(
            42, read_file, name="x.mp4", mode="video", extension="mp4"
        )
        second = await service.convert(
            42, read_file, name="x.mp4", mode="audio", extension="mp3"
        )
        assert not first.path.exists() and service.get(first.token) is None
        assert second.path.exists()
        await service.close()
        assert not second.path.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_phone_rotation_is_applied_to_output_frames(
    tmp_path: Path, rotation: int
) -> None:
    source = tmp_path / "phone.mp4"
    with av.open(str(source), "w") as output:
        video = output.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        video.set_display_rotation(rotation)
        for index in range(24):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), 50 + index, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode(None):
            output.mux(packet)
    target = tmp_path / "result.mp4"
    convert_media(source, target, mode="video", extension="mp4")
    with av.open(str(target)) as result:
        frame = next(result.decode(video=0))
        assert (frame.width, frame.height) == (
            (48, 64) if rotation != 180 else (64, 48)
        )
        assert frame.rotation == 0


def test_worker_applies_limits_and_blocks_network_and_child_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import protogen_delta.services.download_compression as compression
    import protogen_delta.services.media_conversion_worker as worker

    (tmp_path / "options.json").write_text(
        '{"mode":"video","format":"mp4","resolution":"720","bitrate":192}', "utf-8"
    )
    limits = Mock()
    hook = Mock()
    operation = Mock()
    monkeypatch.setattr(compression, "limit_compression_process", limits)
    monkeypatch.setattr(worker.sys, "addaudithook", hook)
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(tmp_path)])
    monkeypatch.setattr(worker, "run", operation)
    worker.main()
    limits.assert_called_once()
    operation.assert_called_once()
    audit = hook.call_args.args[0]
    for event in ("socket.connect", "subprocess.Popen", "os.system", "os.posix_spawn"):
        with pytest.raises(OSError):
            audit(event, ())
    audit("open", ())
    operation.side_effect = ValueError("PRIVATE_SECRET")
    with pytest.raises(SystemExit) as error:
        worker.main()
    assert error.value.code == 1 and "PRIVATE" not in (
        tmp_path / "error.json"
    ).read_text("utf-8")


@pytest.mark.parametrize("reason", ["no_video", "duration", "dimensions"])
def test_worker_rejects_invalid_metadata_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    from types import SimpleNamespace

    source = tmp_path / "input"
    source.write_bytes(b"synthetic")
    video = SimpleNamespace(width=5000 if reason == "dimensions" else 64, height=48)
    incoming = SimpleNamespace(
        duration=(601 if reason == "duration" else 1) * av.time_base,
        streams=SimpleNamespace(
            video=[] if reason == "no_video" else [video], audio=[]
        ),
    )
    context = Mock()
    context.__enter__ = Mock(return_value=incoming)
    context.__exit__ = Mock(return_value=None)
    monkeypatch.setattr(av, "open", Mock(return_value=context))
    with pytest.raises(ValueError):
        convert_media(source, tmp_path / "out.mp4", mode="video", extension="mp4")


def test_api_accepts_video_larger_than_default_json_body_limit(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    _video(source)
    with source.open("ab") as stream:
        padding = 100000
        stream.write(padding.to_bytes(4, "big") + b"free" + bytes(padding - 8))

    async def scenario() -> None:
        server = MiniAppServer(TOKEN, UserStateStore())
        async with TestClient(TestServer(server.application())) as client:
            response = await client.post(
                "/api/tools/convert/file",
                data=source.read_bytes(),
                headers={
                    "X-Telegram-Init-Data": _signed_init_data(),
                    "X-Conversion-Options": quote(
                        json.dumps({"mode": "video", "format": "mp4"})
                    ),
                },
            )
            assert response.status == 200

    asyncio.run(scenario())


def test_service_cancellation_reaps_worker_before_temporary_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import protogen_delta.services.media_conversion as module

    source = tmp_path / "input.mp4"
    _video(source)

    async def scenario() -> None:
        entered = asyncio.Event()
        reaped = asyncio.Event()
        process = SimpleNamespace(returncode=None)

        async def wait() -> int:
            entered.set()
            await asyncio.Event().wait()
            return 0

        process.wait = wait

        async def terminate(child: Any) -> None:
            assert child is process
            process.returncode = -9
            reaped.set()

        create = AsyncMock(return_value=process)
        monkeypatch.setattr(module.asyncio, "create_subprocess_exec", create)
        monkeypatch.setattr(module, "_terminate", terminate)
        service = MediaConversionService()

        async def read_file() -> bytes:
            return source.read_bytes()

        task = asyncio.create_task(
            service.convert(42, read_file, name="x.mp4", mode="video", extension="mp4")
        )
        await entered.wait()
        directory = Path(create.call_args.args[-1])
        assert directory.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reaped.is_set() and not directory.exists() and not service._pending
        await service.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_service_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError):
        MediaConversionService(timeout=timeout)


def test_cache_admission_does_not_read_input_and_does_not_evict_other_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    async def scenario() -> None:
        service = MediaConversionService()
        for user_id in range(4):
            job = Mock(
                user_id=user_id,
                expires_at=service._clock() + 900,
                directory=SimpleNamespace(cleanup=Mock()),
            )
            service._jobs[str(user_id)] = job
        read_file = AsyncMock(return_value=b"input")
        with pytest.raises(ConversionBusyError, match="хранилище"):
            await service.convert(
                42, read_file, name="x.mp4", mode="video", extension="mp4"
            )
        read_file.assert_not_called()
        assert len(service._jobs) == 4
        await service.close()

    asyncio.run(scenario())


def test_input_output_size_limits_are_enforced_before_delivery(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    _video(source)
    with pytest.raises(ValueError, match="100 МБ"):
        convert_media(
            source,
            tmp_path / "out.mp4",
            mode="video",
            extension="mp4",
            max_input_bytes=1,
        )
    with pytest.raises(ValueError, match="Результат"):
        convert_media(
            source,
            tmp_path / "out.mp4",
            mode="video",
            extension="mp4",
            max_output_bytes=1,
        )


def test_cleanup_loop_removes_expired_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import protogen_delta.services.media_conversion as module

    async def scenario() -> None:
        service = MediaConversionService(clock=lambda: 1000)
        cleanup = Mock()
        service._jobs["expired"] = Mock(
            user_id=42, expires_at=0, directory=SimpleNamespace(cleanup=cleanup)
        )
        monkeypatch.setattr(
            module.asyncio,
            "sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError()]),
        )
        with pytest.raises(asyncio.CancelledError):
            await service._cleanup_loop()
        cleanup.assert_called_once()
        assert not service._jobs
        await service.close()

    asyncio.run(scenario())
