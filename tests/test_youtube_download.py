"""YouTube runtime, ограниченные ошибки и приватная сессия вне LLM/других сайтов."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from test_download_compression import _mp4
from test_media_download import _stop_fake_download

import protogen_delta.services.media_download as download
import protogen_delta.services.media_download_worker as worker
import protogen_delta.services.youtube_download as youtube
from protogen_delta.services.download_errors import (
    DownloadFailure,
    classify_failure,
    read_failure,
)


def _cookies(path: Path) -> bytes:
    raw = (
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tsynthetic-session\n"
        ".youtube.com\tTRUE\t/\tTRUE\t1\tOLD\texpired-session\n"
        ".google.com\tTRUE\t/\tTRUE\t0\tSID\tunrelated-google\n"
        ".youtube.com.evil.test\tTRUE\t/\tTRUE\t0\tBAD\tunrelated-domain\n"
    ).encode()
    path.write_bytes(raw)
    return raw


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Sign in to confirm you're not a bot. token=synthetic", "youtube_bot"),
        ("Sign in to confirm your age.", "youtube_age"),
        ("Login required", "login"),
        ("Private video", "unavailable"),
        ("This video is unavailable", "unavailable"),
        ("Requested format is not available", "formats"),
        ("HTTP Error 429", "rate_limited"),
        ("HTTP Error 403", "access_denied"),
        ("HTTP Error 404", "unavailable"),
        ("No supported JavaScript runtime could be found", "runtime"),
        ("Connection timed out", "network"),
        ("opaque token=synthetic", "unknown"),
    ],
)
def test_external_errors_become_static_codes(message: str, expected: str) -> None:
    assert classify_failure(RuntimeError(message)) == expected
    assert classify_failure(DownloadFailure(expected)) == expected
    assert DownloadFailure("token=synthetic").code == "unknown"


@pytest.mark.parametrize(
    "value", [[], {}, {"error": []}, {"error": "secret"}, "x" * 514]
)
def test_parent_rejects_malformed_error_protocol(tmp_path: Path, value: Any) -> None:
    assert read_failure(tmp_path) == "unknown"
    path = tmp_path / "error.json"
    path.write_text(json.dumps(value), "utf-8")
    assert read_failure(tmp_path) == "unknown"
    path.write_text('{"error":"youtube_age"}', "utf-8")
    assert read_failure(tmp_path) == "youtube_age"
    path.write_bytes(b"\xff")
    assert read_failure(tmp_path) == "unknown"


def test_deno_guard_accepts_only_pinned_stdin_sandbox() -> None:
    runtime = youtube.deno_binary()
    args = [runtime, "run", *sorted(youtube.DENO_RUN_FLAGS), "-"]
    assert youtube.allowed_deno_command(runtime, args, runtime)
    assert youtube.allowed_deno_command(runtime, [runtime, "--version"], runtime)
    for bad in (
        [runtime, "eval", "1+1"],
        [*args[:-1], "script.js"],
        [*args[:-1], "--allow-net", "-"],
        [*args[:-1], "--allow-read", "-"],
        [runtime, "run", "-"],
        [runtime, "--version", "script.js"],
        "deno run script.js",
        [],
        [1],
    ):
        assert not youtube.allowed_deno_command(runtime, bad, runtime)
    assert not youtube.allowed_deno_command("other", args, runtime)
    assert not youtube.allowed_deno_command(runtime, args, None)


def test_network_guard_keeps_external_processes_and_lan_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = Mock()
    monkeypatch.setattr(worker.sys, "addaudithook", hook)
    runtime = youtube.deno_binary()
    worker.install_network_guard(runtime)
    audit = hook.call_args.args[0]
    args = [runtime, "run", *sorted(youtube.DENO_RUN_FLAGS), "-"]
    audit("subprocess.Popen", (runtime, args, None, {}))
    audit("os.posix_spawn", (runtime, args, {}))
    for event, data in (
        ("subprocess.Popen", ("ffmpeg", ["ffmpeg"], None, {})),
        ("os.posix_spawn", ("sh", ["sh"], {})),
        ("os.system", ("deno --version",)),
        ("socket.connect", (None, ("127.0.0.1", 80))),
    ):
        with pytest.raises(OSError):
            audit(event, data)


def test_real_deno_runs_under_guard_without_file_or_network_permissions() -> None:
    code = """
import subprocess,json
from protogen_delta.services.youtube_download import deno_binary,DENO_RUN_FLAGS,limit_extraction_process
from protogen_delta.services.media_download_worker import install_network_guard
runtime=deno_binary()
install_network_guard(runtime)
limit_extraction_process()
js=(
 'let denied=0;try{Deno.readTextFileSync("missing-private-file")}catch(e){'
 'if(e.name==="NotCapable")denied++}try{await fetch("https://example.com")}catch(e){'
 'if(e.name==="NotCapable")denied++}console.log(JSON.stringify({sum:1+1,denied}));'
)
result=subprocess.run([runtime,'run',*sorted(DENO_RUN_FLAGS),'-'],input=js,text=True,capture_output=True,timeout=15)
assert result.returncode==0,result.stderr
assert json.loads(result.stdout)=={'sum':2,'denied':2}
print('sandbox_ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=25
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "sandbox_ok"


def test_cookie_snapshot_filters_domains_expiry_and_preserves_source(
    tmp_path: Path,
) -> None:
    source, target = tmp_path / "private.txt", tmp_path / "copy.txt"
    original = _cookies(source)
    youtube.snapshot_youtube_cookies(source, target)
    copied = target.read_text("utf-8")
    assert "synthetic-session" in copied
    assert "expired-session" not in copied and "unrelated" not in copied
    assert source.read_bytes() == original
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "data",
    [b"", b"broken", b"x" * (256 * 1024 + 1)],
    ids=["empty", "bad-header", "oversized"],
)
def test_invalid_session_has_no_raw_data_in_error(tmp_path: Path, data: bytes) -> None:
    source, target = tmp_path / "private.txt", tmp_path / "copy.txt"
    source.write_bytes(data)
    with pytest.raises(DownloadFailure, match="^cookies$"):
        youtube.snapshot_youtube_cookies(source, target)
    assert not target.exists()
    with pytest.raises(DownloadFailure):
        youtube.snapshot_youtube_cookies(tmp_path / "missing", target)


def test_cookies_are_copied_only_for_youtube_and_removed_after_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "private.txt"
    original = _cookies(source)
    seen: list[Path] = []

    async def run(url: str, directory: Path) -> None:
        cookie = directory / "youtube-cookies.txt"
        assert cookie.exists() is download.is_youtube_url(url)
        if cookie.exists():
            seen.append(cookie)
        (directory / "video.mp4").write_bytes(b"video")
        (directory / "result.json").write_text('{"title":"Synthetic"}', "utf-8")

    async def scenario() -> None:
        service = download.MediaDownloader(source)
        monkeypatch.setattr(service, "_run", run)
        for url in ("https://youtu.be/abc", "https://x.com/a/status/123"):
            async with service.download(url):
                pass

    asyncio.run(scenario())
    assert len(seen) == 1 and not seen[0].exists()
    assert source.read_bytes() == original


def test_cancel_waits_for_cookie_copy_before_temp_directory_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, proceed = threading.Event(), threading.Event()
    targets: list[Path] = []

    def copy(source: Path, target: Path) -> None:
        started.set()
        assert proceed.wait(5)
        target.write_bytes(b"synthetic-session")
        targets.append(target)

    monkeypatch.setattr(download, "snapshot_youtube_cookies", copy)

    async def scenario() -> None:
        service = download.MediaDownloader(tmp_path / "private")
        run = AsyncMock()
        monkeypatch.setattr(service, "_run", run)

        async def request() -> None:
            async with service.download("https://youtu.be/abc"):
                pass

        task = asyncio.create_task(request())
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        run.assert_not_awaited()

    asyncio.run(scenario())
    assert targets and not targets[0].exists()


@pytest.mark.parametrize(
    "code", ["youtube_bot", "youtube_age", "too_large", "too_long"]
)
def test_worker_codes_reach_parent_without_external_error_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    (tmp_path / "error.json").write_text(json.dumps({"error": code}), "utf-8")
    process = SimpleNamespace(returncode=1, wait=AsyncMock(), kill=Mock())
    monkeypatch.setattr(download, "_stop_worker", _stop_fake_download)
    monkeypatch.setattr(
        download.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    with pytest.raises(download.MediaDownloadError) as error:
        asyncio.run(download.MediaDownloader()._run("https://youtu.be/abc", tmp_path))
    assert "сайт ограничил загрузку" not in str(error.value)


@pytest.mark.parametrize("code", ["youtube_bot", "youtube_age", "too_large", "formats"])
def test_entrypoint_writes_only_failure_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    for name in (
        "install_network_guard",
        "limit_extraction_process",
        "limit_compression_process",
    ):
        monkeypatch.setattr(worker, name, Mock())
    monkeypatch.setattr(worker.signal, "signal", Mock())
    monkeypatch.setattr(
        worker.sys, "argv", ["worker", str(tmp_path), "https://youtu.be/a"]
    )
    monkeypatch.setattr(worker, "download_one", Mock(side_effect=DownloadFailure(code)))
    with pytest.raises(SystemExit):
        worker.main()
    assert json.loads((tmp_path / "error.json").read_text("utf-8")) == {"error": code}


def test_v8_limits_are_data_cpu_and_core_not_virtual_address_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = SimpleNamespace(
        RLIMIT_DATA=1, RLIMIT_CPU=2, RLIMIT_CORE=3, setrlimit=Mock()
    )
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(youtube, "sys", SimpleNamespace(platform="linux"))
    youtube.limit_extraction_process()
    assert resource.setrlimit.call_args_list == [
        ((1, (1024**3, 1024**3)),),
        ((2, (180, 180)),),
        ((3, (0, 0)),),
    ]


def test_restricted_youtube_without_formats_reports_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yt_dlp  # type: ignore[import-untyped]

    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=None)
    client.extract_info.return_value = {"duration": 60, "formats": [], "age_limit": 18}
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Mock(return_value=client))
    with pytest.raises(DownloadFailure, match="youtube_age"):
        worker.download_one(tmp_path, "https://youtu.be/synthetic")


def test_progress_scan_tolerates_remux_rename_and_excludes_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, part = tmp_path / "video.mp4", tmp_path / "gone.part"
    video.write_bytes(b"media")
    (tmp_path / "youtube-cookies.txt").write_bytes(b"private-cookie")
    original = Path.stat

    def metadata(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if self == part:
            raise FileNotFoundError
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", metadata)
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda self, pattern: iter((part, video, tmp_path / "youtube-cookies.txt")),
    )
    assert download.directory_bytes(tmp_path) == 5


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/synthetic",
        "https://vkvideo.ru/video-1_123",
        "https://www.instagram.com/reel/synthetic/",
        "https://www.tiktok.com/@synthetic/video/123",
        "https://vimeo.com/123",
        "https://x.com/synthetic/status/123",
    ],
)
def test_all_platforms_share_real_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    for name in (
        "install_network_guard",
        "limit_extraction_process",
        "limit_compression_process",
    ):
        monkeypatch.setattr(worker, name, Mock())
    monkeypatch.setattr(worker.signal, "signal", Mock())
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(tmp_path), url])

    def write(directory: Path, *_: Any, **kwargs: Any) -> None:
        source = directory / "video.mp4"
        _mp4(source, True)
        with source.open("r+b") as file:
            file.seek(0, os.SEEK_END)
            padding = worker.MAX_UPLOAD_BYTES + 1 - file.tell()
            file.write(padding.to_bytes(4, "big") + b"free")
            file.truncate(worker.MAX_UPLOAD_BYTES + 1)
        (directory / "result.json").write_text('{"title":"Synthetic"}', "utf-8")

    monkeypatch.setattr(worker, "download_one", write)
    worker.main()
    assert not (tmp_path / "video.mp4").exists()
    assert 0 < (tmp_path / "telegram.mp4").stat().st_size < worker.MAX_UPLOAD_BYTES
    assert json.loads((tmp_path / "result.json").read_text("utf-8"))["compressed"]


def test_too_large_input_is_rejected_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "install_network_guard",
        "limit_extraction_process",
        "limit_compression_process",
    ):
        monkeypatch.setattr(worker, name, Mock())
    monkeypatch.setattr(worker.signal, "signal", Mock())
    monkeypatch.setattr(
        worker.sys, "argv", ["worker", str(tmp_path), "https://vimeo.com/1"]
    )

    def write(directory: Path, *_: Any, **kwargs: Any) -> None:
        with (directory / "video.mp4").open("wb") as file:
            file.truncate(worker.MAX_DOWNLOAD_BYTES + 1)

    encode = Mock()
    monkeypatch.setattr(worker, "download_one", write)
    monkeypatch.setattr(worker, "compress_download", encode)
    with pytest.raises(SystemExit):
        worker.main()
    encode.assert_not_called()
    assert read_failure(tmp_path) == "too_large"


@pytest.mark.parametrize("seconds,code", [(601, "too_long"), (5, "too_large")])
def test_limits_are_reported_before_media_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: int, code: str
) -> None:
    import yt_dlp

    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=None)
    client.extract_info.return_value = {
        "duration": seconds,
        "formats": [{"vcodec": "avc1", "filesize": worker.MAX_DOWNLOAD_BYTES + 1}],
    }
    monkeypatch.setattr(yt_dlp, "YoutubeDL", Mock(return_value=client))
    with pytest.raises(DownloadFailure, match=code):
        worker.download_one(tmp_path, "https://vimeo.com/123")
    client.process_info.assert_not_called()
    client.dl.assert_not_called()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process group and /proc")
def test_cancellation_reaps_running_deno_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = """
import signal,subprocess,sys
from pathlib import Path
from protogen_delta.services.youtube_download import deno_binary,DENO_RUN_FLAGS
from protogen_delta.services.media_download_worker import install_network_guard
from yt_dlp.utils import Popen
def terminate(signum,frame):raise SystemExit(1)
signal.signal(signal.SIGTERM,terminate)
runtime=deno_binary()
install_network_guard(runtime)
with Popen([runtime,'run',*sorted(DENO_RUN_FLAGS),'-'],text=True,
           stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as child:
    Path(sys.argv[1],'child.pid').write_text(str(child.pid))
    child.communicate_or_kill('while(true){}')
"""

    async def scenario() -> None:
        start = asyncio.create_subprocess_exec

        async def launch(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            return await start(sys.executable, "-c", code, str(tmp_path), **kwargs)

        monkeypatch.setattr(download.asyncio, "create_subprocess_exec", launch)
        task = asyncio.create_task(
            download.MediaDownloader()._run("https://youtu.be/abc", tmp_path)
        )
        async with asyncio.timeout(5):
            while not (tmp_path / "child.pid").exists():
                await asyncio.sleep(0.01)
        pid = int((tmp_path / "child.pid").read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    asyncio.run(scenario())
