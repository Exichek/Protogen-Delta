"""Локальный Deno в песочнице и временная копия сессии только YouTube."""

import os
import subprocess
import sys
import sysconfig
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Any

from protogen_delta.services.download_errors import DownloadFailure

_COOKIE_BYTES = 256 * 1024
DENO_HEAP_FLAG = "--v8-flags=--max-old-space-size=256"
DENO_RUN_FLAGS = frozenset(
    {
        "--ext=js",
        "--no-code-cache",
        "--no-prompt",
        "--no-remote",
        "--no-lock",
        "--node-modules-dir=none",
        "--no-config",
        "--no-npm",
        "--cached-only",
        DENO_HEAP_FLAG,
    }
)


def deno_binary() -> str:
    # Fixed installed path; no discovery through user PATH or downloaded code.
    path = Path(sysconfig.get_path("scripts")) / (
        "deno.exe" if os.name == "nt" else "deno"
    )
    if not path.is_file():
        raise DownloadFailure("runtime")
    return str(path.resolve())


def allowed_deno_command(executable: Any, argv: Any, runtime: str | None) -> bool:
    # Windows audits the rendered command line, with executable often None.
    # Only our fixed quoted binary prefix and space-free known flags are parsed.
    if os.name == "nt" and runtime and isinstance(argv, str):
        prefix = subprocess.list2cmdline([runtime]) + " "
        if not argv.startswith(prefix):
            return False
        argv = [runtime, *argv.removeprefix(prefix).split(" ")]
        if executable is None:
            executable = runtime
    if (
        not runtime
        or not isinstance(executable, str)
        or not isinstance(argv, (list, tuple))
        or not all(isinstance(arg, str) for arg in argv)
        or not argv
        or executable != runtime
        or argv[0] != runtime
    ):
        return False
    if list(argv[1:]) == ["--version"]:
        return True
    # No eval, file script, remote modules, permission grants or unknown flags.
    return (
        len(argv) == len(DENO_RUN_FLAGS) + 3
        and argv[1] == "run"
        and argv[-1] == "-"
        and set(argv[2:-1]) == DENO_RUN_FLAGS
    )


def youtube_options() -> dict[str, Any]:
    from yt_dlp.extractor.youtube.jsc._builtin.deno import (  # type: ignore[import-untyped]
        DenoJCP,
    )

    # yt-dlp is pinned. Bound the V8 heap in this one-shot process only.
    if DENO_HEAP_FLAG not in DenoJCP._DENO_BASE_OPTIONS:
        DenoJCP._DENO_BASE_OPTIONS = [*DenoJCP._DENO_BASE_OPTIONS, DENO_HEAP_FLAG]
    return {"js_runtimes": {"deno": {"path": deno_binary()}}, "remote_components": []}


def limit_extraction_process() -> None:
    """V8 reserves large address space; limit writable data rather than AS."""
    if sys.platform.startswith("linux"):
        import resource

        resource.setrlimit(resource.RLIMIT_DATA, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def snapshot_youtube_cookies(source: Path, target: Path) -> None:
    """Исходник не перезаписывается; чужие домены не передаются загрузчику."""
    try:
        if not source.is_file():
            raise ValueError("not_regular")
        with source.open("rb") as stream:
            raw = stream.read(_COOKIE_BYTES + 1)
        if len(raw) > _COOKIE_BYTES:
            raise ValueError("size")
        target.write_bytes(raw)
        target.chmod(0o600)
        jar = MozillaCookieJar(str(target))
        jar.load(ignore_discard=True, ignore_expires=True)
        for cookie in list(jar):
            # Netscape exporters use zero for session cookies, as yt-dlp does.
            if cookie.expires == 0:
                cookie.expires = None
            domain = cookie.domain.lstrip(".").casefold()
            if cookie.is_expired() or (
                domain != "youtube.com" and not domain.endswith(".youtube.com")
            ):
                jar.clear(cookie.domain, cookie.path, cookie.name)
        if not list(jar):
            raise ValueError("no_youtube_session")
        jar.save(ignore_discard=True, ignore_expires=False)
    except OSError, ValueError:
        target.unlink(missing_ok=True)
        raise DownloadFailure("cookies") from None
