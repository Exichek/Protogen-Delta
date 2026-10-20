"""Подлинный ZIP, ресурсы, секреты и авторизованная загрузка в Mini App."""

import asyncio
import io
import json
import stat
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from aiohttp.test_utils import TestClient, TestServer
from test_miniapp import TOKEN, _signed_init_data

import protogen_delta.services.archive_documents as archives
from protogen_delta.core.runtime_health import RuntimeHealth
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.miniapp.server import MiniAppServer
from protogen_delta.services.documents import (
    DocumentReadError,
    DocumentTooLargeError,
    extract_document,
)
from protogen_delta.services.native_work import NativeWorkPool
from protogen_delta.services.response_engine import ResponseEngine


def _zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_zip_reads_code_and_lists_unsupported_nested_and_secret_files(
    tmp_path: Path,
) -> None:
    data = _zip(
        {
            "src/main.py": b"print('ARCHIVE_CODE_MARKER')",
            "readme.md": "Привет".encode(),
            "photo.png": b"\x00PNG",
            "inside.zip": _zip({"deep.txt": b"NOT_READ"}),
            "../evil.txt": b"PATH_MARKER",
            ".env": b"SECRET_MARKER",
            "keys/server.key": b"KEY_MARKER",
        }
    )
    result = extract_document(data, "example.zip", "application/zip")
    assert (
        result.kind == "ZIP"
        and "ARCHIVE_CODE_MARKER" in result.text
        and "Привет" in result.text
    )
    assert all(
        marker not in result.text
        for marker in ["NOT_READ", "PATH_MARKER", "SECRET_MARKER", "KEY_MARKER"]
    )
    assert "вложенный архив" in result.text and "формат не прочитан" in result.text
    assert list(tmp_path.iterdir()) == []


def test_zip_symlink_is_never_read() -> None:
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        info = ZipInfo("link.txt")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "TARGET_MARKER")
    assert "TARGET_MARKER" not in archives.extract_zip(output.getvalue()).text


def test_small_repository_with_over_200_entries_is_read_and_listing_keeps_text() -> (
    None
):
    """Малый проект с сотнями файлов не считается слишком большим из-за числа записей."""
    entries = {"main.py": b"PROJECT_CONTENT_MARKER"}
    entries.update(
        {f"src/{'long_name_' * 10}{index}.py": b"pass" for index in range(377)}
    )
    result = archives.extract_zip(_zip(entries))
    assert result.truncated and "PROJECT_CONTENT_MARKER" in result.text
    assert "378 записей" in result.text and len(result.text) <= 60000
    assert "Список имён показан частично" in result.text


def test_heavy_nested_document_is_skipped_without_rejecting_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Превышение лимита одного DOCX не мешает прочитать соседние исходники."""
    from protogen_delta.services.documents import ExtractedDocument

    def read(data: bytes, name: str, mime: str | None) -> ExtractedDocument:
        if name.endswith(".docx"):
            raise DocumentTooLargeError("Office limit")
        return ExtractedDocument("SURVIVING_CODE", "TEXT", False)

    monkeypatch.setattr(archives, "extract_document", read)
    result = archives.extract_zip(_zip({"large.docx": b"office", "main.py": b"code"}))
    assert result.truncated and "SURVIVING_CODE" in result.text
    assert "внутренний документ превышает лимиты" in result.text


def test_archive_limit_reason_survives_real_worker() -> None:
    """Родитель получает число записей, а не общую ошибку про мегабайты."""

    async def run() -> None:
        data = _zip({f"{index}.txt": b"hi" for index in range(2001)})
        with pytest.raises(archives.ArchiveLimitError, match="2001 записей") as caught:
            await NativeWorkPool().run(archives.extract_zip, data)
        assert caught.value.reason == "entries"

    asyncio.run(run())


@pytest.mark.parametrize(
    "reason,actual", [("bad", 1), ("entries", True), ("expanded", -1)]
)
def test_invalid_archive_limit_metadata_is_rejected(reason: str, actual: int) -> None:
    from protogen_delta.services.native_work import NativeWorkError

    with pytest.raises(ValueError):
        archives.ArchiveLimitError(reason, actual)
    with pytest.raises(NativeWorkError):
        NativeWorkPool._decode(
            "archive", {"error": "archive_limit", "reason": reason, "actual": actual}
        )


@pytest.mark.parametrize(
    "field,value", [("MAX_ZIP_BYTES", 3), ("MAX_ZIP_EXPANDED", 3), ("MAX_ZIP_FILES", 0)]
)
def test_zip_limits_are_checked_before_read(
    monkeypatch: pytest.MonkeyPatch, field: str, value: int
) -> None:
    data = _zip({"a.txt": b"text"})
    monkeypatch.setattr(archives, field, value)
    with pytest.raises(DocumentTooLargeError):
        archives.extract_zip(data)


def test_zip_read_count_is_explicitly_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archives, "MAX_ZIP_READ_FILES", 1)
    result = archives.extract_zip(
        _zip({"a.txt": b"FIRST_MARKER", "b.txt": b"SECOND_MARKER"})
    )
    assert result.truncated and "b.txt" in result.text
    assert "FIRST_MARKER" in result.text and "SECOND_MARKER" not in result.text


def test_bad_and_encrypted_archives_are_rejected() -> None:
    with pytest.raises(DocumentReadError):
        archives.extract_zip(b"not a zip")
    raw = bytearray(_zip({"a.txt": b"hi"}))
    central = raw.index(b"PK\x01\x02")
    raw[central + 8] |= 1
    with pytest.raises(DocumentReadError, match="паролем"):
        archives.extract_zip(bytes(raw))


def test_real_archive_worker_accepts_over_twenty_mb_without_raising_other_limits() -> (
    None
):
    async def run() -> None:
        data = b"\0" * (21 * 1024 * 1024) + _zip(
            {"main.py": b"print('LARGE_ZIP_MARKER')"}
        )
        result = await NativeWorkPool().run(archives.extract_zip, data)
        assert result.kind == "ZIP" and "LARGE_ZIP_MARKER" in result.text
        with pytest.raises(DocumentTooLargeError):
            await NativeWorkPool().run(
                extract_document, data, "archive.zip", "application/zip"
            )

    asyncio.run(run())


def _headers(
    *, analyze: bool = False, question: str = "", owner: int = 42
) -> dict[str, str]:
    return {
        "X-Telegram-Init-Data": _signed_init_data(user_id=owner),
        "X-Archive-Options": quote(
            json.dumps(
                {"file_name": "example.zip", "question": question, "analyze": analyze}
            ),
            safe="",
        ),
        "Content-Type": "application/zip",
    }


@pytest.mark.parametrize("analyze", [False, True])
def test_miniapp_archive_is_owned_and_model_only_runs_when_requested(
    analyze: bool,
) -> None:
    async def run() -> None:
        engine = AsyncMock(spec=ResponseEngine)
        engine.respond.return_value = "Ответ по ZIP"
        app = MiniAppServer(
            TOKEN, UserStateStore(), response_engine=cast(ResponseEngine, engine)
        ).application()
        async with TestClient(TestServer(app)) as client:
            data = _zip({"a.txt": b"OWNER_ARCHIVE_MARKER"})
            response = await client.post(
                "/api/tools/archive",
                data=data,
                headers=_headers(analyze=analyze, question="Что тут?", owner=77),
            )
            assert response.status == 200
            result = await response.json()
            assert "OWNER_ARCHIVE_MARKER" in result["text"]
            assert response.headers["Cache-Control"] == "no-store"
            if analyze:
                engine.respond.assert_awaited_once()
                assert engine.respond.await_args.args == (77, "Что тут?")
                assert (
                    "OWNER_ARCHIVE_MARKER"
                    in engine.respond.await_args.kwargs["attachment_text"]
                )
                assert result["reply"] == "Ответ по ZIP"
            else:
                engine.respond.assert_not_awaited()
                assert result["reply"] == ""

    asyncio.run(run())


@pytest.mark.parametrize(
    "kind,status",
    [("no_auth", 401), ("bad_zip", 400), ("bad_options", 400), ("disabled", 503)],
)
def test_miniapp_rejects_invalid_archives_before_model(kind: str, status: int) -> None:
    async def run() -> None:
        engine = AsyncMock(spec=ResponseEngine)
        server = MiniAppServer(
            TOKEN,
            UserStateStore(),
            response_engine=(
                None if kind == "disabled" else cast(ResponseEngine, engine)
            ),
        )
        async with TestClient(TestServer(server.application())) as client:
            headers = _headers(analyze=kind == "disabled")
            if kind == "no_auth":
                headers.pop("X-Telegram-Init-Data")
            if kind == "bad_options":
                headers["X-Archive-Options"] = "invalid"
            response = await client.post(
                "/api/tools/archive",
                data=b"bad" if kind == "bad_zip" else _zip({"a.txt": b"hi"}),
                headers=headers,
            )
            assert response.status == status
            engine.respond.assert_not_awaited()
            assert not server._archive_pending

    asyncio.run(run())


def test_archive_upload_is_visible_to_health_until_work_finishes() -> None:
    async def run() -> None:
        health = RuntimeHealth()
        health.polling_succeeded()
        entered, release = asyncio.Event(), asyncio.Event()
        pool = AsyncMock(spec=NativeWorkPool)

        async def read(*args: object) -> object:
            entered.set()
            await release.wait()
            return archives.extract_zip(_zip({"a.txt": b"text"}))

        pool.run.side_effect = read
        server = MiniAppServer(
            TOKEN, UserStateStore(), native_work=pool, runtime_health=health
        )
        async with TestClient(TestServer(server.application())) as client:
            task = asyncio.create_task(
                client.post(
                    "/api/tools/archive",
                    data=_zip({"a.txt": b"text"}),
                    headers=_headers(),
                )
            )
            await asyncio.wait_for(entered.wait(), 2)
            assert health.snapshot()["diagnostics"]["archives"]["active"] == 1
            assert 42 in server._archive_pending
            repeat = await client.post(
                "/api/tools/archive", data=b"unused", headers=_headers()
            )
            assert repeat.status == 429
            release.set()
            result = await task
            assert result.status == 200
            assert health.snapshot()["diagnostics"]["archives"]["active"] == 0
            assert not server._archive_pending and pool.run.await_count == 1

    asyncio.run(run())
