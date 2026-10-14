"""Публичное чтение GitHub, закреплённые строки и изоляция выбранного проекта."""

import asyncio
import base64
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientError
from test_deepseek_service import _create_service
from test_multiple_links import _plain_response
from test_response_engine import _create_engine

from protogen_delta.core.llm_budget import current_llm_chat, llm_user
from protogen_delta.services import public_github as module
from protogen_delta.services.link_review import compact_page_result
from protogen_delta.services.public_github import (
    GITHUB_TOOLS,
    GithubNotFound,
    PublicGithubError,
    PublicGithubReader,
    _safe_path,
    github_links,
    repository_url,
)
from protogen_delta.services.tools import ToolExecutor, default_registry

REVISION, TREE, README, CODE, SUBTREE = (letter * 40 for letter in "abcde")
URL = "https://github.com/acme/project"


def entry(
    name: str, sha: str, *, kind: str = "blob", mode: str = "100644", size: int = 200
) -> dict[str, Any]:
    """Создать синтетические метаданные обычного файла или каталога."""
    return {"path": name, "sha": sha, "type": kind, "mode": mode, "size": size}


class FixtureReader(PublicGithubReader):
    """Подставить дерево и текст без сети, исполнения исходников и настоящей модели."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.calls: list[str] = []
        self.data: dict[str, dict[str, Any]] = {
            "/repos/acme/project": {
                "private": False,
                "default_branch": "main",
                "language": "Python",
                "description": "Проверочный проект",
            },
            "/repos/acme/project/branches/main": {
                "commit": {"sha": REVISION, "commit": {"tree": {"sha": TREE}}}
            },
            "/repos/acme/project/git/commits/"
            + REVISION: {"sha": REVISION, "tree": {"sha": TREE}},
            "/repos/acme/project/git/trees/"
            + TREE: {
                "tree": [
                    entry("README.md", README),
                    entry("main.py", CODE),
                    entry("src", SUBTREE, kind="tree", mode="040000"),
                    entry("link.py", CODE, mode="120000"),
                    entry("module", CODE, kind="commit", mode="160000"),
                ]
            },
            "/repos/acme/project/git/trees/"
            + SUBTREE: {"tree": [entry("memory.py", CODE)]},
            "/repos/acme/project/git/blobs/"
            + README: {
                "encoding": "base64",
                "content": base64.b64encode(b"# Fixture\nREADME_MARKER").decode(),
            },
            "/repos/acme/project/git/blobs/"
            + CODE: {
                "encoding": "base64",
                "content": base64.b64encode(
                    b'API_KEY = "secret_fixture"\ndef save_memory():\n    return "CODE_MARKER"\n'
                ).decode(),
            },
        }

    async def _get(self, path: str) -> dict[str, Any]:
        self.calls.append(path)
        if path not in self.data:
            raise GithubNotFound("Не найдена синтетическая ревизия.")
        return self.data[path]


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a/b",
        "https://github.com.evil/a/b",
        "https://github.com@evil/a/b",
        "https://github.com:443/a/b",
        "https://github.com/a",
        "https://github.com/a/..",
        "https://github.com/a/.git",
        "https://github.com/a/b?token=fixture",
        "https://github.com/a/b/pull/1",
        "https://github.com/a/b/blob",
        "https://github.com/a/b/tree/..",
        "https://github.com/a/b/tree/" + "x" * 101,
        "https://github.com/a/b/" + "x" * 1500,
    ],
)
def test_only_supported_public_urls(url: str) -> None:
    with pytest.raises(ValueError):
        repository_url(url)


@pytest.mark.parametrize(
    "path",
    [
        "../file",
        "/file",
        "a/../file",
        "a\\file",
        "x:y",
        ".env",
        "a/.env.local",
        "private/file",
        ".git/config",
        "secrets.py",
        "a.key",
        "file\x00",
        "x" * 201,
    ],
)
def test_secret_and_escaping_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        _safe_path(path)


def test_url_lines_and_direct_link_policy() -> None:
    assert repository_url(URL + ".git")[0] == "acme/project"
    assert repository_url(URL + "/blob/main/main.py#L2-L3")[2:] == (
        "main",
        "main.py",
        2,
    )
    assert github_links("Вот проект " + URL + ", " + URL) == (URL,)
    assert github_links("Не открывай " + URL) == ()
    assert github_links("Он написал «" + URL + "»") == ()
    assert github_links("```" + URL + "```") == ()
    assert github_links("Данные <document_content>" + URL) == ()
    assert github_links("https://github.com/acme/project/pull/1") == ()
    assert _safe_path("src/./main.py") == "src/main.py"


def test_repository_file_and_followups_stay_on_one_commit(tmp_path: Path) -> None:
    async def run() -> None:
        reader = FixtureReader(tmp_path)
        with llm_user(42, -100):
            result = await reader.get_repo({"url": URL})
            assert result["available"] and result["revision"] == REVISION
            assert "README_MARKER" in result["readme"]["text"]
            assert REVISION in await reader.context()
            code = await reader.read_file({"path": "src/memory.py", "start": "2"})
            assert "2: def save_memory" in code["text"] and "3: " in code["text"]
            assert "secret_fixture" not in code["text"]
            assert code["url"] == URL + "/blob/" + REVISION + "/src/memory.py#L2"
            assert (await reader.search_files({"path": "src", "query": "memory"}))[
                "matches"
            ][0]["path"] == "src/memory.py"
            assert (await reader.list_tree({"path": "src"}))["entries"][0][
                "path"
            ] == "src/memory.py"
            assert REVISION in await PublicGithubReader(tmp_path).context()
        with llm_user(43, -100):
            assert await reader.context() == ""
            assert not (await reader.list_tree({}))["available"]
        with llm_user(42, -200):
            assert await reader.context() == ""
        with llm_user(42, -100):
            await reader.clear_user(43)
            assert await reader.context()
            await reader.clear_user(42)
            assert await reader.context() == ""
        assert current_llm_chat.get() is None

    asyncio.run(run())


def test_file_link_prefetch_contains_requested_lines(tmp_path: Path) -> None:
    async def run() -> None:
        reader = FixtureReader(tmp_path)
        executor = ToolExecutor(default_registry(github_reader=reader))
        raw = await executor.execute(
            "github_get_repo", json.dumps({"url": URL + "/blob/main/main.py#L2"})
        )
        result = json.loads(raw)
        assert result["ok"]
        compact, status = compact_page_result(raw)
        assert "CODE_MARKER" in compact and "2: def save_memory" in compact
        assert "указанного файла" in status
        assert "secret_fixture" not in raw

    asyncio.run(run())


@pytest.mark.parametrize(
    "case",
    [
        "private",
        "invalid_revision",
        "missing_directory",
        "not_directory",
        "symlink",
        "submodule",
        "large",
        "binary",
        "invalid_base64",
        "wrong_encoding",
        "missing_text",
        "invalid_python",
        "bad_rows",
        "bad_tree_sha",
        "no_readme",
    ],
)
def test_partial_and_unavailable_results_are_honest(tmp_path: Path, case: str) -> None:
    async def run() -> None:
        reader = FixtureReader(tmp_path)
        args = {"url": URL, "path": "main.py"}
        if case == "private":
            reader.data["/repos/acme/project"]["private"] = True
        elif case == "invalid_revision":
            reader.data["/repos/acme/project/branches/main"]["commit"]["sha"] = "bad"
        elif case in {"missing_directory", "not_directory"}:
            args["path"] = (
                "unknown/file" if case == "missing_directory" else "main.py/file"
            )
        elif case in {"symlink", "submodule"}:
            args["path"] = "link.py" if case == "symlink" else "module"
        elif case == "large":
            reader.data["/repos/acme/project/git/trees/" + TREE]["tree"][1][
                "size"
            ] = 400000
        elif case in {
            "binary",
            "invalid_base64",
            "wrong_encoding",
            "missing_text",
            "invalid_python",
        }:
            blob = reader.data["/repos/acme/project/git/blobs/" + CODE]
            if case == "binary":
                blob["content"] = base64.b64encode(b"\0bad").decode()
            elif case == "invalid_base64":
                blob["content"] = "!invalid"
            elif case == "wrong_encoding":
                blob["encoding"] = "raw"
            elif case == "missing_text":
                blob["content"] = None
            else:
                blob["content"] = base64.b64encode(
                    b'API_KEY = "secret_fixture"\nbad syntax\n'
                ).decode()
        elif case == "bad_rows":
            reader.data["/repos/acme/project/git/trees/" + TREE]["tree"] = None
        elif case == "bad_tree_sha":
            reader.data["/repos/acme/project/git/trees/" + TREE]["tree"][2][
                "sha"
            ] = "bad"
            args["path"] = "src/file"
        elif case == "no_readme":
            reader.data["/repos/acme/project/git/blobs/" + README]["encoding"] = "raw"
        if case in {"bad_rows", "no_readme"}:
            result = await reader.get_repo({"url": URL})
        else:
            result = await reader.read_file(args)
        if case == "invalid_python":
            assert result["available"] and "secret_fixture" not in result["text"]
        elif case == "no_readme":
            assert (
                result["available"]
                and result["readme_notice"]
                and "readme" not in result
            )
        else:
            assert (
                result["available"] is False and result["content_scope"] == "not_read"
            )
            assert (
                compact_page_result(json.dumps({"ok": True, "data": result}))[1]
                == "не прочитана"
            )

    asyncio.run(run())


def test_directory_pages_and_result_budget(tmp_path: Path) -> None:
    async def run() -> None:
        reader = FixtureReader(tmp_path)
        reader.data["/repos/acme/project/git/trees/" + TREE]["tree"] = [
            entry(f"src_{i:03d}.py", CODE) for i in range(150)
        ]
        reader.data["/repos/acme/project/git/trees/" + TREE]["tree"].append(
            entry("x" * 250, CODE)
        )
        first = await reader.list_tree({"url": URL})
        second = await reader.list_tree({"url": URL, "offset": first["next_offset"]})
        assert first["truncated"] and first["next_offset"] == "40"
        assert second["entries"][0]["path"] == "src_040.py"
        assert not (
            await reader.search_files({"url": URL, "query": "140", "offset": "120"})
        )["next_offset"]
        executor = ToolExecutor(default_registry(github_reader=reader))
        assert json.loads(
            await executor.execute("github_get_repo", json.dumps({"url": URL}))
        )["ok"]
        for args in (
            {"url": URL, "query": ""},
            {"url": URL, "offset": "-1", "query": "x"},
        ):
            with pytest.raises(ValueError):
                await reader.search_files(args)
        with pytest.raises(ValueError):
            await reader.read_file({"url": URL, "path": "", "start": "0"})

    asyncio.run(run())


@pytest.mark.parametrize(
    "kind", ["branch_slash", "tag", "annotated_tag", "bad_tag", "tag_chain"]
)
def test_revisions_resolve_to_confirmed_commit(tmp_path: Path, kind: str) -> None:
    async def run() -> None:
        reader = FixtureReader(tmp_path)
        if kind == "branch_slash":
            reader.data["/repos/acme/project/branches/feature%2Ftopic"] = reader.data[
                "/repos/acme/project/branches/main"
            ]
            result = await reader.get_repo({"url": URL + "/blob/feature/topic/main.py"})
        else:
            reader.data["/repos/acme/project/git/ref/tags/v1"] = {
                "object": {"type": "commit", "sha": REVISION}
            }
            if kind == "bad_tag":
                reader.data["/repos/acme/project/git/ref/tags/v1"]["object"][
                    "sha"
                ] = "bad"
            if kind in {"annotated_tag", "tag_chain"}:
                reader.data["/repos/acme/project/git/ref/tags/v1"]["object"] = {
                    "type": "tag",
                    "sha": CODE,
                }
                reader.data["/repos/acme/project/git/tags/" + CODE] = {
                    "object": {
                        "type": "commit" if kind == "annotated_tag" else "tag",
                        "sha": REVISION if kind == "annotated_tag" else CODE,
                    }
                }
            result = await reader.get_repo({"url": URL, "ref": "v1"})
        if kind in {"bad_tag", "tag_chain"}:
            assert not result["available"]
        else:
            assert result["available"] and result["revision"] == REVISION
            if kind == "branch_slash":
                assert "CODE_MARKER" in result["file"]["text"]

    asyncio.run(run())


def test_selection_expiry_and_no_activation(tmp_path: Path) -> None:
    async def run() -> None:
        reader = FixtureReader(tmp_path)
        with llm_user(42):
            await reader.get_repo({"url": URL, "activate": "no"})
            assert await reader.context() == ""
            await reader.get_repo({"url": URL})
            assert await reader.context()
            with sqlite3.connect(tmp_path / "memories.db") as db:
                db.execute("UPDATE github_contexts SET updated=0")
            assert await reader.context() == ""
        assert await reader.context() == ""

    asyncio.run(run())


def test_automatic_link_review_and_multiple_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        service, _, create, _ = _create_service(monkeypatch)
        create.return_value = _plain_response("Обзор")
        reader = FixtureReader(tmp_path)
        service._tools = ToolExecutor(default_registry(github_reader=reader))
        with llm_user(42, -100):
            answer = await service.chat("system", "Вот " + URL, tool_names=GITHUB_TOOLS)
            assert "сведения и каталог" in answer and "README_MARKER" in str(
                create.await_args
            )
            assert REVISION in await reader.context()
            create.reset_mock()
            answer = await service.chat(
                "system",
                URL + " https://github.com/acme/missing",
                tool_names=GITHUB_TOOLS,
            )
            assert "не прочитана" in answer
            assert URL in await reader.context()
            assert create.await_count == 1
        before = len(reader.calls)
        await service.chat("system", "Не открывай " + URL, tool_names=GITHUB_TOOLS)
        assert len(reader.calls) == before

    asyncio.run(run())


def test_engine_followup_uses_actual_chat_and_context_failure_is_nonfatal(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        reader = FixtureReader(tmp_path)
        with llm_user(123456, -100):
            await reader.get_repo({"url": URL})
        engine, _, model, _, _, _ = _create_engine()
        engine._github_reader = reader
        await engine.respond(
            user_id=123456, chat_id=-100, user_message="А где там память реализована?"
        )
        assert GITHUB_TOOLS <= model.chat.await_args.kwargs["tool_names"]
        assert REVISION in model.chat.await_args.kwargs["system_prompt"]
        model.chat.reset_mock()
        await engine.respond(user_id=123456, chat_id=-200, user_message="Спасибо")
        assert REVISION not in model.chat.await_args.kwargs["system_prompt"]
        reader.context = AsyncMock(  # type: ignore[method-assign]
            side_effect=sqlite3.OperationalError("fixture failure")
        )
        assert await engine.respond(user_id=123456, user_message="Привет")

    asyncio.run(run())


class HttpResponse:
    """Синтетический поток HTTP без настоящих ключей и соединений."""

    def __init__(self, status: int = 200, raw: bytes = b'{"value":1}') -> None:
        self.status, self.raw = status, raw
        self.content = self

    async def __aenter__(self) -> "HttpResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def iter_chunked(self, size: int) -> Any:
        yield self.raw


@pytest.mark.parametrize(
    "status,raw",
    [
        (404, b"secret"),
        (403, b"secret"),
        (429, b"secret"),
        (302, b"secret"),
        (500, b"secret"),
        (200, b"bad json"),
        (200, b"[]"),
        (200, b"x" * 1048577),
    ],
    ids=["missing", "forbidden", "rate", "redirect", "error", "json", "shape", "large"],
)
def test_http_errors_do_not_leak_provider_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, raw: bytes
) -> None:
    async def run() -> None:
        response = HttpResponse(status, raw)
        response.get = lambda *args, **kwargs: response  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "_session", lambda proxy: response)
        reader = PublicGithubReader(tmp_path)
        with pytest.raises(PublicGithubError) as error:
            await reader._get("/repos/acme/project")
        assert "secret" not in str(error.value)
        await reader.close()

    asyncio.run(run())


def test_http_deduplicates_and_survives_waiter_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        arrived, release = asyncio.Event(), asyncio.Event()
        response = HttpResponse()
        calls: list[dict[str, Any]] = []

        def get(url: str, **kwargs: Any) -> HttpResponse:
            assert url == "https://api.github.com/repos/acme/project"
            assert (
                "Authorization" not in kwargs["headers"]
                and kwargs["allow_redirects"] is False
            )
            calls.append(kwargs)
            return response

        async def limited(value: Any) -> bytes:
            arrived.set()
            await release.wait()
            return b'{"value":1}'

        response.get = get  # type: ignore[attr-defined]
        monkeypatch.setattr(module, "_session", lambda proxy: response)
        monkeypatch.setattr(module, "_read_limited", limited)
        reader = PublicGithubReader(tmp_path)
        first = asyncio.create_task(reader._get("/repos/acme/project"))
        await arrived.wait()
        second = asyncio.create_task(reader._get("/repos/acme/project"))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second == {"value": 1}
        assert await reader._get("/repos/acme/project") == {"value": 1}
        assert len(calls) == 1
        reader._requests.extend([module.monotonic()] * 50)
        with pytest.raises(PublicGithubError):
            await reader._get("/other")
        await reader.close()

    asyncio.run(run())


@pytest.mark.parametrize("exception", [ClientError, TimeoutError])
def test_network_failure_has_safe_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: type[Exception]
) -> None:
    async def run() -> None:
        def session(proxy: str | None) -> Any:
            raise exception("secret request")

        monkeypatch.setattr(module, "_session", session)
        with pytest.raises(PublicGithubError) as error:
            await PublicGithubReader(tmp_path)._get("/repos/acme/project")
        assert "secret" not in str(error.value)

    asyncio.run(run())
