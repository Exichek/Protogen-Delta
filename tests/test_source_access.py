"""Исходники доступны только доверенному личному запросу, без файлов данных."""

import asyncio
import sys
from pathlib import Path

import pytest
from test_response_engine import _create_engine

from protogen_delta.services.source_access import (
    SourceAccess,
    current_source_owner,
    source_owner,
)
from protogen_delta.services.tools import ToolExecutor, default_registry


def source(tmp_path: Path) -> SourceAccess:
    (tmp_path / "services").mkdir(exist_ok=True)
    (tmp_path / "services/sample.py").write_text(
        "value = 42\n" + "# line\n" * 200, "utf-8"
    )
    (tmp_path / ".env").write_text("API_KEY=PRIVATE_SECRET", "utf-8")
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config/private.py").write_text("secret = 'PRIVATE_SECRET'", "utf-8")
    return SourceAccess(frozenset({10}), tmp_path)


@pytest.mark.parametrize("with_as", [False, True])
def test_python314_exception_syntax_is_parsed_without_executing(
    tmp_path: Path,
    with_as: bool,
) -> None:
    async def run() -> None:
        access = source(tmp_path)
        suffix = " as error" if with_as else ""
        # Если файл импортировать, эта строка упадёт. Обзор только разбирает AST.
        (tmp_path / "services/sample.py").write_text(
            "raise RuntimeError('Не исполнять исходник')\n"
            "try:\n    pass\n"
            f"except TypeError, ValueError, KeyError{suffix}:\n    pass\n",
            "utf-8",
        )
        if with_as:
            assert not access.snapshot_sync()
            return
        with source_owner(10):
            result = await access.read_tool({"path": "services/sample.py"})
        assert "except TypeError, ValueError, KeyError:" in result["source"]
        assert result["python_version"] == ".".join(
            str(part) for part in sys.version_info[:3]
        )
        assert "ast.parse до маскирования" in result["syntax_check"]

    asyncio.run(run())


def test_canonical_alias_to_private_directory_is_not_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access = source(tmp_path)
    original = Path.resolve
    public = tmp_path / "services/sample.py"

    def resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == public:
            return tmp_path / "data/private.py"
        return original(path)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert access.snapshot_sync() == ()


@pytest.mark.parametrize(
    "name",
    [".env.py", "secrets.py", "credentials.py", "keys.py", "tokens.py", "env.py"],
)
def test_explicit_secret_file_denylist_applies_to_read_and_search(
    tmp_path: Path, name: str
) -> None:
    async def run() -> None:
        access = source(tmp_path)
        (tmp_path / "services" / name).write_text(
            "anything='SHOULD_NEVER_BE_INDEXED'", "utf-8"
        )
        with source_owner(10):
            with pytest.raises(ValueError):
                await access.read_tool({"path": "services/" + name})
            assert not (await access.search_tool({"query": "SHOULD_NEVER_BE_INDEXED"}))[
                "matches"
            ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "declaration",
    [
        "def send(api_key='PRIVATE_SECRET'): pass",
        "async def send(*, password='PRIVATE_SECRET'): pass",
        "config={'credentials': {'value': 'PRIVATE_SECRET'}}",
    ],
)
def test_defaults_and_nested_secret_literals_are_hidden(
    tmp_path: Path, declaration: str
) -> None:
    access = source(tmp_path)
    (tmp_path / "services/sample.py").write_text(declaration, "utf-8")
    assert "PRIVATE_SECRET" not in access.snapshot_sync()[0].text


def test_actual_runtime_credential_is_hidden_even_under_unrelated_variable_name(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        source(tmp_path)
        (tmp_path / "services/sample.py").write_text(
            "value='FAKE_RUNTIME_CREDENTIAL_997'", "utf-8"
        )
        access = SourceAccess(
            frozenset({10}), tmp_path, secrets=("FAKE_RUNTIME_CREDENTIAL_997",)
        )
        with source_owner(10):
            assert (
                "FAKE_RUNTIME_CREDENTIAL_997"
                not in (await access.read_tool({"path": "services/sample.py"}))[
                    "source"
                ]
            )
            assert not (
                await access.search_tool({"query": "FAKE_RUNTIME_CREDENTIAL_997"})
            )["matches"]

    asyncio.run(run())


@pytest.mark.parametrize("owner", [None, 11])
def test_source_requires_trusted_owner(tmp_path: Path, owner: int | None) -> None:
    async def run() -> None:
        access = source(tmp_path)
        with source_owner(owner), pytest.raises(ValueError):
            await access.read_tool({"path": "services/sample.py"})
        with source_owner(owner), pytest.raises(ValueError):
            await access.search_tool({"query": "value"})
        assert current_source_owner.get() is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "path", ["../.env", ".env", "config/private.py", "services/../.env", "C:/secret.py"]
)
def test_paths_outside_packaged_allowlist_are_rejected(
    tmp_path: Path, path: str
) -> None:
    async def run() -> None:
        with source_owner(10), pytest.raises(ValueError):
            await source(tmp_path).read_tool({"path": path})

    asyncio.run(run())


def test_read_search_limits_and_model_registry_auth(tmp_path: Path) -> None:
    async def run() -> None:
        access = source(tmp_path)
        executor = ToolExecutor(default_registry(source_access=access))
        with source_owner(10):
            result = await access.read_tool(
                {"path": "services/sample.py", "start": "2"}
            )
            assert (
                result["source"].startswith("2: # line")
                and len(result["source"].splitlines()) == 120
            )
            found = await access.search_tool({"query": "line"})
            assert len(found["matches"]) == 15 and found["limited"]
            assert "PRIVATE_SECRET" not in str(
                await access.search_tool({"query": "PRIVATE_SECRET"})
            )
        assert '"ok": false' in await executor.execute(
            "read_source", '{"path":"services/sample.py"}'
        )

    asyncio.run(run())


@pytest.mark.parametrize("start", ["0", "-1", "100001", "nan"])
def test_invalid_line_offsets_fail(tmp_path: Path, start: str) -> None:
    async def run() -> None:
        with source_owner(10), pytest.raises(ValueError):
            await source(tmp_path).read_tool(
                {"path": "services/sample.py", "start": start}
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "declaration",
    [
        "API_KEY = 'PRIVATE_SECRET'",
        "PASSWORD: str = 'PRIVATE_SECRET'",
        "self.token = 'PRIVATE_SECRET'",
        "API_KEYS = ['PRIVATE_SECRET']",
        "config = {'api_key': 'PRIVATE_SECRET'}",
    ],
)
def test_secret_literals_are_redacted_without_execution(
    tmp_path: Path, declaration: str
) -> None:
    access = source(tmp_path)
    (tmp_path / "services/sample.py").write_text(
        declaration + "\nraise RuntimeError('never execute')", "utf-8"
    )
    files = access.snapshot_sync()
    assert len(files) == 1 and "PRIVATE_SECRET" not in files[0].text
    assert "never execute" in files[0].text


def test_invalid_python_and_external_symlinks_are_not_read(tmp_path: Path) -> None:
    access = source(tmp_path)
    (tmp_path / "services/sample.py").write_text("this is invalid python !", "utf-8")
    assert not access.snapshot_sync()
    external = tmp_path.parent / (tmp_path.name + "-external.py")
    external.write_text("value='PRIVATE_SECRET'", "utf-8")
    try:
        (tmp_path / "services/link.py").symlink_to(external)
    except OSError:
        return  # Windows может запрещать симлинки без Developer Mode.
    assert not access.snapshot_sync()


@pytest.mark.parametrize(
    "scope", ["private", "group", "other", "no_facts", "restricted"]
)
def test_engine_source_scope_is_bound_to_authenticated_private_actor(
    tmp_path: Path, scope: str
) -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        engine._source_access = source(tmp_path)
        owner = 11 if scope == "other" else 10
        kwargs: dict[str, object] = {}
        if scope == "group":
            kwargs["chat_id"] = -100
        if scope == "no_facts":
            kwargs["use_personal_facts"] = False
        if scope == "restricted":
            engine._user_states.get(owner).age_restricted = True

        async def chat(**args: object) -> str:
            assert current_source_owner.get() == (10 if scope == "private" else None)
            return "Ответ"

        model.chat.side_effect = chat
        await engine.respond(owner, "Проверь код services/sample.py", **kwargs)  # type: ignore[arg-type]
        assert ("read_source" in model.chat.await_args.kwargs["tool_names"]) is (
            scope == "private"
        )
        assert current_source_owner.get() is None

    asyncio.run(run())
