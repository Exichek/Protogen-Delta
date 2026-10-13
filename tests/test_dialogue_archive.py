"""Архив переживает перезапуск, находится поиском и не смешивает аккаунты."""

import asyncio
import json
import sqlite3
import sys
import threading
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import Response, SendMessage
from aiogram.types import Chat, Message, PhotoSize, TelegramObject, User

from protogen_delta import dialogue_archive_cli
from protogen_delta.config.settings import _parse_archive_ids
from protogen_delta.core.dialogue_import import telegram_text_messages
from protogen_delta.core.input_operations import (
    InputOperations,
    ReceiveInputsMiddleware,
)
from protogen_delta.handlers.dialogue_archive import (
    ArchiveIncomingMiddleware,
    ArchiveOutgoingMiddleware,
    archive_edit,
    archive_message,
)
from protogen_delta.repositories.dialogue_archive import (
    DialogueArchiveRepository,
    DialogueMessage,
)
from protogen_delta.services.dialogue_archive import (
    DialogueArchiveService,
    archive_generation,
    current_archive_message,
    current_dialogue_owner,
    dialogue_owner,
)
from protogen_delta.services.tools import ToolExecutor, default_registry

_BOT = cast(Bot, object())


def repository(tmp_path: Path) -> DialogueArchiveRepository:
    return DialogueArchiveRepository(tmp_path)


def message(
    key: str = "old",
    text: str = "Проект драконов называется ORBIT_ARCHIVE_913",
    *,
    role: Literal["user", "assistant"] = "user",
    at: float = 1000,
) -> DialogueMessage:
    return DialogueMessage(key, role, text, at)


def telegram(
    owner: int = 1,
    *,
    mid: int = 1,
    text: str = "Текст",
    group: bool = False,
    bot: bool = False,
) -> Message:
    return Message(
        message_id=mid,
        date=datetime.now(timezone.utc),
        chat=Chat(id=-10 if group else owner, type="group" if group else "private"),
        from_user=User(id=owner, is_bot=bot, first_name="Тест"),
        text=text,
    )


def test_full_archive_survives_retention_restart_and_backup_restore(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        await repo.save(1, message())
        await repo.save(
            1, message("reply", "Договорились про проект", role="assistant", at=1001)
        )
        for index in range(80):
            await repo.save(
                1,
                message(str(index), f"Обсуждали погоду номер {index}", at=2000 + index),
            )
        await repo.save(
            2, message("secret", "ORBIT_ARCHIVE_913 чужой пароль", at=1000.5)
        )
        reopened = repository(tmp_path)
        hits = await reopened.search(1, "драконами ORBIT_ARCHIVE_913")
        assert [item.source_key for item in hits] == ["old", "reply"]
        snapshot = tmp_path / "backup" / "dialogue.jsonl"
        assert await reopened.export(1, snapshot) == 82
        assert "чужой пароль" not in snapshot.read_text("utf-8")
        restored = repository(tmp_path / "restore")
        assert await restored.import_snapshot(1, snapshot) == 82
        assert await restored.import_snapshot(1, snapshot) == 82
        assert await restored.export(1, tmp_path / "again.jsonl") == 82
        assert await restored.search(1, "драконы") == hits
        assert not await restored.search(2, "ORBIT_ARCHIVE_913")

    asyncio.run(run())


def test_edits_deduplicate_and_update_index_while_preserving_timestamp(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        await repo.save(1, message("telegram:1", "Старая лисица"))
        await repo.save(1, message("telegram:1", "Новая акула", at=2000))
        assert not await repo.search(1, "лисица")
        hits = await repo.search(1, "акулы")
        assert len(hits) == 1 and hits[0].created_at == 1000
        await repo.export(1, tmp_path / "copy.jsonl")
        assert len((tmp_path / "copy.jsonl").read_text("utf-8").splitlines()) == 2
        with closing(sqlite3.connect(tmp_path / "dialogue_archive.db")) as db:
            db.execute(
                "INSERT INTO dialogue_fts(dialogue_fts,rank) VALUES('integrity-check',1)"
            )
            db.commit()

    asyncio.run(run())


def test_index_recovers_when_missing_and_searches_long_message_tail(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        text = "обычный текст " * 1800 + "PHOENIX_END_915"
        await repo.save(1, message(text=text))
        with closing(sqlite3.connect(tmp_path / "dialogue_archive.db")) as db, db:
            db.execute("DROP TABLE dialogue_fts")
        repo = repository(tmp_path)
        assert (await repo.search(1, "PHOENIX_END_915"))[0].text == text
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        assert "PHOENIX_END_915" in " ".join(
            await service.context(1, "PHOENIX_END_915")
        )
        with dialogue_owner(1):
            result = await service.search_tool({"query": "PHOENIX_END_915"})
        assert result["messages"][0]["truncated"]
        assert "PHOENIX_END_915" in result["messages"][0]["text"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "query",
    ["", "я ты мы", '" OR 1=1; DROP TABLE dialogue_messages; --', "🙂", "* NEAR()"],
)
def test_empty_and_sql_like_queries_are_safe(tmp_path: Path, query: str) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        await repo.save(1, message())
        assert await repo.search(1, query) == []
        assert await repo.search(1, "драконами", limit=0) == []
        assert await repo.search(1, "ORBIT_ARCHIVE_913")

    asyncio.run(run())


@pytest.mark.parametrize("owner", [True, 0, -1, 2**63, "1"])
def test_archive_rejects_invalid_owner(tmp_path: Path, owner: int) -> None:
    with pytest.raises(ValueError):
        asyncio.run(repository(tmp_path).save(owner, message()))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_key", ""),
        ("source_key", "x" * 201),
        ("role", "system"),
        ("text", ""),
        ("text", "x" * 65537),
        ("created_at", float("nan")),
        ("created_at", float("inf")),
        ("created_at", -1),
        ("created_at", True),
        ("kind", "x" * 33),
    ],
    ids=[str(i) for i in range(10)],
)
def test_archive_validates_imported_records(
    tmp_path: Path, field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        invalid = replace(message(), **{field: cast(Any, value)})
        asyncio.run(repository(tmp_path).save(1, invalid))


@pytest.mark.parametrize("broken", ["foreign", "bad_record", "truncated", "bool_owner"])
def test_restore_is_atomic_and_rejects_foreign_data(
    tmp_path: Path, broken: str
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        snapshot = tmp_path / "bad.jsonl"
        owner = 2 if broken == "foreign" else True if broken == "bool_owner" else 1
        rows = [
            json.dumps({"schema": 1, "user_id": owner}),
            json.dumps(
                {
                    "source_key": "new",
                    "role": "user",
                    "text": "NEW_ARCHIVE_822",
                    "created_at": 0,
                    "kind": "text",
                }
            ),
        ]
        rows.append(
            "{"
            if broken == "truncated"
            else json.dumps(
                {
                    "source_key": "invalid",
                    "role": "system",
                    "text": "x",
                    "created_at": 0,
                    "kind": "text",
                }
            )
        )
        snapshot.write_text("\n".join(rows), encoding="utf-8")
        with pytest.raises(ValueError):
            await repo.import_snapshot(1, snapshot)
        assert not await repo.search(1, "NEW_ARCHIVE_822")

    asyncio.run(run())


def test_forgetting_reset_and_snapshots_do_not_revive_removed_data(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1, 2}), tmp_path)
        await service.record(1, message(text="Друга зовут Илья"), 0)
        await service.record(2, message(text="Друга зовут Илья"), 0)
        await service.backup()
        assert list((tmp_path / "backups/dialogues").glob("1-*.jsonl"))
        await service.forget_sources(1, ("Ильёй",))
        assert not await repo.search(1, "Илья")
        assert await repo.search(2, "Илья")
        assert not list((tmp_path / "backups/dialogues").glob("1-*.jsonl"))
        await service.record(1, message(), 0)  # Старая операция не переживает удаление.
        assert not await repo.search(1, "ORBIT_ARCHIVE_913")
        await service.record(1, message(), 1)
        await service.backup()
        await service.delete_user(1)
        await service.record(1, message("stale"), 1)
        assert not await repo.search(1, "ORBIT_ARCHIVE_913")
        assert not list((tmp_path / "backups/dialogues").glob("1-*.jsonl"))
        await service.delete_user(3)
        await service.forget_sources(3, ("x",))
        await service.forget_sources(1, ())

    asyncio.run(run())


def test_model_tool_owner_is_trusted_and_parallel_scopes_are_isolated(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1, 2}), tmp_path)
        await repo.save(1, message(text="Проект OWNER_ONE_829"))
        await repo.save(2, message(text="Проект OWNER_TWO_829"))
        executor = ToolExecutor(default_registry(dialogue_archive=service))

        async def search(owner: int) -> str:
            with dialogue_owner(owner):
                await asyncio.sleep(0)
                return await executor.execute("search_dialogue", '{"query":"Проект"}')

        one, two = await asyncio.gather(search(1), search(2))
        assert "OWNER_ONE_829" in one and "OWNER_TWO_829" not in one
        assert "OWNER_TWO_829" in two and "OWNER_ONE_829" not in two
        assert current_dialogue_owner.get() is None
        assert not (await service.search_tool({"query": "Проект"}))["available"]
        with dialogue_owner(3):
            assert not (await service.search_tool({"query": "Проект"}))["available"]
        invalid = await executor.execute(
            "search_dialogue", '{"query":"Проект","user_id":"2"}'
        )
        assert not json.loads(invalid)["ok"]
        assert await service.context(3, "Проект") == []
        assert await service.context(1, "NO_MATCH_892") == []

    asyncio.run(run())


def test_search_excludes_current_question_and_distant_neighbors(tmp_path: Path) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        await repo.save(1, message(text="Проект ORBIT_ARCHIVE_913"))
        await repo.save(1, message("next", "Сосед через месяц", at=100000))
        await repo.save(1, message("telegram:99", "ORBIT_ARCHIVE_913?", at=200000))
        token = current_archive_message.set("telegram:99")
        try:
            lines = await service.context(1, "ORBIT_ARCHIVE_913")
            assert "Сосед через месяц" not in str(lines)
            assert "913?" not in str(lines)
        finally:
            current_archive_message.reset(token)

    asyncio.run(run())


def test_hourly_snapshots_rotate_and_worker_survives_backup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        for day in range(10):
            monkeypatch.setattr(
                "protogen_delta.services.dialogue_archive.time",
                lambda day=day: 86400 * day,
            )
            await service.backup()
        snapshots = sorted((tmp_path / "backups/dialogues").glob("*.jsonl"))
        assert len(snapshots) == 7 and snapshots[0].name == "1-19700104.jsonl"
        original = service.backup
        started = asyncio.Event()

        async def broken() -> None:
            started.set()
            raise OSError("private text must not leak")

        service.backup = broken  # type: ignore[method-assign]
        service.start()
        service.start()
        await started.wait()
        await service.close()
        service.backup = original  # type: ignore[method-assign]
        service.start()
        await asyncio.sleep(0.01)
        await service.close()
        disabled = DialogueArchiveService(repo, frozenset(), tmp_path)
        disabled.start()
        await disabled.close()

    asyncio.run(run())


def test_cancelled_sqlite_write_finishes_before_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        started = threading.Event()
        release = threading.Event()
        original = repo._save_sync

        def slow(owner: int, item: DialogueMessage) -> None:
            started.set()
            assert release.wait(3)
            original(owner, item)

        monkeypatch.setattr(repo, "_save_sync", slow)
        write = asyncio.create_task(service.record(1, message(), 0))
        assert await asyncio.to_thread(started.wait, 3)
        write.cancel()
        reset = asyncio.create_task(service.delete_user(1))
        await asyncio.sleep(0)
        assert not reset.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await write
        await reset
        assert not await repo.search(1, "ORBIT_ARCHIVE_913")

    asyncio.run(run())


@pytest.mark.parametrize(
    "group,bot,owner",
    [(False, False, 1), (True, False, 1), (False, True, 1), (False, False, 2)],
)
def test_incoming_archive_is_private_and_resets_contextvars(
    tmp_path: Path, group: bool, bot: bool, owner: int
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        operations = InputOperations()
        middleware = ArchiveIncomingMiddleware(service, operations)
        handler = AsyncMock(return_value="ok")
        result = await middleware(
            handler, telegram(owner, text="ORBIT_ARCHIVE_913", group=group, bot=bot), {}
        )
        assert (
            result == "ok"
            and archive_generation.get() is None
            and current_archive_message.get() is None
        )
        assert bool(await repo.search(owner, "ORBIT_ARCHIVE_913")) is (
            not group and not bot and owner == 1
        )
        await middleware(handler, User(id=1, is_bot=False, first_name="Тест"), {})

    asyncio.run(run())


def test_incoming_stale_ticket_and_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        operations = InputOperations()
        receive = ReceiveInputsMiddleware(operations)
        archive = ArchiveIncomingMiddleware(service, operations)
        handler = AsyncMock()

        async def stale(event: TelegramObject, data: dict[str, Any]) -> None:
            async with operations.resetting(1):
                pass
            await archive(handler, event, data)

        await receive(stale, telegram(), {})
        handler.assert_not_awaited()
        monkeypatch.setattr(service, "record", AsyncMock(side_effect=OSError("secret")))
        await archive(handler, telegram(), {})
        handler.assert_awaited_once()

    asyncio.run(run())


def test_media_archive_has_caption_without_private_download_url() -> None:
    image = telegram().model_copy(
        update={
            "text": None,
            "caption": "Надпись на картинке",
            "photo": [
                PhotoSize(
                    file_id="SECRET_FILE_ID",
                    file_unique_id="unique",
                    width=20,
                    height=20,
                )
            ],
        }
    )
    item = archive_message(image, incoming=True)
    assert (
        item.kind == "photo"
        and "Надпись" in item.text
        and "SECRET_FILE_ID" not in item.text
    )
    assert (
        "Вложение"
        in archive_message(
            image.model_copy(update={"caption": None}), incoming=False
        ).text
    )


@pytest.mark.parametrize(
    "wrapped,album", [(False, False), (True, False), (False, True)]
)
def test_only_confirmed_outgoing_delivery_is_archived(
    tmp_path: Path, wrapped: bool, album: bool
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        middleware = ArchiveOutgoingMiddleware(service)
        reply = telegram(mid=20, text="Подтверждённый ORBIT_ARCHIVE_913", bot=True)
        value = (
            [reply, telegram(mid=21, text="Часть два", bot=True)] if album else reply
        )
        response = Response(ok=True, result=value) if wrapped else value
        request = AsyncMock(return_value=response)
        method = SendMessage(chat_id=1, text="ORBIT_ARCHIVE_913")
        assert await middleware(request, _BOT, method) == response
        assert await repo.search(1, "ORBIT_ARCHIVE_913")
        assert request.await_count == 1
        request.side_effect = RuntimeError("Telegram failed")
        with pytest.raises(RuntimeError):
            await middleware(request, _BOT, method)
        assert await repo.export(1, tmp_path / "out.jsonl") == (2 if album else 1)

    asyncio.run(run())


def test_outgoing_failure_does_not_repeat_delivery_and_stale_send_cannot_revive_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        service = DialogueArchiveService(repo, frozenset({1}), tmp_path)
        middleware = ArchiveOutgoingMiddleware(service)
        reply = telegram(mid=20, text="ORBIT_ARCHIVE_913", bot=True)
        method = SendMessage(chat_id=1, text="reply")

        async def stale(*args: object) -> Message:
            await service.delete_user(1)
            return reply

        await middleware(AsyncMock(side_effect=stale), _BOT, method)
        assert not await repo.search(1, "ORBIT_ARCHIVE_913")
        monkeypatch.setattr(service, "record", AsyncMock(side_effect=OSError("secret")))
        request = AsyncMock(return_value=reply)
        assert await middleware(request, _BOT, method) == reply
        request.assert_awaited_once()
        for target, result in [(2, reply), (1, True), (1, telegram(group=True))]:
            await middleware(
                AsyncMock(return_value=result),
                _BOT,
                SendMessage(chat_id=target, text="reply"),
            )

    asyncio.run(run())


@pytest.mark.parametrize("value", ["0", "-1", "no", str(2**63)])
def test_archive_configuration_rejects_invalid_accounts(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_archive_ids(value)


def test_archive_configuration_and_cli_snapshot_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _parse_archive_ids("") == frozenset()
    assert _parse_archive_ids("1,2,1") == frozenset({1, 2})
    source = tmp_path / "dialogue.txt"
    source.write_text(
        "[13.10.2026 18:08] Exi: CLI_ARCHIVE_915\n[13.10.2026 18:09] Delta: Ответ",
        encoding="utf-8",
    )
    data = tmp_path / "data"
    snapshot = tmp_path / "snapshot.jsonl"
    for action, path in [
        ("import-text", source),
        ("export", snapshot),
        ("restore", snapshot),
    ]:
        folder = tmp_path / "restored" if action == "restore" else data
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "archive",
                action,
                "--data-dir",
                str(folder),
                "--user-id",
                "1",
                "--file",
                str(path),
                "--user-name",
                "Exi",
                "--bot-name",
                "Delta",
            ],
        )
        asyncio.run(dialogue_archive_cli.run())
    assert asyncio.run(repository(tmp_path / "restored").search(1, "CLI_ARCHIVE_915"))
    asyncio.run(archive_edit(telegram()))


def test_export_failure_preserves_old_snapshot_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        await repo.save(1, message())
        target = tmp_path / "archive.jsonl"
        await repo.export(1, target)
        old = target.read_bytes()
        monkeypatch.setattr(
            "protogen_delta.repositories.dialogue_archive.os.replace",
            lambda *_: (_ for _ in ()).throw(OSError("failed")),
        )
        with pytest.raises(OSError):
            await repo.export(1, target)
        assert target.read_bytes() == old
        assert not target.with_suffix(".jsonl.tmp").exists()

    asyncio.run(run())


def test_bulk_import_validation_is_atomic_and_word_forgetting_scans_all_rows(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        repo = repository(tmp_path)
        with pytest.raises(ValueError):
            await repo.save_many(1, [message(), message("bad", "")])
        assert not await repo.search(1, "ORBIT_ARCHIVE_913")
        assert await repo.save_many(1, [message(str(i)) for i in range(20)]) == 20
        await repo.forget_sources(1, ("ORBIT_ARCHIVE_913",))
        assert not await repo.search(1, "ORBIT_ARCHIVE_913")
        await repo.forget_sources(1, ())

    asyncio.run(run())


def test_import_size_and_entry_count_are_bounded(tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    with source.open("wb") as file:
        file.truncate(20 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        telegram_text_messages(source, "Exi", "Delta")
    source.write_text("[13.10.2026 18:08] Exi: hello\n" * 10001, encoding="utf-8")
    with pytest.raises(ValueError):
        telegram_text_messages(source, "Exi", "Delta")


def test_text_import_preserves_speakers_dates_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "dialogue.txt"
    source.write_text(
        "[13.10.2026 18:08] Exi: Наш проект ORBIT_IMPORT_1\n"
        "[13.10.2026 18:08] Delta: Ответ\nВторая строка\n"
        "[13.10.2026 18:09] Exi: \n",
        encoding="utf-8-sig",
    )
    messages = telegram_text_messages(source, "Exi", "Delta")
    assert [item.role for item in messages] == ["user", "assistant"]
    assert (
        messages[0].created_at
        == datetime(2026, 10, 13, 15, 8, tzinfo=timezone.utc).timestamp()
    )
    assert messages[1].text == "Ответ\nВторая строка"
    assert telegram_text_messages(source, "Exi", "Delta") == messages

    async def run() -> None:
        repo = repository(tmp_path)
        for item in messages + messages:
            await repo.save(1, item)
        assert await repo.export(1, tmp_path / "import.jsonl") == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "contents,user,bot",
    [
        ("bad", "Exi", "Delta"),
        ("[13.10.2026 18:08] Stranger: hi", "Exi", "Delta"),
        ("x", "", "Delta"),
        ("x", "Delta", "Delta"),
        ("[31.02.2026 18:08] Exi: hi", "Exi", "Delta"),
    ],
)
def test_text_import_rejects_unknown_participants_or_bad_dates(
    tmp_path: Path, contents: str, user: str, bot: str
) -> None:
    source = tmp_path / "invalid.txt"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        telegram_text_messages(source, user, bot)
