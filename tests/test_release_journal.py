"""Заметки текущей сборки доступны своему создателю и отмечаются после доставки."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.core.runtime_health import RuntimeHealth
from protogen_delta.services.release_journal import ReleaseJournal


def test_journal_scope_and_ack_survive_restart(tmp_path: Path) -> None:
    async def run() -> None:
        journal = ReleaseJournal(tmp_path, frozenset({7, 8}))
        assert await journal.context(9) == ""
        assert "release_id" in await journal.context(7, initiative=True)
        await journal.acknowledge(9)
        assert not journal._seen(9)
        await journal.acknowledge(7)
        restored = ReleaseJournal(tmp_path, frozenset({7, 8}))
        assert await restored.context(7, initiative=True) == ""
        assert "release_id" in await restored.context(8, initiative=True)
        assert "release_id" in await restored.context(7)

    asyncio.run(run())


def test_diagnostics_are_current_counters_without_raw_logs(tmp_path: Path) -> None:
    health = RuntimeHealth(clock=lambda: 100)
    health.polling_succeeded()
    health.polling_failed()
    journal = ReleaseJournal(tmp_path, frozenset({7}), health=health)
    text = asyncio.run(journal.context(7))
    assert '"polling": true' in text and '"polling_errors": 1' in text
    assert "не устанавливает причину" in text
    health.stop()
    assert '"polling": false' in asyncio.run(journal.context(7))


@pytest.mark.parametrize("scope", ["own", "group", "other", "delivery_error"])
def test_engine_journal_is_scoped_and_not_acknowledged_when_delivery_fails(
    tmp_path: Path, scope: str
) -> None:
    async def run() -> None:
        engine, _, model, _, _, _ = _create_engine()
        journal = ReleaseJournal(tmp_path, frozenset({TEST_USER_ID}))
        engine._release_journal = journal
        owner = TEST_USER_ID + 1 if scope == "other" else TEST_USER_ID
        if scope == "delivery_error":
            with pytest.raises(RuntimeError):
                await engine.respond(
                    owner,
                    "Что обновилось?",
                    deliver=AsyncMock(side_effect=RuntimeError("delivery")),
                )
        else:
            await engine.respond(
                owner, "Что обновилось?", chat_id=-1 if scope == "group" else None
            )
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert ("Журнал установленной сборки" in prompt) is (
            scope in {"own", "delivery_error"}
        )
        assert journal._seen(TEST_USER_ID) is (scope == "own")

    asyncio.run(run())
