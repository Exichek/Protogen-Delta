"""Заметки текущей сборки доступны своему создателю и отмечаются после доставки."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_response_engine import TEST_USER_ID, _create_engine

from protogen_delta.core.runtime_health import RuntimeHealth
from protogen_delta.services.release_journal import ReleaseJournal, is_release_question


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


@pytest.mark.parametrize(
    "question",
    [
        "Йоу, что логи показывают? видишь в себе какие-то изменения щас?)",
        "Какие обновы?",
        "Расскажи патчноут",
        "Что нового?",
        "Когда добавили ZIP? Что обновилось?",
    ],
)
def test_questions_about_updates_receive_patch_history(
    tmp_path: Path, question: str
) -> None:
    journal = ReleaseJournal(tmp_path, frozenset({7}))
    assert is_release_question(question)
    context = asyncio.run(journal.context(7, question=question))
    assert "Это твои патчноуты" in context and "История известных патчноутов" in context
    assert "не заканчивай ответ только фразой" in context
    assert asyncio.run(journal.context(8, question=question)) == ""
    assert "История известных патчноутов" not in asyncio.run(journal.context(7))


def test_quoted_updates_do_not_switch_roleplay_or_expose_history() -> None:
    assert not is_release_question("Он написал: «Какие обновы?»")
    assert not is_release_question("```patch notes```")


def test_notes_from_previous_start_are_kept_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Замена файла текущего релиза не стирает сохранённый предыдущий патчноут."""
    import protogen_delta.services.release_journal as module

    original = module.load_json
    old = {
        "release_id": "test-old",
        "date": "2026-10-13",
        "changes": ["Уникальное изменение persistence-marker"],
    }
    new = {"release_id": "test-new", "date": "2026-10-14", "changes": ["Новая функция"]}
    monkeypatch.setattr(
        module,
        "load_json",
        lambda name: old if name == "release_journal.json" else {"releases": []},
    )
    ReleaseJournal(tmp_path, frozenset({7}))
    monkeypatch.setattr(
        module,
        "load_json",
        lambda name: new if name == "release_journal.json" else {"releases": []},
    )
    journal = ReleaseJournal(tmp_path, frozenset({7}))
    text = asyncio.run(
        journal.context(7, question="Что обновилось в persistence-marker?")
    )
    assert "test-old" in text and "test-new" in text and "persistence-marker" in text
    monkeypatch.setattr(module, "load_json", original)


def test_update_question_uses_facts_instead_of_rp_prompt(tmp_path: Path) -> None:
    async def run() -> None:
        engine, _, model, *_ = _create_engine()
        engine._release_journal = ReleaseJournal(tmp_path, frozenset({TEST_USER_ID}))
        engine._user_states.get(TEST_USER_ID).roleplay_active = True
        await engine.respond(TEST_USER_ID, "Что нового в твоих обновлениях?")
        prompt = model.chat.await_args.kwargs["system_prompt"]
        assert (
            "Это твои патчноуты" in prompt and "История известных патчноутов" in prompt
        )
        assert "RP PROMPT" not in prompt
        assert engine._user_states.get(TEST_USER_ID).roleplay_active

    asyncio.run(run())


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
