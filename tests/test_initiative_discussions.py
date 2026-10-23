"""Рабочая политика выбирает обсуждения и новости, без автоматических словесных игр."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from test_initiative_news import NOW, URL, web_and_model
from test_lively_initiative import build

from protogen_delta.core.emotional_memory import EmotionalEvent
from protogen_delta.core.initiative_topics import DEFAULT_TOPICS, DISCUSSION_KINDS
from protogen_delta.core.user_state import ConversationTurn
from protogen_delta.services.initiative_news import InitiativeNewsService, NewsMessage
from protogen_delta.services.initiative_planner import choose_initiative
from protogen_delta.services.proactive import ProactiveConfig


def choose(value: float, **kwargs: Any) -> Any:
    values: dict[str, Any] = dict(
        joy=0.5,
        warmth=0.5,
        playfulness=1.0,
        last_kind="",
        memories=(),
        history=(),
        value=value,
        discussion_topics=DEFAULT_TOPICS,
    )
    values.update(kwargs)
    return choose_initiative(**values)


def test_production_policy_selects_concrete_discussions_and_optional_news() -> None:
    assert {choose(n / 1000).kind for n in range(1000)} == DISCUSSION_KINDS
    assert {
        choose(n / 1000, news_available=True).kind for n in range(1000)
    } == DISCUSSION_KINDS | {"news"}
    assert all(
        choose(n / 1000, last_kind="discussion_games").kind != "discussion_games"
        for n in range(1000)
    )
    assert (
        choose(0.9, discussion_topics=("games",), last_kind="discussion_games").kind
        == "discussion_games"
    )
    with pytest.raises(ValueError):
        choose(0.9, discussion_topics=("unknown",))


def test_closed_technical_topic_is_not_proposed_again() -> None:
    history = (
        ConversationTurn(
            "Закрыли тему кода", "Больше не возвращаемся", technical_topic_closed=True
        ),
    )
    assert {choose(n / 100, history=history).kind for n in range(100)} == {
        "discussion_games"
    }
    assert (
        choose(0.5, history=history, discussion_topics=("ai", "tech")).kind == "silence"
    )


def test_real_news_delivery_preserves_source_and_records_only_after_telegram(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [NOW.timestamp()]
        memories, initiative, _, states, bot, model, service = build(
            tmp_path, now, value=0.99
        )
        web, draft_model = web_and_model()
        model.chat.return_value = draft_model.chat.return_value
        service._config = replace(service._config, discussion_topics=("games",))
        service._news = InitiativeNewsService(web)
        states.get(7).history.append(
            ConversationTurn("PRIVATE_USER_TEXT", "PRIVATE_OLD_REPLY")
        )
        states.get(7).emotions.memory.events.append(
            EmotionalEvent("question", now[0], source="PRIVATE_EMOTIONAL_SOURCE")
        )
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 1
        model.chat.assert_awaited_once()
        assert model.chat.await_args.kwargs["json_response"] is True
        assert "PRIVATE_" not in model.chat.await_args.kwargs["system_prompt"]
        assert "PRIVATE_" not in model.chat.await_args.kwargs["user_message"]
        args = bot.send_message.await_args
        assert "23.10.2026" in args.args[1]
        assert URL in args.args[1]
        assert args.kwargs["parse_mode"] is None
        assert args.kwargs["link_preview_options"].is_disabled
        assert await initiative.seen_news(7, now[0]) == (URL,)
        assert (
            states.get(7).history[-1].initiative_reason
            == "Поделиться проверенной новостью: игры"
        )
        assert states.get(7).history[-1].proactive
        assert await service.run_once() == 0

    asyncio.run(run())


def test_failed_news_attempt_does_not_search_again_and_later_discusses_topic(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        now = [NOW.timestamp()]
        memories, _, _, _, bot, model, service = build(tmp_path, now, value=0.99)
        web, _ = web_and_model()
        web.search.return_value = {"results": []}
        service._config = replace(service._config, discussion_topics=("games",))
        service._news = InitiativeNewsService(web)
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 0
        model.chat.assert_not_awaited()
        bot.send_message.assert_not_awaited()
        now[0] += 3600
        model.chat.return_value = (
            "Хочешь обсудить, что делает кооперативную игру удобной для команды?"
        )
        assert await service.run_once() == 1
        web.search.assert_awaited_once()
        model.chat.assert_awaited_once()
        assert "один конкретный вопрос" in model.chat.await_args.kwargs["user_message"]

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["read", "model"])
def test_new_user_input_prevents_news_send(tmp_path: Path, stage: str) -> None:
    async def run() -> None:
        now = [NOW.timestamp()]
        memories, initiative, _, _, bot, model, service = build(
            tmp_path, now, value=0.99
        )
        web, draft_model = web_and_model()
        service._config = replace(service._config, discussion_topics=("games",))
        service._news = InitiativeNewsService(web)
        model.chat.return_value = draft_model.chat.return_value
        if stage == "read":

            async def read(*_: Any) -> Any:
                await memories.note_user_activity(7, now[0] + 1)
                return web.read.return_value

            web.read.side_effect = read
        else:

            async def chat(**_: Any) -> str:
                await memories.note_user_activity(7, now[0] + 1)
                return str(draft_model.chat.return_value)

            model.chat.side_effect = chat
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 0
        bot.send_message.assert_not_awaited()
        assert await initiative.seen_news(7, now[0]) == ()
        if stage == "read":
            model.chat.assert_not_awaited()

    asyncio.run(run())


def test_active_rp_does_not_search_for_news(tmp_path: Path) -> None:
    async def run() -> None:
        now = [NOW.timestamp()]
        memories, _, _, states, _, model, service = build(tmp_path, now, value=0.99)
        web, _ = web_and_model()
        service._config = replace(service._config, discussion_topics=DEFAULT_TOPICS)
        service._news = InitiativeNewsService(web)
        states.get(7).roleplay_active = True
        states.get(7).history.append(
            ConversationTurn("*Сажусь у костра*", "*Ставлю кружку рядом*")
        )
        model.chat.return_value = "Хочешь продолжить нашу сцену у костра?"
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 1
        assert (
            "Последние реплики текущей личной сцены"
            in model.chat.await_args.kwargs["user_message"]
        )
        web.search.assert_not_awaited()

    asyncio.run(run())


def test_news_is_not_cut_in_the_middle_of_a_link(tmp_path: Path) -> None:
    async def run() -> None:
        now = [NOW.timestamp()]
        memories, _, _, _, bot, _, service = build(tmp_path, now, value=0.99)
        service._config = replace(service._config, discussion_topics=("games",))
        news = AsyncMock(spec=InitiativeNewsService)
        news.enabled = True
        news.generate.return_value = NewsMessage("я" * 701, URL, "games")
        service._news = news
        await memories.note_user_activity(7, now[0] - 10800)
        assert await service.run_once() == 0
        bot.send_message.assert_not_awaited()
        with pytest.raises(ValueError):
            service.__class__(
                bot=bot,
                deepseek=AsyncMock(),
                repository=memories,
                system_prompt="core",
                config=ProactiveConfig(discussion_topics=("unknown",)),
            )

    asyncio.run(run())
