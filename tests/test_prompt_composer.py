"""Тесты тематической сборки системного промпта."""

import pytest

from protogen_delta.services.prompt_composer import PromptComposer, PromptSections


def _composer() -> PromptComposer:
    """Создать компоновщик с различимыми тестовыми секциями."""
    return PromptComposer(
        PromptSections(
            core="CORE",
            lore="LORE",
            body="BODY",
            roleplay="RP",
        )
    )


def test_prompt_composer_requires_core_section() -> None:
    """Компоновщик не должен работать без обязательного ядра личности."""
    with pytest.raises(ValueError, match="не может быть пустой"):
        PromptComposer(PromptSections(core="   "))


def test_prompt_composer_keeps_regular_request_compact() -> None:
    """Обычный вопрос не должен загружать лор, тело и RP."""
    composer = _composer()
    assert (
        composer.compose(
            "Найди последнюю стабильную версию Python",
            is_roleplay=False,
        )
        == "CORE"
    )
    assert composer.compose("Дельта, привет", is_roleplay=False) == "CORE"


def test_prompt_composer_adds_only_relevant_optional_section() -> None:
    """Вопросы о виде и теле должны подключать подходящие секции."""
    composer = _composer()

    assert (
        composer.compose(
            "Расскажи, кто такие протогены",
            is_roleplay=False,
        )
        == "CORE\n\nLORE"
    )
    assert (
        composer.compose(
            "Как выглядит твой хвост?",
            is_roleplay=False,
        )
        == "CORE\n\nBODY"
    )


def test_prompt_composer_does_not_project_body_rules_onto_image() -> None:
    """Картинка с анатомией не должна автоматически получать тело Дельты."""
    assert (
        _composer().compose(
            "Как тебе этот хвост?",
            is_roleplay=False,
            has_images=True,
        )
        == "CORE"
    )


def test_prompt_composer_adds_all_physical_sections_for_roleplay() -> None:
    """Активному RP нужны лор, тело и правила сцены."""
    assert (
        _composer().compose(
            "*подхожу ближе*",
            is_roleplay=True,
        )
        == "CORE\n\nLORE\n\nBODY\n\nRP"
    )


@pytest.mark.parametrize(
    ("message", "tools"),
    [
        ("Привет, как дела?", frozenset()),
        ("Какая погода в Москве?", frozenset({"get_weather"})),
        ("Курс доллара к рублю", frozenset({"get_exchange_rate"})),
        ("Сколько времени в Москве?", frozenset({"get_current_time"})),
        ("Найди последнюю версию Python", frozenset({"web_search", "fetch_web_page"})),
        (
            "А можешь глянуть в интернете в каком году вышел крепкий орешек 2",
            frozenset({"web_search", "fetch_web_page"}),
        ),
        (
            "А сколько цветов энергетиков у марки Burn?",
            frozenset({"web_search", "fetch_web_page"}),
        ),
        ("Прочитай https://example.com/page", frozenset({"fetch_web_page"})),
    ],
)
def test_prompt_composer_selects_only_relevant_tools(
    message: str, tools: frozenset[str]
) -> None:
    assert _composer().select_tools(message) == tools


@pytest.mark.parametrize(
    "message",
    [
        "Так, поиск тебе новый добавили, а ну ка найди мне последнюю инфу "
        "о Call of duty modern warfare 4",
        "Найди последнюю инфу о MW4",
        "Свежая инфа про игру есть?",
        "Актуальную инфу о релизе, пожалуйста",
        "Поищи мне информацию о Call of Duty",
        "Найди сведения о новой игре",
        "В инете посмотри, откуда фраза «Всё идёт по плану»",
    ],
)
def test_information_requests_allow_search_and_reading_sources(message: str) -> None:
    """Разговорная просьба даёт поиск и чтение результатов, без иных полномочий."""
    assert _composer().select_tools(message) == {"web_search", "fetch_web_page"}


@pytest.mark.parametrize(
    "message",
    [
        "*передаю последнюю записку*",
        "Мне прислали инфу о нашей встрече",
        "У меня свежий кофе и новая игра",
    ],
)
def test_ordinary_information_mentions_do_not_enable_search(message: str) -> None:
    assert not _composer().select_tools(message)


@pytest.mark.parametrize(
    "message",
    ["Что означает флоккуляция?", "Как дела?", "Объясни DNS"],
)
def test_optional_search_does_not_require_keyword(message: str) -> None:
    """Обычный ответ может проверить незнакомый факт без отдельной просьбы."""
    assert _composer().select_tools(message, allow_web_fallback=True) == {
        "web_search",
        "fetch_web_page",
    }


@pytest.mark.parametrize(
    "message",
    [
        "Не ищи в интернете, объясни DNS своими словами",
        "Без поиска: что такое DNS?",
        "Не гугли последнюю версию Python",
        "Не используй интернет, расскажи о TCP",
        "Don't search, explain TCP",
    ],
)
def test_user_can_disable_optional_and_keyword_search(message: str) -> None:
    assert "web_search" not in _composer().select_tools(
        message, allow_web_fallback=True
    )


def test_quoted_search_ban_does_not_disable_user_request() -> None:
    assert _composer().select_tools(
        "Проверь в интернете фразу «не ищи»", allow_web_fallback=True
    ) == {"web_search", "fetch_web_page"}


def test_search_ban_does_not_disable_explicit_page_reading() -> None:
    assert _composer().select_tools(
        "Без поиска, прочитай https://example.org/page", allow_web_fallback=True
    ) == {"fetch_web_page"}


def test_optional_search_keeps_weather_clarification() -> None:
    from protogen_delta.core.user_state import ConversationTurn

    history = [ConversationTurn("Погода в Springfield", "Какой вариант? 1 или 2?")]
    assert _composer().select_tools("1", history, allow_web_fallback=True) == {
        "get_weather",
        "web_search",
        "fetch_web_page",
    }


def test_prompt_composer_summarizes_old_history_and_keeps_recent_turns() -> None:
    from protogen_delta.core.user_state import ConversationTurn

    history = tuple(
        ConversationTurn(f"вопрос {index}", f"ответ {index}") for index in range(6)
    )
    selection = _composer().compact_history(history, live_turns=2)

    assert "вопрос 0" in selection.summary
    assert "ответ 3" in selection.summary
    assert [turn.user_message for turn in selection.recent] == ["вопрос 4", "вопрос 5"]

    limited = _composer().compact_history(
        history[-2:],
        live_turns=2,
        history_chars=20,
    )
    assert [turn.user_message for turn in limited.recent] == ["вопрос 5"]

    with pytest.raises(ValueError, match="больше нуля"):
        _composer().compact_history(history, live_turns=0)


def test_tools_remain_available_for_weather_clarification_only() -> None:
    from protogen_delta.core.user_state import ConversationTurn

    history = [
        ConversationTurn(
            "Погода в Springfield", "Какой вариант? 1 — Миссури, 2 — Иллинойс."
        )
    ]
    assert _composer().select_tools("1", history) == {"get_weather"}
    assert _composer().select_tools("Миссури", history) == {"get_weather"}
    assert _composer().select_tools("Спасибо", history) == frozenset()
    assert _composer().select_tools("Как дела?", history) == frozenset()
    assert _composer().select_tools("Объясни DNS", history) == frozenset()
    assert (
        _composer().select_tools("Да, лучше расскажи про DNS", history) == frozenset()
    )
    history.append(
        ConversationTurn(
            "Москва — город, Springfield — Миссури", "Подтверди первый вариант."
        )
    )
    assert _composer().select_tools("Да", history) == {"get_weather"}


def test_temporary_appearance_does_not_load_conflicting_base_body() -> None:
    assert (
        _composer().compose("*подхожу*", is_roleplay=True, has_custom_appearance=True)
        == "CORE\n\nRP"
    )


def test_large_latest_turn_is_bounded_without_changing_saved_history() -> None:
    from protogen_delta.core.user_state import ConversationTurn

    original = ConversationTurn(
        "USER START " + "u" * 9000 + " USER END",
        "ANSWER START " + "a" * 9000 + " ANSWER END",
    )
    selected = _composer().compact_history([original], history_chars=8000)
    assert len(selected.recent) == 1
    shortened = selected.recent[0]
    assert len(shortened.user_message) + len(shortened.assistant_message) == 8000
    assert shortened.user_message.startswith("USER START")
    assert shortened.user_message.endswith("USER END")
    assert shortened.assistant_message.startswith("ANSWER START")
    assert shortened.assistant_message.endswith("ANSWER END")
    assert "сокращено" in shortened.user_message
    assert len(original.user_message) > 9000 and len(original.assistant_message) > 9000


@pytest.mark.parametrize("limit", [1, 2, 20, 8000])
@pytest.mark.parametrize("user,answer", [("u" * 9000, "short"), ("", "a" * 9000)])
def test_history_budget_includes_marker_and_short_field_rebalance(
    limit: int, user: str, answer: str
) -> None:
    from protogen_delta.core.user_state import ConversationTurn

    selected = _composer().compact_history(
        [ConversationTurn(user, answer)], history_chars=limit
    )
    assert (
        sum(
            len(turn.user_message) + len(turn.assistant_message)
            for turn in selected.recent
        )
        <= limit
    )
    assert (
        _composer().compose(
            "Как ты выглядишь?", is_roleplay=False, has_custom_appearance=True
        )
        == "CORE"
    )
