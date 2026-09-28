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
        ("Найди последнюю версию Python", frozenset({"web_search"})),
        ("Прочитай https://example.com/page", frozenset({"fetch_web_page"})),
    ],
)
def test_prompt_composer_selects_only_relevant_tools(
    message: str, tools: frozenset[str]
) -> None:
    assert _composer().select_tools(message) == tools


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
