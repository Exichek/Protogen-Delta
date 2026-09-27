"""Тесты тематической сборки системного промпта."""

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
