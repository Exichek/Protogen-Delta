"""Общая загрузка правил диалога для рабочего бота и оценки поведения."""

from collections.abc import Callable

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.config.settings import Settings
from protogen_delta.core.initiative_limits import (
    DAILY_LIMIT,
    MIN_INTERVAL_SECONDS,
    QUIET_END_HOUR,
    QUIET_START_HOUR,
)
from protogen_delta.services.capabilities import build_capabilities
from protogen_delta.services.fetishes import FetishTriggers
from protogen_delta.services.prompt_composer import ExampleGuidance
from protogen_delta.services.response_engine import ResponseEngineConfig


def load_response_configuration(
    settings: Settings,
    *,
    fetish_triggers: FetishTriggers,
    fetish_names: dict[str, str],
    loader: Callable[[str], str] = load_prompt,
) -> ResponseEngineConfig:
    """Загрузить одинаковые секции и реальные настройки сроков инициативы."""
    core = loader("personality/core.txt")
    schedule = (
        "Включён режим живой инициативы. После выбранного срока ещё учитываются "
        "накопленное побуждение, эмоции и подходящий повод; можно ничего не отправлять. "
        f"Между самостоятельными отправлениями минимум {MIN_INTERVAL_SECONDS / 3600:g} часа; "
        f"максимум {DAILY_LIMIT} инициативы за московские сутки. "
        "После инициативы без ответа повторов нет до новой активности пользователя. "
        "Ответ пользователя снова разрешает инициативу после новой паузы, "
        "но не сбрасывает минимальный интервал и суточный предел."
        if settings.proactive_lively_enabled
        else (
            "Включён прежний режим инициативы. Если пользователь не ответил на первое приглашение, "
            f"через {settings.proactive_cooldown_seconds / 3600:g} часов допускается "
            "одно последнее повторное сообщение; затем нужно ждать нового сообщения пользователя."
        )
    )
    core += "\n\n" + loader("initiative_system").format(
        minimum_hours=settings.proactive_idle_seconds / 3600,
        maximum_hours=settings.proactive_idle_max_seconds / 3600,
        schedule=schedule,
        quiet_start=f"{QUIET_START_HOUR:02}:00",
        quiet_end=f"{QUIET_END_HOUR:02}:00",
    )
    core += "\n\n" + loader("emotional_system")
    capabilities = build_capabilities(settings)
    return ResponseEngineConfig(
        fetish_triggers=fetish_triggers,
        fetish_names=fetish_names,
        system_prompt=core,
        protogen_lore_prompt=loader("personality/protogen_lore.txt"),
        body_prompt=loader("personality/body.txt"),
        rp_prompt=loader("personality/rp.txt"),
        species_prompt=loader("furry_species_reference"),
        adult_conversation_prompt=loader("adult_conversation_style"),
        female_body_prompt=loader("personality/body_female"),
        adult_body_male_prompt=loader("personality/adult_body_male"),
        adult_body_female_prompt=loader("personality/adult_body_female"),
        adult_rp_prompt=loader("personality/adult_rp"),
        technical_prompt=loader("conversation_technical"),
        visual_prompt=loader("conversation_visual"),
        voice_prompt=loader("voice_examples"),
        scene_voice_prompt=loader("scene_examples"),
        capabilities_context=capabilities.summary,
        implementation_context=capabilities.implementation,
        capabilities_overview_context=capabilities.overview,
    )


def load_example_guidance(
    *,
    loader: Callable[[str], str] = load_prompt,
) -> ExampleGuidance:
    """Загрузить отдельный кандидат только по явному выбору оценочного стенда."""
    return ExampleGuidance(
        voice=loader("voice_guidance"), scene=loader("scene_guidance")
    )
