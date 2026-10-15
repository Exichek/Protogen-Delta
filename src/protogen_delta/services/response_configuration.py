"""Общая загрузка правил диалога для рабочего бота и оценки поведения."""

from collections.abc import Callable

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.config.settings import Settings
from protogen_delta.services.capabilities import build_capabilities
from protogen_delta.services.fetishes import FetishTriggers
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
    core += "\n\n" + loader("initiative_system").format(
        minimum_hours=settings.proactive_idle_seconds / 3600,
        maximum_hours=settings.proactive_idle_max_seconds / 3600,
        followup_hours=settings.proactive_cooldown_seconds / 3600,
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
