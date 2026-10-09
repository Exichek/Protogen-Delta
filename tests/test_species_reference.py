"""Справочник помогает распознаванию, не подменяя индивидуальный референс."""

import asyncio
from dataclasses import replace

import pytest
from appearance_fixtures import observation, verified
from test_response_engine import _create_engine

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.prompt_composer import PromptComposer, PromptSections


@pytest.mark.parametrize(
    "message,images,roleplay,custom,expected",
    [
        ("Привет", False, False, False, False),
        ("Кто такие сергалы?", False, False, False, True),
        ("Shark or dolphin?", False, False, False, True),
        ("Опиши", True, False, False, True),
        ("Идём", False, True, True, False),
    ],
)
def test_species_section_is_thematic(
    message: str, images: bool, roleplay: bool, custom: bool, expected: bool
) -> None:
    composer = PromptComposer(
        PromptSections(core="CORE", species="SPECIES", body="BODY")
    )
    result = composer.compose(
        message, is_roleplay=roleplay, has_images=images, has_custom_appearance=custom
    )
    assert ("SPECIES" in result) is expected


def test_appearance_extraction_receives_species_reference() -> None:
    engine, _, model, _, _, _ = _create_engine()
    guide = load_prompt("furry_species_reference.txt")
    engine._config = replace(engine._config, species_prompt=guide)
    model.analyze_visual_features.return_value = observation(
        "head_wedge", "fur", "tail_long"
    )
    model.chat.return_value = verified(
        "Голубая шерсть, клиновидная голова.", "sergal", "probable"
    )
    asyncio.run(
        engine.set_delta_appearance_from_image(
            42, ImageInput(b"image", "image/png", "референс")
        )
    )
    sent = model.chat.await_args.kwargs["system_prompt"]
    assert guide not in sent
    assert "Карточки содержат ориентиры" in sent
    assert load_prompt("appearance_identification.txt") in sent
    assert '"sergal"' in model.chat.await_args.kwargs["user_message"]
    assert (
        model.analyze_visual_features.await_args.kwargs["images"]
        == model.chat.await_args.kwargs["images"]
    )
