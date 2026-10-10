"""Два визуальных прохода, ограниченный каталог и проверенное происхождение вида."""

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from protogen_delta.config.prompt_loader import load_prompt
from protogen_delta.core.appearance_species import AppearanceSpecies
from protogen_delta.services.appearance_description import (
    APPEARANCE_LIMIT,
    validate_description,
)
from protogen_delta.services.deepseek import ImageInput
from protogen_delta.services.species_catalog import SpeciesCatalog, species_catalog
from protogen_delta.services.visual_model import VisualModel


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Some JSON-mode replies repeat a selection flag with the same
            # boolean value. This carries no conflicting selection; all other
            # duplicates, including false/true or bool/int, remain invalid.
            if (
                key in {"readable", "ambiguous"}
                and type(value) is bool
                and type(result[key]) is bool
                and result[key] is value
            ):
                continue
            raise ValueError("Duplicate visual JSON field")
        result[key] = value
    return result


def _json(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError("Invalid visual response")
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    item = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(item, dict):
        raise ValueError("Expected visual JSON object")
    return item


def validate_species_hint(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Название вида должно быть текстом.")
    value = value.strip()
    if value and not re.fullmatch(r"[\w\-' А-Яа-яЁё]{1,80}", value):
        raise ValueError("Название вида: до 80 букв, цифр, пробелов или дефисов.")
    return value


def validate_reference_notes(value: str) -> str:
    if not isinstance(value, str) or len(value) > 400:
        raise ValueError("Комментарий к референсу: до 400 символов.")
    return value.strip()


def declared_species(text: str) -> str:
    """Только собственная явная декларация, не цитата или ласковое обращение."""
    match = re.search(
        r"(?:^|\n)\s*(?:вид(?: персонажа)?\s*[:—-]|моя фурсона\s*[—:-])"
        r"\s*([\w\-' А-Яа-яЁё]{1,80})[.!]?\s*(?:$|\n)",
        text,
        re.I,
    )
    if match:
        return validate_species_hint(match[1])
    # «Это твой образ для RP» assigns an appearance, not a species name.
    match = re.fullmatch(r"\s*это\s+([\w\-' ]{1,80})[.!]?\s*", text, re.I)
    if match and species_catalog().find(match[1]):
        return validate_species_hint(match[1])
    return ""


@dataclass(frozen=True, slots=True)
class AppearanceResult:
    description: str
    species: AppearanceSpecies
    minor_reference: bool = False


def _region(trait: str) -> str:
    if trait.startswith("head_") or trait in {
        "beak",
        "jagged_jaw",
        "whiskers",
        "rodent_incisors",
        "visor",
        "no_visible_pupils",
        "face_mask",
        "horns",
        "antlers",
    }:
        return "head"
    if "ears" in trait or trait == "ear_tufts":
        return "ears"
    if trait.startswith("tail_"):
        return "tail"
    if trait in {
        "fur",
        "scales",
        "feathers",
        "smooth_skin",
        "striped_fur",
        "spotted_fur",
    }:
        return "coat"
    return "body"


class AppearanceAnalyzer:
    def __init__(
        self, model: VisualModel, catalog: SpeciesCatalog | None = None
    ) -> None:
        self._model = model
        self._catalog = catalog or species_catalog()

    async def analyze(
        self,
        images: Sequence[ImageInput],
        user_message: str,
        content_rules: str,
        *,
        species_hint: str = "",
        subject: Literal["delta", "user"] = "delta",
    ) -> AppearanceResult:
        if not images:
            raise ValueError("No appearance image")
        hint = validate_species_hint(species_hint) or declared_species(user_message)
        # Shared budget for both SDK calls. Cancellation preserves the old card.
        async with asyncio.timeout(120):
            raw = await self._model.analyze_visual_features(
                system_prompt=load_prompt("appearance_observation")
                + "\n\nОграничения описания:\n"
                + content_rules
                + "\n\nСловарь признаков (данные):\n"
                + self._catalog.observation_vocabulary(),
                user_message="Наблюдай внешность выбранного персонажа. Подпись (данные): "
                + repr(user_message)
                + (
                    "\nВид, указанный пользователем (данные): "
                    + repr(hint)
                    + ". Используй его как исходный ориентир для поиска видимых "
                    "признаков и точек крепления. Скрытые типичные детали "
                    "этого вида остаются неизвестными."
                    if hint
                    else ""
                )
                + "\nВерни JSON с обязательными readable, ambiguous, layout, observations "
                "и features. features — массив признаков по переданному словарю, "
                "не пропускай это поле. Не возвращай только текст наблюдений.",
                images=images,
            )
            observed = _json(raw)
            if (
                type(observed.get("readable")) is not bool
                or type(observed.get("ambiguous")) is not bool
            ):
                raise ValueError("Missing visual selection")
            if not observed["readable"] or observed["ambiguous"]:
                raise ValueError("Unusable reference")
            layout = observed.get("layout")
            if (
                not isinstance(layout, dict)
                or set(layout)
                != {"orientation", "head", "torso", "pelvis", "tail_base"}
                or any(
                    not isinstance(value, str) or not 1 <= len(value.strip()) <= 120
                    for value in layout.values()
                )
            ):
                raise ValueError("Invalid reference layout")
            observations = observed.get("observations")
            features = observed.get("features")
            if (
                not isinstance(observations, str)
                or not 1 <= len(observations) <= 1500
                or not isinstance(features, list)
                or len(features) > 30
            ):
                raise ValueError("Invalid visual observations")
            states: dict[str, str] = {}
            for feature in features:
                if (
                    not isinstance(feature, dict)
                    or not isinstance(feature.get("trait"), str)
                    or not isinstance(feature.get("state"), str)
                    or feature.get("trait") not in self._catalog.traits
                    or feature.get("state")
                    not in {"present", "absent", "unobservable", "uncertain"}
                    or not isinstance(feature.get("evidence"), str)
                    or len(feature["evidence"]) > 160
                    or feature["trait"] in states
                ):
                    raise ValueError("Invalid visual feature")
                states[feature["trait"]] = feature["state"]
            cards = self._catalog.select(states, hint)
            extraction = load_prompt("appearance_extraction").format(
                limit=APPEARANCE_LIMIT, content_rules=content_rules
            )
            if subject == "user":
                extraction = extraction.replace(
                    "сохранённого облика Дельты",
                    "сохранённого RP-персонажа пользователя",
                )
            prompt = extraction + "\n\n" + load_prompt("appearance_identification")
            prompt += "\n\n" + load_prompt("appearance_verification")
            payload = {
                "caption": user_message,
                "author_species": hint,
                "preliminary": observed,
                "cards": [
                    card.prompt_data(allowed_ids={item.id for item in cards})
                    for card in cards
                ],
                "trait_vocabulary": self._catalog.traits,
            }
            raw = await self._model.chat(
                system_prompt=prompt,
                user_message=json.dumps(payload, ensure_ascii=False),
                images=images,
                tool_names=frozenset(),
                json_response=True,
            )
            verified = _json(raw)
        description = verified.get("description")
        minor_reference = verified.get("minor_reference", True)
        if type(minor_reference) is not bool:
            raise ValueError("Invalid reference age marker")
        # Временная поза референса намеренно не входит в AppearanceResult:
        # она не должна стать сохранённой внешностью или действием сцены.
        reference_state = verified.get("reference_state", "")
        if not isinstance(reference_state, str) or len(reference_state) > 400:
            raise ValueError("Invalid reference state")
        status, key, evidence = (
            verified.get("status"),
            verified.get("species_id"),
            verified.get("evidence_traits"),
        )
        allowed = {card.id for card in cards}
        if (
            not isinstance(status, str)
            or not isinstance(key, str)
            or status not in {"probable", "hybrid", "unknown"}
            or key not in allowed
            or not isinstance(evidence, list)
            or len(evidence) > 8
            or any(
                not isinstance(trait, str) or trait not in self._catalog.traits
                for trait in evidence
            )
        ):
            raise ValueError("Invalid species verification")
        evidence = list(dict.fromkeys(evidence))
        if (
            (
                status == "probable"
                and (key in {"unknown", "hybrid"} or len(evidence) < 2)
            )
            or (status == "unknown" and key != "unknown")
            or (status == "hybrid" and key != "hybrid")
        ):
            raise ValueError("Unsupported species conclusion")
        if status == "probable":
            candidate = self._catalog.cards[key]
            supported = [trait for trait in evidence if trait in candidate.positive]
            if len({_region(trait) for trait in supported}) < 2 or not any(
                candidate.positive[trait] >= 3 for trait in supported
            ):
                raise ValueError("Insufficient distinguishing evidence")
        if not isinstance(description, str):
            raise ValueError("Missing visual description")
        body = validate_description(description)
        if len(body) > 1800:
            raise ValueError("Visual card too long")
        if hint:
            card = self._catalog.find(hint)
            visual_check: Literal["match", "conflict", "unconfirmed"] = "unconfirmed"
            if card and status == "probable":
                visual_check = "match" if card.id == key else "conflict"
            species = AppearanceSpecies(
                hint,
                card.id if card else None,
                "user",
                "declared",
                tuple(evidence),
                visual_check,
            )
            prefix = "Вид со слов пользователя: " + hint + ". "
        else:
            card = self._catalog.cards[key]
            species = AppearanceSpecies(
                card.name,
                key,
                "vision",
                cast(Literal["probable", "hybrid", "unknown"], status),
                tuple(evidence),
            )
            prefix = (
                ("Вероятно, " + card.name + ". ")
                if status == "probable"
                else card.name + ". "
            )
        return AppearanceResult(
            validate_description(prefix + body), species, minor_reference
        )
