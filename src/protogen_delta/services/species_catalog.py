"""Версионированные карточки и подбор кандидатов по наблюдаемым признакам."""

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any


@dataclass(frozen=True, slots=True)
class SpeciesCard:
    id: str
    name: str
    category: str
    aliases: tuple[str, ...]
    positive: dict[str, int]
    contradictions: tuple[str, ...]
    confusable: tuple[str, ...]
    features: tuple[str, ...]
    exceptions: tuple[str, ...]
    sources: tuple[dict[str, str], ...]

    def prompt_data(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "features": self.features,
            "supporting_traits": list(self.positive),
            "diagnostic_traits": [
                trait for trait, weight in self.positive.items() if weight >= 3
            ],
            "exceptions": self.exceptions,
            "confusable_with": self.confusable,
            "sources": self.sources,
        }


class SpeciesCatalog:
    def __init__(self, value: dict[str, Any]) -> None:
        if value.get("version") != 1 or not isinstance(value.get("traits"), dict):
            raise ValueError("Unsupported species catalog")
        self.traits: dict[str, str] = value["traits"]
        self.cards: dict[str, SpeciesCard] = {}
        self.aliases: dict[str, str] = {}
        for item in value["cards"]:
            card = SpeciesCard(
                id=item["id"],
                name=item["name"],
                category=item["category"],
                aliases=tuple(item["aliases"]),
                positive=dict(item["positive"]),
                contradictions=tuple(item["contradictions"]),
                confusable=tuple(item["confusable_with"]),
                features=tuple(item["features"]),
                exceptions=tuple(item["exceptions"]),
                sources=tuple(item["sources"]),
            )
            if card.id in self.cards or not card.sources or not card.features:
                raise ValueError("Invalid species card")
            if (card.positive.keys() | set(card.contradictions)) - self.traits.keys():
                raise ValueError("Unknown catalog trait")
            if any(
                type(weight) is not int or not 1 <= weight <= 5
                for weight in card.positive.values()
            ):
                raise ValueError("Invalid trait weight")
            self.cards[card.id] = card
            for alias in (*card.aliases, card.name):
                key = alias.casefold().strip()
                if key in self.aliases and self.aliases[key] != card.id:
                    raise ValueError("Ambiguous species alias")
                self.aliases[key] = card.id
        if any(
            set(card.confusable) - self.cards.keys() for card in self.cards.values()
        ):
            raise ValueError("Unknown confusable species")

    def find(self, name: str) -> SpeciesCard | None:
        key = self.aliases.get(name.casefold().strip())
        return self.cards.get(key) if key else None

    def select(
        self, features: dict[str, str], declared: str = ""
    ) -> tuple[SpeciesCard, ...]:
        """Оценка служит только поиску карточек, не вероятности правильного вида."""
        present = {key for key, state in features.items() if state == "present"}
        absent = {key for key, state in features.items() if state == "absent"}

        def score(card: SpeciesCard) -> int:
            return (
                sum(
                    weight
                    for trait, weight in card.positive.items()
                    if trait in present
                )
                - sum(2 for trait in card.contradictions if trait in present)
                - sum(1 for trait in card.positive if trait in absent)
            )

        ranked = sorted(self.cards.values(), key=lambda card: (-score(card), card.id))
        selected = [
            card
            for card in ranked
            if card.id not in {"hybrid", "unknown"} and score(card) > 0
        ][:2]
        hint = self.find(declared) if declared else None
        if hint and hint not in selected:
            selected = [hint, *selected[:2]]
        # Include a confusing alternative; otherwise the verifier cannot repair
        # a first-pass head mistake that ranks only related canine categories.
        alternatives = sorted(
            selected,
            key=lambda card: -max(
                (weight for trait, weight in card.positive.items() if trait in present),
                default=0,
            ),
        )
        for card in alternatives:
            if len(selected) >= 3:
                break
            for key in card.confusable:
                alternative = self.cards[key]
                if alternative not in selected and key not in {"hybrid", "unknown"}:
                    selected.append(alternative)
                    break
        selected.extend(
            self.cards[key]
            for key in ("hybrid", "unknown")
            if self.cards[key] not in selected
        )
        return tuple(selected)

    def observation_vocabulary(self) -> str:
        return json.dumps(self.traits, ensure_ascii=False)


@lru_cache(maxsize=1)
def species_catalog() -> SpeciesCatalog:
    resource = files("protogen_delta.config.data").joinpath("species_catalog.json")
    return SpeciesCatalog(json.loads(resource.read_text("utf-8")))
