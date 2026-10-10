"""Название вида и его происхождение отдельно от описания внешности."""

import json
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AppearanceSpecies:
    name: str
    species_id: str | None
    source: Literal["user", "vision"]
    status: Literal["declared", "probable", "hybrid", "unknown"]
    evidence: tuple[str, ...] = ()
    visual_check: Literal["match", "conflict", "unconfirmed"] | None = None

    def encode(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def decode(cls, value: str) -> "AppearanceSpecies | None":
        if not isinstance(value, str) or not value or len(value) > 4096:
            return None
        try:
            item = json.loads(value)
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not 1 <= len(item["name"]) <= 100
                or item.get("source") not in {"user", "vision"}
                or item.get("status")
                not in {"declared", "probable", "hybrid", "unknown"}
                or (
                    item.get("species_id") is not None
                    and not isinstance(item["species_id"], str)
                )
                or not isinstance(item.get("evidence", []), list)
                or len(item.get("evidence", [])) > 8
                or any(
                    not isinstance(trait, str) or len(trait) > 80
                    for trait in item.get("evidence", [])
                )
                or (item["source"] == "user") != (item["status"] == "declared")
                or item.get("visual_check")
                not in {None, "match", "conflict", "unconfirmed"}
                or (item.get("visual_check") is not None and item["source"] != "user")
            ):
                return None
            return cls(
                item["name"],
                item.get("species_id"),
                item["source"],
                item["status"],
                tuple(item.get("evidence", [])),
                item.get("visual_check"),
            )
        except ValueError, RecursionError, TypeError:
            return None
