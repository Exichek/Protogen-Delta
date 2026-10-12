"""Синтетические ответы двух визуальных проходов без внешнего API."""

import json


def observation(*traits: str) -> str:
    return json.dumps(
        {
            "readable": True,
            "ambiguous": False,
            "layout": {
                "orientation": "Synthetic orientation",
                "head": "Synthetic head position",
                "torso": "Synthetic torso position",
                "pelvis": "Synthetic pelvis position",
                "tail_base": "Synthetic tail base position",
            },
            "observations": "Synthetic visible character",
            "features": [
                {
                    "trait": trait,
                    "state": "present",
                    "evidence": "Synthetic visible detail",
                }
                for trait in traits or ("horns", "wings_membrane", "scales")
            ],
        }
    )


def verified(
    description: str, species_id: str = "unknown", status: str = "unknown"
) -> str:
    return json.dumps(
        {
            "species_id": species_id,
            "status": status,
            "evidence_traits": (
                ["head_wedge", "fur"] if species_id == "sergal" else ["horns", "scales"]
            ),
            "description": description,
            "minor_reference": False,
        },
        ensure_ascii=False,
    )
