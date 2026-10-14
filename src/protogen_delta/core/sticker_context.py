"""Возрастной фильтр и повод для реакции — разные свойства стикера."""

from typing import Literal

StickerContext = Literal[
    "greeting",
    "uncertain",
    "error",
    "agreement",
    "refusal",
    "joy",
    "laugh",
    "affection",
    "everyday",
    "adult",
]
CONTEXT_LABELS: dict[StickerContext, str] = {
    "greeting": "приветствие",
    "uncertain": "замешательство и спор",
    "error": "ошибка приложения",
    "agreement": "согласие",
    "refusal": "отказ",
    "joy": "радость",
    "laugh": "смех",
    "affection": "нежность",
    "everyday": "повседневное",
    "adult": "взрослый контекст",
}


def legacy_contexts(tags: tuple[str, ...], rating: str) -> tuple[StickerContext, ...]:
    if rating == "adult":
        return ("adult",)
    mapping: tuple[tuple[StickerContext, set[str]], ...] = (
        ("greeting", {"greeting"}),
        ("refusal", {"refusal", "angry"}),
        ("agreement", {"agreement", "okay"}),
        ("uncertain", {"confused", "oops", "tired", "bored"}),
        ("error", {"error"}),
        ("laugh", {"laugh"}),
        ("affection", {"love", "sweet", "cuddle", "kiss", "boop"}),
        ("joy", {"happy"}),
    )
    result = tuple(category for category, keys in mapping if keys.intersection(tags))
    return result or ("everyday",)
