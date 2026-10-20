"""Выбирать повод реакции по реплике, а не по возрастному рейтингу."""

import re

from protogen_delta.core.sticker_context import StickerContext


def reaction_context(
    tags: set[str], reply: str, *, joyful: bool = False
) -> StickerContext:
    # Слова из цитат, кода и описаний чужих реплик не означают собственное решение.
    visible = re.sub(
        r'```[\s\S]*?```|`[^`]*`|«[^»]*»|"[^"\n]*"|\*[^*\n]*\*', "", reply
    ).strip()
    if "greeting" in tags:
        return "greeting"
    if re.match(
        r"(?:нет(?:[,.!\s]|$)|не\s+(?:хочу|буду|согласен|согласна)\b|отказываюсь\b)",
        visible,
        re.I,
    ):
        return "refusal"
    if re.search(
        r"\b(?:не\s+(?:уверен|уверена|знаю)|сомневаюсь|спорно)\b", visible, re.I
    ) or tags.intersection({"oops", "tired", "confused", "bored"}):
        return "uncertain"
    if (
        "okay" in tags
        or "agreement" in tags
        or re.match(
            r"(?:да[,.!\s]|согласен\b|согласна\b|договорились\b|окей\b)", visible, re.I
        )
    ):
        return "agreement"
    if tags.intersection({"love", "cuddle", "sweet", "kiss", "boop"}):
        return "affection"
    if "laugh" in tags:
        return "laugh"
    if "happy" in tags or joyful:
        return "joy"
    return "everyday"
