"""Прямые вопросы о сохранённом поле облика, без генерации сцены."""

import re

from protogen_delta.core.appearance_profile import AppearanceProfile
from protogen_delta.core.user_state import UserState
from protogen_delta.services.fetishes import FetishTriggers, detect_fetishes

_TARGET = (
    r"(?:фетиши(?:\s+и\s+предпочтения)?|"
    r"предпочтения\s+(?:(?:текущего|выбранного|этого)\s+)?облика)"
)
_WHERE = (
    r"(?:у\s+(?:меня|тебя|дельты)|сейчас|ща|щас|там|"
    r"(?:в|для)\s+(?:(?:моем|твоем|текущем|выбранном)\s+)?(?:профиле|облике))"
)
_SAVED = r"(?:сохранены|записаны|указаны|заданы|выбраны|выставлены|стоят)"
_QUERY = re.compile(
    r"(?:дельта[\s,]+)?(?:"
    rf"какие\s+(?:{_WHERE}\s+)*{_TARGET}(?:\s+(?:{_WHERE}|{_SAVED}))*|"
    r"(?:покажи|перечисли|напомни)\s+"
    r"(?:(?:мне|сохраненные|заданные|выбранные|мои|твои|текущие)\s+)*"
    rf"{_TARGET}(?:\s+{_WHERE})*|"
    rf"что\s+(?:{_WHERE}\s+)*(?:(?:сохранено|записано|указано|задано|выбрано|стоит)\s+)?"
    rf"в\s+(?:поле|списке)\s+[«„“\"]?{_TARGET}[»“”\"]?"
    rf"(?:\s+{_WHERE})*"
    r")[\s!?.,…]*",
    re.I,
)


def is_appearance_preferences_query(message: str) -> bool:
    """Только отдельный вопрос о собственных настройках, не цитата или сцена."""
    return len(message) <= 240 and bool(
        _QUERY.fullmatch(message.strip().casefold().replace("ё", "е"))
    )


def appearance_preferences_reply(
    state: UserState, triggers: FetishTriggers, names: dict[str, str]
) -> str:
    """Показать исходный текст и словарные метки текущего облика владельца."""
    if not state.delta_appearance:
        return (
            "У тебя сейчас не выбран отдельный облик Дельты. "
            "Выбери его в «RP и функции», чтобы заполнить поля профиля."
        )
    profile = AppearanceProfile.decode(state.delta_appearance_profile)
    if profile is None or not profile.fetishes:
        return (
            "В поле «Фетиши и предпочтения» текущего облика пока ничего не записано. "
            "Заполнить его можно в «RP и функции»."
        )
    topics = detect_fetishes(profile.fetishes, triggers)
    labels = ", ".join(names.get(key, key) for key in topics)
    return (
        "В поле «Фетиши и предпочтения» текущего облика у тебя сейчас записано:\n"
        + profile.fetishes
        + "\n\nРаспознанные темы: "
        + (labels or "совпадений со словарём нет")
        + "."
    )
