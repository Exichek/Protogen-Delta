"""Прямые вопросы о сохранённом поле облика, без генерации сцены."""

import re

from protogen_delta.core.appearance_profile import active_appearance_profile
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

_PROFILE_QUERY = re.compile(
    r"(?:дельта[\s,]+)?(?:"
    r"какие\s+(?:(?:у\s+(?:тебя|дельты)|твои|сохраненные|заданные)\s+)*предпочтения|"
    r"(?:покажи|перечисли|напомни)\s+(?:мне\s+)?(?:твой\s+|сохраненный\s+)?"
    r"(?:профиль|характер|предпочтения|границы)(?:\s+(?:дельты|облика))?|"
    r"(?:какой\s+(?:у\s+тебя\s+|твой\s+)?характер)|"
    r"(?:не\s+в\s*смысле[\s,]*)?(?:какие|что)\s+я\s+"
    r"(?:для\s+(?:этого\s+)?облика\s+|в\s+профиле\s+|тебе\s+|там\s+)*"
    r"(?:указал|написал|задал|сохранил)"
    r"(?:\s+(?:для\s+(?:этого\s+)?облика|в\s+профиле))?"
    r")[\s!?.,…)\]]*",
    re.I,
)


def _normalized(message: str) -> str:
    return message.strip().casefold().replace("ё", "е")


def is_appearance_preferences_query(message: str) -> bool:
    """Только отдельный вопрос о собственных настройках, не цитата или сцена."""
    return len(message) <= 240 and bool(
        _QUERY.fullmatch(_normalized(message))
        or _PROFILE_QUERY.fullmatch(_normalized(message))
    )


def appearance_preferences_reply(
    state: UserState, triggers: FetishTriggers, names: dict[str, str], message: str = ""
) -> str:
    """Показать исходный текст и словарные метки активного образа владельца."""
    profile = active_appearance_profile(state)
    if _PROFILE_QUERY.fullmatch(_normalized(message)):
        if profile is None or not profile.encode():
            return (
                "В сохранённом профиле активного образа Дельты пока ничего не записано. "
                "Заполни поля и нажми «Сохранить характер и поведение». "
                "У стандартного и загруженного облика отдельные настройки."
            )
        fields = (
            ("Характер", profile.personality),
            ("Манера поведения", profile.behavior),
            ("Предпочтения в сценах", profile.preferences),
            ("Границы", profile.boundaries),
            ("Фетиши и предпочтения", profile.fetishes),
        )
        return "В сохранённом профиле активного образа Дельты указано:\n\n" + "\n".join(
            label + ": " + value for label, value in fields if value
        )
    if profile is None or not profile.fetishes:
        return (
            "В поле «Фетиши и предпочтения» активного образа пока ничего не записано. "
            "Заполнить его можно в «RP и функции»."
        )
    topics = detect_fetishes(profile.fetishes, triggers)
    labels = ", ".join(names.get(key, key) for key in topics)
    return (
        "В поле «Фетиши и предпочтения» активного образа у тебя сейчас записано:\n"
        + profile.fetishes
        + "\n\nРаспознанные темы: "
        + (labels or "совпадений со словарём нет")
        + "."
    )
