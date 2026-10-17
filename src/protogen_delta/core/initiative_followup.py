"""Консервативно распознавать собственный нейтральный план с явным днём."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from protogen_delta.services.turn_contract import unquoted_control_text

MSK = timezone(timedelta(hours=3))
TOPICS = {
    "interview": r"собеседовани\w*",
    "exam": r"экзамен\w*|зач[её]т\w*",
    "presentation": r"презентаци\w*|выступлени\w*",
    "project": r"защит[ау]\s+(?:диплом\w*|проект\w*)",
    "release": r"релиз\w*",
    "competition": r"турнир\w*|соревновани\w*",
}
_DAY = re.compile(r"\b(?:сегодня|завтра|послезавтра|\d{4}-\d{2}-\d{2})\b", re.I)
_OWN = re.compile(
    r"\b(?:у\s+меня|мне\s+предстоит|пойду|поеду|буду|иду|сдаю|сдам|выступаю|"
    r"участвую|защищаю|запускаю|я\s+(?:(?:сегодня|завтра|послезавтра)\s+)?на)\b",
    re.I,
)
_UNCERTAIN = re.compile(
    r"\b(?:если|может|возможно|наверно|наверное|вряд|допустим|представь|например|"
    r"спросил\w*|сказал\w*|говорил\w*|рассказал\w*|думаю)\b",
    re.I,
)
_SENSITIVE = re.compile(
    r"\b(?:врач\w*|медицин\w*|больниц\w*|диагноз\w*|болезн\w*|лечени\w*|"
    r"секс\w*|интим\w*|свидани\w*|псих\w*|похорон\w*|смерт\w*)\b",
    re.I,
)
_CLOSED = re.compile(
    r"\b(?:не|нет|отмен\w*|перен[её]с\w*|передумал\w*|прош[её]л|прошла|"
    r"сдал\w*|сходил\w*|состоял\w*|закончил\w*|завершил\w*)\b",
    re.I,
)
_STOP = re.compile(
    r"(?:^|[.!?\n])\s*(?:дельта[,\s]+)?(?:пожалуйста[,\s]+)?"
    r"(?:не\s+(?:спрашивай|напоминай|интересуйся)|забудь\s+(?:(?:мой|мои|все|эти)\s+)?план\w*)\b",
    re.I,
)
_SECRET = re.compile(
    r"sk-[\w-]+|\b(?:пароль|password|api[_ -]?key|токен|ключ)\b|\d{10,}", re.I
)


@dataclass(frozen=True, slots=True)
class FollowupPlan:
    topic: str
    source: str
    event_day: str
    due_at: float
    expires_at: float


def visible_plan_text(text: str) -> str:
    """Пересылки отсекаются вызывающим кодом; здесь убрать цитаты и команды."""
    if not 1 <= len(text) <= 600 or text.lstrip().startswith(("/", "*")):
        return ""
    return unquoted_control_text(text).strip()


def followup_topics(text: str) -> tuple[str, ...]:
    return tuple(
        key
        for key, pattern in TOPICS.items()
        if re.search(r"\b(?:" + pattern + r")\b", text, re.I)
    )


def closes_followups(text: str) -> bool:
    return bool(_STOP.search(text))


def followup_preference(text: str) -> bool | None:
    """Явный отказ относится ко всем планам; отдельная отмена не меняет настройку."""
    if re.search(
        r"(?:^|[.!?\n])\s*(?:дельта[,\s]+)?(?:пожалуйста[,\s]+)?"
        r"не\s+(?:спрашивай|интересуйся)\b.{0,40}\bплан\w*",
        text,
        re.I,
    ):
        return False
    if re.match(
        r"(?:можешь|разрешаю)\s+(?:снова\s+)?(?:спрашивать|интересоваться)\b.{0,40}\bплан\w*",
        text,
        re.I,
    ):
        return True
    return None


def closed_followup_topics(text: str) -> tuple[str, ...]:
    if "?" in text or _UNCERTAIN.search(text):
        return ()
    completed = re.search(
        r"\b(?:прош[её]л|прошла|сдал\w*|сходил\w*|состоял\w*|закончил\w*|завершил\w*)\b",
        text,
        re.I,
    )
    rejected = re.search(r"\bне\s+(?:пойду|буду|сдам|участвую|иду|еду)\b", text, re.I)
    cancelled = any(
        not re.search(r"\bне(?:\s+\w+){0,2}\s*$", text[: match.start()], re.I)
        for match in re.finditer(
            r"\b(?:отмен\w*|перен[её]с\w*|передумал\w*)\b", text, re.I
        )
    )
    return followup_topics(text) if completed or rejected or cancelled else ()


def extract_followup_plan(text: str, now: float) -> FollowupPlan | None:
    """Не угадывать срок; спросить после указанного дня, а не в разгар события."""
    visible = visible_plan_text(text)
    topics = followup_topics(visible)
    days = list(_DAY.finditer(visible))
    if (
        len(topics) != 1
        or len(days) != 1
        or "?" in visible
        or visible != text.strip()
        or _SECRET.search(visible)
        or not _OWN.search(visible)
        or _UNCERTAIN.search(visible)
        or _SENSITIVE.search(visible)
        or _CLOSED.search(visible)
    ):
        return None
    today = datetime.fromtimestamp(now, MSK).date()
    value = days[0].group().casefold()
    if value in {"сегодня", "завтра", "послезавтра"}:
        day = today + timedelta(
            days={"сегодня": 0, "завтра": 1, "послезавтра": 2}[value]
        )
    else:
        try:
            day = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
    if not today <= day <= today + timedelta(days=14):
        return None
    due = datetime.combine(day + timedelta(days=1), datetime.min.time(), MSK).replace(
        hour=12
    )
    return FollowupPlan(
        topics[0],
        visible,
        day.isoformat(),
        due.timestamp(),
        (due + timedelta(days=3)).timestamp(),
    )
