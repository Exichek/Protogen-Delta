"""Разбирать явные сроки напоминания, не угадывая намерение моделью."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

MSK = timezone(timedelta(hours=3), "МСК")
MAX_DELAY = 365 * 86400
_REMIND_START = re.compile(
    r"\s*(?:дельта[,\s]+)?(?:пожалуйста[,\s]+)?напомни"
    r"(?:[,\s]+пожалуйста)?(?:\s+мне)?(?:[,\s]+пожалуйста)?\s+",
    re.I,
)
REMINDER_HELP = (
    "Напиши, когда и о чём напомнить:\n"
    "• /remind через 1 час выпить воду\n"
    "• Напомни завтра в 18:30 позвонить\n"
    "• /remind 2026-12-20 12:00 встреча\n"
    "Время — московское. /reminders — список; /reminders cancel 12 — отмена."
)


@dataclass(frozen=True, slots=True)
class ReminderRequest:
    text: str
    due_at: float


def is_reminder_message(text: str | None) -> bool:
    if not text or len(text) > 1500:
        return False
    value = reminder_payload(text)
    if _REMIND_START.match(text):
        return bool(
            re.match(
                r"(?:через|сегодня|завтра|послезавтра|в\s+\d|\d{4}-\d{2}-\d{2})",
                value,
                re.I,
            )
        )
    return bool(
        re.fullmatch(
            r"(?:покажи|список)\s+(?:мои\s+)?напоминани[яй][.!?]*|"
            r"(?:отмени|удали)\s+напоминание(?:\s+#?\d{1,10})?[.!]?",
            value,
            re.I,
        )
    )


def reminder_payload(text: str) -> str:
    value = re.sub(r"^\s*дельта[,\s]+", "", text, flags=re.I)
    return _REMIND_START.sub("", value, count=1).strip()


def parse_reminder(payload: str, now: float) -> ReminderRequest:
    """Срок хранится в UTC; абсолютное время без зоны трактуется как МСК."""
    if not payload.strip():
        raise ValueError(REMINDER_HELP)
    if not 1 <= len(payload) <= 1500:
        raise ValueError("Просьба слишком длинная: не более 1500 символов")
    local_now = datetime.fromtimestamp(now, MSK)
    value = payload.strip()
    text = ""
    due: float | None = None
    relative = re.match(
        r"через\s+(?:(?P<number>\d{1,7}(?:[.,]\d{1,3})?)\s*)?"
        r"(?P<unit>секунд(?:у|ы)?|сек|с|минут(?:у|ы)?|мин|м|час(?:а|ов)?|ч|день|дня|дней|д)"
        r"(?:\s+|:\s*)(?P<text>.+)$",
        value,
        re.I | re.S,
    )
    half_hour = re.match(r"через\s+полчаса(?:\s+|:\s*)(.+)$", value, re.I | re.S)
    if relative:
        unit = relative["unit"].casefold()
        seconds = (
            1
            if unit.startswith("с")
            else 60 if unit.startswith("м") else 3600 if unit.startswith("ч") else 86400
        )
        amount = Decimal((relative["number"] or "1").replace(",", ".")) * seconds
        if not 1 <= amount <= MAX_DELAY:
            raise ValueError("Срок должен быть от секунды до 365 дней")
        due = now + float(amount)
        text = relative["text"]
    elif half_hour:
        due, text = now + 1800, half_hour[1]
    else:
        absolute = re.match(
            r"(?:(?P<day>сегодня|завтра|послезавтра)\s+)?в\s+"
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
            r"(?:\s+(?:мск|по\s+москве))?(?:\s+|:\s*)(?P<text>.+)$",
            value,
            re.I | re.S,
        )
        dated = re.match(
            r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?:в\s+)?"
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})"
            r"(?:\s+(?:мск|по\s+москве))?(?:\s+|:\s*)(?P<text>.+)$",
            value,
            re.I | re.S,
        )
        match = absolute or dated
        if match is not None:
            try:
                if dated:
                    target = datetime.fromisoformat(dated["date"]).replace(tzinfo=MSK)
                else:
                    day = (match["day"] or "").casefold()
                    offset = {"": 0, "сегодня": 0, "завтра": 1, "послезавтра": 2}[day]
                    target = local_now + timedelta(days=offset)
                target = target.replace(
                    hour=int(match["hour"]),
                    minute=int(match["minute"]),
                    second=0,
                    microsecond=0,
                )
                if absolute and not absolute["day"] and target.timestamp() <= now:
                    target += timedelta(days=1)
                due, text = target.timestamp(), match["text"]
            except ValueError as error:
                raise ValueError("Некорректная дата или время") from error
    if due is None:
        raise ValueError(REMINDER_HELP)
    text = " ".join(text.split()).strip(" :")
    if not text or len(text) > 500:
        raise ValueError("Текст напоминания должен содержать от 1 до 500 символов")
    if re.match(r"(?:UTC|GMT|по\s+(?!москве)\w+)", text, re.I):
        raise ValueError("Укажи срок по московскому времени (МСК)")
    if not now < due <= now + MAX_DELAY:
        raise ValueError("Время уже прошло или срок превышает 365 дней")
    return ReminderRequest(text, due)


def reminder_time(at: float) -> str:
    return datetime.fromtimestamp(at, MSK).strftime("%d.%m.%Y в %H:%M:%S МСК")
