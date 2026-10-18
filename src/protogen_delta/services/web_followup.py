"""Привязать уточнение о сайте к последнему ходу, не подменяя его старой темой."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from protogen_delta.core.user_state import ConversationTurn

_PAGE = re.compile(
    r"\b(?:на|с|со|в|по|этот|эту|эти|этих|этой|все|всех|первом|втором|третьем)\s+"
    r"(?:(?:этом|том|этих|тех|самом|же)\s+){0,2}"
    r"(?:сайт\w*|страниц\w*|ссылк\w*|источник\w*|стать[яюеи]\w*)\b",
    re.I,
)
_REQUEST = re.compile(
    r"\b(?:что|ещ[её]|интересн\w*|посмотри|открой|прочитай|разбери|"
    r"проанализируй|анализируй\w*|изучи|глянь|видишь)\b",
    re.I,
)
_ACTION = re.compile(
    r"\s*(?:анализируй|проанализируй|разбери|посмотри|прочитай|открой)\s*[*!?.]*\s*",
    re.I,
)
_CORRECTION = re.compile(
    r"\s*(?:анализируй|проанализируй|разбери|посмотри|прочитай)\s*\*\s*[.!]?\s*",
    re.I,
)
_DENIAL = re.compile(
    r"\b(?:не\s+(?:надо|нужно|открывай|читай|анализируй|смотри)|"
    r"без\s+(?:поиска|интернета|чтения|просмотра))\b",
    re.I,
)
_DATA = re.compile(r'```[\s\S]*?```|`[^`]*`|«[^»]*»|“[^”]*”|"[^"\n]*"|(?m:^\s*>[^\n]*)')
_ORDINAL = re.compile(
    r"\b(?P<index>[1-8]|перв\w*|втор\w*|трет\w*|четв[её]рт\w*)\b", re.I
)


def _urls(text: str) -> tuple[str, ...]:
    urls: list[str] = []
    for match in re.finditer(r"https?://[^\s<>]+", text, re.I):
        url = match.group().rstrip(".,;!?")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        url = url.rstrip("]}")
        try:
            parsed = urlsplit(url)
            valid = parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            continue
        if valid and len(url) <= 2048 and url not in urls:
            urls.append(url)
        if len(urls) == 8:
            break
    return tuple(urls)


def _reference(text: str) -> bool:
    return bool(_PAGE.search(text) and _REQUEST.search(text) or _ACTION.fullmatch(text))


@dataclass(frozen=True, slots=True)
class WebFollowup:
    urls: tuple[str, ...]
    question: str = ""

    def model_message(self, request: str) -> str:
        return (
            request + "\n\nТекущее уточнение относится к ссылкам из предыдущего хода, "
            "а не к сайтам из более старой темы. Сами ссылки не доказывают, что "
            "страницы уже прочитаны. Прочитай выбранные страницы доступным "
            "инструментом и анализируй только полученные данные:\n"
            + "\n".join(self.urls)
        )


def resolve_web_followup(
    message: str, history: Sequence[ConversationTurn]
) -> WebFollowup | None:
    """Уточнить неоднозначную ссылку; не искать подходящую тему в старой истории."""
    visible = _DATA.sub(" ", message.split("<document_content>", 1)[0]).strip()
    if _urls(visible) or _DENIAL.search(visible) or len(visible) > 300:
        return None
    turns = tuple(history)
    while _CORRECTION.fullmatch(visible) and turns:
        previous = turns[-1]
        previous_visible = _DATA.sub(" ", previous.user_message).strip()
        if (
            previous.context_closed
            or previous.proactive
            or not _reference(previous_visible)
        ):
            return None
        # Ответ на опечатку мог уже ошибочно сменить сайт; он не задаёт цель исправления.
        visible = previous_visible
        turns = turns[:-1]
    if (
        not turns
        or turns[-1].context_closed
        or turns[-1].proactive
        or _DENIAL.search(visible)
    ):
        return None
    turn = turns[-1]
    urls = _urls(_DATA.sub(" ", turn.user_message)) or _urls(turn.assistant_message)
    choice = _ORDINAL.search(visible)
    # Короткий выбор применяется только после нашего явного вопроса со ссылками.
    selecting = bool(
        len(visible) <= 80
        and turn.assistant_message.startswith("Какой сайт разобрать?")
        and (
            choice
            or re.fullmatch(r"все[.!]?", visible, re.I)
            or any(
                (urlsplit(url).hostname or "").casefold() in visible.casefold()
                for url in urls
            )
        )
    )
    if not (_reference(visible) or selecting):
        return None
    if not urls:
        return WebFollowup((), "Пришли ссылку на сайт, который нужно разобрать.")
    if re.search(r"\b(?:все|вс[её]х|кажд\w*)\b", visible, re.I):
        return WebFollowup(urls)
    if choice:
        word = choice["index"].casefold()
        index = (
            int(word) - 1
            if word.isdigit()
            else next(
                i
                for i, prefix in enumerate(("перв", "втор", "трет", "четв"))
                if word.startswith(prefix)
            )
        )
        if index < len(urls):
            return WebFollowup((urls[index],))
    hosts = [(url, urlsplit(url).hostname or "") for url in urls]
    matching = [url for url, host in hosts if host.casefold() in visible.casefold()]
    if len(matching) == 1:
        return WebFollowup(tuple(matching))
    if len(urls) == 1 and choice is None:
        return WebFollowup(urls)
    options = "\n".join(
        f"{i}. {host} — {url}" for i, (url, host) in enumerate(hosts, 1)
    )
    return WebFollowup(
        (),
        "Какой сайт разобрать?\n"
        + options
        + "\nМожно выбрать номер или попросить разобрать все.",
    )
