"""Нормализация слов для поиска эпизодов без обращения к модели."""

import re
import unicodedata

SEARCH_VERSION = 1
_WORDS = re.compile(r"[a-zа-я0-9]+")
_VOWELS = frozenset("аеиоуыэюя")
_STOP_WORDS = frozenset(
    "а без был была были было быть в во вот все вы где да для до его ее если есть "
    "еще же за и из или их к как ко когда кто ли мне мной мы на не него нее нет "
    "но о об он она они оно от по под при про с со так там те то ты у уже что "
    "это я меня мой моя мои мое твой твоя твои твое наш наша наши наше "
    "the a an and or is are was were be to of in on for with i you we it "
    "давай помнишь вспомни расскажи поговорим вернемся обсуждали говорили".split()
)


def _suffixes(text: str) -> tuple[str, ...]:
    return tuple(sorted(text.split(), key=len, reverse=True))


_GERUND = _suffixes("ив ивши ившись ыв ывши ывшись")
_GERUND_CONDITIONAL = _suffixes("в вши вшись")
_ADJECTIVE = _suffixes(
    "ее ие ые ое ими ыми ей ий ый ой ем им ым ом его ого ему ому их ых ую юю ая яя ою ею"
)
_PARTICIPLE = _suffixes("ивш ывш ующ")
_PARTICIPLE_CONDITIONAL = _suffixes("ем нн вш ющ щ")
_VERB = _suffixes(
    "ила ыла ена ейте уйте ите или ыли ей уй ил ыл им ым ен ило ыло ено "
    "ят ует уют ит ыт ены ить ыть ишь ую ю"
)
_VERB_CONDITIONAL = _suffixes("ла на ете йте ли й л ем н ло но ет ют ны ть ешь нно")
_NOUN = _suffixes(
    "а ев ов ие ье е иями ями ами еи ии и ией ей ой ий й иям ям ием ем "
    "ам ом о у ах иях ях ы ь ию ью ю ия ья я"
)


def _remove(
    word: str, suffixes: tuple[str, ...], start: int, *, conditional: bool = False
) -> tuple[str, bool]:
    for suffix in suffixes:
        end = len(word) - len(suffix)
        if end < start or not word.endswith(suffix):
            continue
        if conditional and (end <= start or word[end - 1] not in "ая"):
            continue
        return word[:end], True
    return word, False


def _region(word: str, start: int = 0) -> int:
    for index in range(start + 1, len(word)):
        if word[index - 1] in _VOWELS and word[index] not in _VOWELS:
            return index + 1
    return len(word)


def _russian_stem(word: str) -> str:
    """Снять окончания по правилам русского Snowball; это не поиск синонимов.

    Описание алгоритма: https://snowballstem.org/algorithms/russian/stemmer.html
    """
    rv = next((i + 1 for i, char in enumerate(word) if char in _VOWELS), len(word))
    r2 = _region(word, _region(word))
    word, removed = _remove(word, _GERUND, rv)
    if not removed:
        word, removed = _remove(word, _GERUND_CONDITIONAL, rv, conditional=True)
    if not removed:
        word, _ = _remove(word, ("ся", "сь"), rv)
        word, adjective = _remove(word, _ADJECTIVE, rv)
        if adjective:
            word, participle = _remove(word, _PARTICIPLE, rv)
            if not participle:
                word, _ = _remove(word, _PARTICIPLE_CONDITIONAL, rv, conditional=True)
        else:
            word, verb = _remove(word, _VERB, rv)
            if not verb:
                word, verb = _remove(word, _VERB_CONDITIONAL, rv, conditional=True)
            if not verb:
                word, _ = _remove(word, _NOUN, rv)
    word, _ = _remove(word, ("и",), rv)
    word, _ = _remove(word, ("ость", "ост"), max(rv, r2))
    if word.endswith("нн") and len(word) - 1 >= rv:
        word = word[:-1]
    else:
        word, superlative = _remove(word, ("ейше", "ейш"), rv)
        if superlative:
            if word.endswith("нн") and len(word) - 1 >= rv:
                word = word[:-1]
        else:
            word, _ = _remove(word, ("ь",), rv)
    return word


def memory_terms(text: str, *, limit: int = 128) -> tuple[str, ...]:
    """Получить ограниченный набор уникальных слов, пригодных для FTS MATCH."""
    if limit <= 0:
        return ()
    normalized = unicodedata.normalize("NFKC", text[:8000]).casefold().replace("ё", "е")
    terms: dict[str, None] = {}
    for word in _WORDS.findall(normalized):
        if not 2 <= len(word) <= 64 or word in _STOP_WORDS:
            continue
        stem = _russian_stem(word) if re.fullmatch(r"[а-я]+", word) else word
        if len(stem) < 2:
            continue
        terms[stem] = None
        if len(terms) >= limit:
            break
    return tuple(terms)


def memory_index_text(text: str) -> str:
    """Сохранить нормализованный текст отдельно от оригинального эпизода."""
    return " ".join(memory_terms(text))
