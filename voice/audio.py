import hashlib
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

import config
from config import (
    SAMPLE_RATE, DEFAULT_ACTOR_SPEED, TARGET_DBFS,
    DC_REF_WAV, OUTPUT_DIR, WHISPER_LANGUAGE, TTS_TAIL_SILENCE_MS,
    TTS_END_PADDING,
)
# DEVICE is read as `config.DEVICE` at the synthesis call site, NOT imported by
# name here: config resolves it by importing Torch on first access, and naming it
# in this `from` list would drag the whole 1.9 s Torch import into every process
# that so much as touches audio.py (the GUI, the bot, most test suites).
from utils import audio_hash_from_path, safe_empty_cuda_cache

logger = logging.getLogger("assistant.audio")

try:
    from num2words import num2words as _num2words
except Exception:  # pragma: no cover - dependency missing
    _num2words = None


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:\d{1,2}[.)]|[-•*–—])[ \t]+(?=\S)", re.M)
_ORD_RU = ("Во-первых", "Во-вторых", "В-третьих", "В-четвёртых", "В-пятых",
           "В-шестых", "В-седьмых", "В-восьмых", "В-девятых", "В-десятых")
_ORD_EN = ("First", "Second", "Third", "Fourth", "Fifth",
           "Sixth", "Seventh", "Eighth", "Ninth", "Tenth")


def spoken_lists(text: str) -> str:
    """«1. … 2. … 3. …» read aloud as «Во-первых, … Во-вторых, … И в заключение, …».

    A list said as "один точка … два точка" is painful to listen to, and once
    the newlines are collapsed the numbers run into the sentences around them.
    Only a run of two or more items is a list; a lone "1." stays as written.
    """
    items = list(_LIST_ITEM_RE.finditer(text))
    if len(items) < 2:
        return text
    ru = bool(re.search(r"[А-Яа-яЁё]", text))
    words, last = (_ORD_RU, "И в заключение") if ru else (_ORD_EN, "Finally")
    out, pos = [], 0
    for k, m in enumerate(items):
        out.append(text[pos:m.start()])
        if k == len(items) - 1 and len(items) >= 3:
            word = last
        elif k < len(words):
            word = words[k]
        else:
            word = ("Пункт %d" if ru else "Point %d") % (k + 1)
        out.append(word + ", ")
        pos = m.end()
    out.append(text[pos:])
    # each item ends a sentence, or the next "Во-вторых" runs into it
    return re.sub(r"([^.!?:;\s])[ \t]*\n(?=(?:%s|%s|Пункт|Point))" % (
        "|".join(words), last), r"\1.\n", "".join(out))


def clean_text(text: str) -> str:
    # Source code is shown, never spoken: a fenced block is dropped from the
    # voice track (the chat gets it as text regardless of the voice setting).
    text = _FENCE_RE.sub(" ", text)
    text = spoken_lists(text)
    # final safety net: drop any leftover markup tag (e.g. <think>) so it is never voiced
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r'[—–−\-«»"""„()\[\]{}\\/|]', " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def censor_profanity(text: str) -> str:
    replacements = {
        "damn": "darn", "shit": "stuff", "crap": "junk",
        "hell": "heck", "ass": "butt", "bastard": "rascal",
        "freaking": "flipping", "screwed up": "messed up",
    }
    for bad, good in replacements.items():
        text = re.sub(r"\b" + re.escape(bad) + r"\b", good, text, flags=re.IGNORECASE)
    return text


_APOS = frozenset("''")  # russtress places these AFTER the stressed vowel

_VOWELS_ANY = set("аеёиоуыэюяaeiouy")


def _apos_to_plus(text: str) -> str:
    """Convert russtress apostrophe-after-vowel to plus-before-vowel format."""
    out = []
    i = 0
    # Only a VOWEL followed by the marker is a stress mark. Any other
    # apostrophe -- the quotes of a leaked JSON blob, an English possessive,
    # "don't" -- used to become a '+' too, and F5 read those aloud as "плюс"
    # (live, 2026-09-12).
    while i < len(text):
        if i + 1 < len(text) and text[i + 1] in _APOS and text[i].lower() in _VOWELS_ANY:
            out.append("+")
            out.append(text[i])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def stress_plus(ctx, text: str) -> str:
    if not text.strip():
        return text
    out = text
    # `ctx.models` is None whenever nothing has been loaded — a headless run, a
    # test context, or a turn that reached TTS before the model bundle was ready.
    # This line used to reach straight through it and raise AttributeError, which
    # took down the whole TTS call and, with it, the reply that call was voicing.
    # Every OTHER step in this function already degrades to plain text; this one
    # now does too — unaccented speech beats no speech.
    _models = getattr(ctx, "models", None)
    if getattr(_models, "accentor_loaded", False):
        try:
            out = _apos_to_plus(_models.accentor(text))
        except Exception as exc:
            logger.debug("Accentor failed: %s", exc)
            out = text
    # Manual per-word overrides take PRECEDENCE over the automatic marker (and work
    # even when the auto accentor is unavailable). Edge cases set in the GUI win.
    try:
        from stress_overrides import get_overrides
        out = get_overrides().apply(out)
    except Exception as exc:
        logger.debug("Stress overrides skipped: %s", exc)
    return out


# The leading minus is only a negative sign when it is NOT glued to a preceding
# word char: "-5" / "(-5)" stay negative, but a hyphen inside a range or
# hyphenated group ("2020-2024", "10-15", "8-800-555", "Windows-10") must not be
# read as "минус" — there each number is voiced on its own as a positive value.
_NUMBER_RE = re.compile(r"(?:(?<!\w)-)?\d+(?:[.,]\d+)?")
# A number written with space/NBSP/narrow-NBSP thousands separators: 1+3-digit
# groups (e.g. "1 000 000", "12 500"), with an optional decimal tail.
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?")


# Calendar dates must be converted BEFORE generic number voicing: an ISO date
# "2026-06-12" would otherwise be read as "две тысячи двадцать шесть минус шесть
# минус двенадцать", and "12.06.2026" as "двенадцать целых шесть сотых…".
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DOT_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_MONTH_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря")


def _ru_ordinal(n: int, neuter_day: bool) -> Optional[str]:
    """Russian ordinal from num2words, inflected for dates: day of month as
    neuter nominative ('первое', 'третье'), year as genitive masc.
    ('две тысячи двадцать шестого'). Deterministic suffix morphology on the
    masc. nominative that num2words produces."""
    if _num2words is None:
        return None
    try:
        ord_masc = _num2words(n, lang="ru", to="ordinal")
    except Exception:
        return None
    if ord_masc.endswith("ий"):                  # 'третий' -> 'третье'/'третьего'
        return ord_masc[:-2] + ("ье" if neuter_day else "ьего")
    if ord_masc.endswith(("ый", "ой")):          # 'первый'/'шестой' -> 'первое'/'шестого'
        return ord_masc[:-2] + ("ое" if neuter_day else "ого")
    return ord_masc


def _date_to_russian(day: int, month: int, year: int) -> Optional[str]:
    """'двенадцатое июня две тысячи двадцать шестого года'; None if not a date."""
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1000 <= year <= 9999):
        return None
    d = _ru_ordinal(day, neuter_day=True)
    y = _ru_ordinal(year, neuter_day=False)
    if not d or not y:
        return None
    return f"{d} {_MONTH_GEN[month - 1]} {y} года"


_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _time_to_russian(match: "re.Match") -> str:
    """'14:30' -> 'четырнадцать тридцать', '14:05' -> 'четырнадцать ноль пять'."""
    if _num2words is None:
        return match.group(0)
    h, m = int(match.group(1)), int(match.group(2))
    try:
        hw = _num2words(h, lang="ru")
        if m == 0:
            mw = "ноль ноль"
        elif m < 10:
            mw = "ноль " + _num2words(m, lang="ru")
        else:
            mw = _num2words(m, lang="ru")
        return f"{hw} {mw}"
    except Exception:
        return match.group(0)


def _number_to_russian_words(match: "re.Match") -> str:
    """Convert one numeric token to Russian words for TTS. Decimals use a comma or
    dot; a fractional part of all zeros is voiced as the integer. Falls back to the
    original token if num2words is unavailable or the value can't be parsed."""
    token = match.group(0)
    if _num2words is None:
        return token
    norm = token.replace(",", ".")
    try:
        if "." in norm:
            value = float(norm)
            if value.is_integer():
                return _num2words(int(value), lang="ru")
            return _num2words(value, lang="ru")
        return _num2words(int(norm), lang="ru")
    except Exception:
        return token


# "На этикетке 2 строки" was voiced "два строки" (live, 2026-09-12): num2words
# knows nothing about the noun that follows. One and two are the only numerals
# that agree in gender, and the noun's ending after them says which gender.
_ONE_TWO_RE = re.compile(r"(?<![\d.,])(\d{1,4})(?!\d)([ \u00a0]+)([а-яё]+)", re.IGNORECASE)


def _agree_one_two(match: "re.Match") -> str:
    num, gap, noun = match.group(1), match.group(2), match.group(3)
    n = int(num)
    if _num2words is None:
        return match.group(0)
    if noun.lower() in _MONTH_GEN and 1 <= n <= 31:
        # "1 января" is a date, not a count: "первое января".
        d = _ru_ordinal(n, neuter_day=True)
        return (d + gap + noun) if d else match.group(0)
    if n % 100 in (11, 12):
        return match.group(0)
    last = n % 10
    if last not in (1, 2):
        return match.group(0)
    low = noun.lower()
    word = ""
    if last == 2 and low.endswith(("ы", "и")):              # две строки, две минуты
        word = "две"
    elif last == 1:
        if low.endswith(("а", "я")):                        # одна строка, одна неделя
            word = "одна"
        elif low.endswith(("о", "е")) and len(low) > 2:     # одно окно, одно письмо
            word = "одно"
    if not word:
        return match.group(0)
    try:
        head = _num2words(n - last, lang="ru") + " " if n - last else ""
    except Exception:
        return match.group(0)
    return head + word + gap + noun


# A Cyrillic acronym with no vowel ("ЦБ РФ", "ФСБ", "МВД") has no pronunciation
# as a word: the voice produced "цбров" for "ЦБ РФ". Read it letter by letter.
# One with a vowel (США, ООН, СМИ, НАТО) is said as a word and is left alone.
_RU_LETTER_NAMES = {
    "а": "а", "б": "бэ", "в": "вэ", "г": "гэ", "д": "дэ", "е": "е", "ё": "ё",
    "ж": "жэ", "з": "зэ", "и": "и", "й": "и краткое", "к": "ка", "л": "эль",
    "м": "эм", "н": "эн", "о": "о", "п": "пэ", "р": "эр", "с": "эс", "т": "тэ",
    "у": "у", "ф": "эф", "х": "ха", "ц": "цэ", "ч": "че", "ш": "ша", "щ": "ща",
    "ъ": "твёрдый знак", "ы": "ы", "ь": "мягкий знак", "э": "э", "ю": "ю", "я": "я",
}
_RU_VOWELS = set("аеёиоуыэюя")
_RU_ACRONYM_RE = re.compile(r"(?<![А-ЯЁа-яё])([А-ЯЁ]{2,})(?![а-яё])")
_RU_SPELLED_ACRONYMS = frozenset(("ГИБДД",))   # said letter by letter despite the vowel


def _ru_acronym(match: "re.Match") -> str:
    word = match.group(1)
    if word not in _RU_SPELLED_ACRONYMS and any(ch.lower() in _RU_VOWELS for ch in word):
        # Said as a word -- and said better in ordinary case: the voice read
        # the sign "У ОЛЬГИ" as "у Олюги" (live, 2026-09-12); the fine-tune
        # rarely saw shouting. Three letters or fewer (США, ООН, СМИ) stay as
        # they are, they are acronyms the voice knows as words.
        return word if len(word) <= 3 else word.capitalize()
    return "-".join(_RU_LETTER_NAMES.get(ch.lower(), ch) for ch in word)


def spell_cyrillic_acronyms(text: str) -> str:
    """ЦБ РФ -> цэ-бэ эр-эф; ОЛЬГИ -> Ольги. Idempotent."""
    return _RU_ACRONYM_RE.sub(_ru_acronym, text or "")


# A link is for the screen, not the voice. Read aloud, "https://cbr.ru/..."
# came out as "Вплюс отс плюс ссылка" (live 2026-09-13): the scheme, the
# slashes and the Latin host are all outside what the Russian voice can say.
# The written reply keeps the URL; the spoken one says "ссылка" in its place
# (or drops it when the sentence already says so: "вот ссылка: https://...").
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_URL_LEAD_RE = re.compile(r"ссылк|link", re.IGNORECASE)


def _voice_urls(text: str) -> str:
    def _repl(m):
        # The same sentence, up to the link: "вот ссылка: https://..."
        lead = re.split(r"[.!?\n]", text[max(0, m.start() - 80):m.start()])[-1]
        return "" if _URL_LEAD_RE.search(lead) else "ссылка"
    out = _URL_RE.sub(_repl, text)
    return re.sub(r"[ \t]{2,}", " ", out).replace(" ,", ",").replace(" .", ".")


def post_process_answer(text: str) -> str:
    # Spell out numbers in Russian words so F5-TTS voices them naturally (the model
    # is told to write digits for on-screen accuracy; this runs ONLY on TTS input,
    # never on the displayed text). num2words handles any integer/decimal correctly
    # — the old hand map only covered a few values and corrupted decimals.
    text = spoken_lists(text)            # before digits become words and "*" bullets vanish
    text = text.replace("*", "")
    text = _voice_urls(text)
    # Dates first — see _ISO_DATE_RE: generic number voicing would butcher them.
    text = _ISO_DATE_RE.sub(
        lambda m: _date_to_russian(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        or m.group(0), text)
    text = _DOT_DATE_RE.sub(
        lambda m: _date_to_russian(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        or m.group(0), text)
    # Times next ("14:30" / "14:05") — the bare colon survives into TTS otherwise.
    text = _TIME_RE.sub(_time_to_russian, text)
    # Collapse space/NBSP thousands separators between 3-digit groups so "1 000 000"
    # is voiced as one number ("один миллион"), not split into "один ноль ноль".
    # Requires at least one full 3-digit group, so unrelated adjacent numbers
    # ("в 5 7 раз") are left untouched.
    text = _THOUSANDS_RE.sub(
        lambda m: re.sub(r"[   ]", "", m.group(0)), text)
    text = _ONE_TWO_RE.sub(_agree_one_two, text)
    text = _NUMBER_RE.sub(_number_to_russian_words, text)
    text = spell_cyrillic_acronyms(text)
    # Voice a trailing percent sign as a word (e.g. "18%" -> "восемнадцать процентов").
    text = re.sub(r"\s*%", " процентов", text)
    return text


# ---------------------------------------------------------------------------
# Latin words in Russian speech
# ---------------------------------------------------------------------------
# vocab.txt carries 82 Latin tokens because the F5 base vocabulary is
# multilingual, so a word like "Arduino" tokenises cleanly and nothing warns.
# The RUSSIAN fine-tune (model_212000) never heard those tokens against Russian
# audio, so their pronunciation is undefined -- it emits a noise burst. Whisper
# transcribed one such reply as "разбираюсь в 1-0 с написанием кода", which is
# how this was found: the voice had not broken, it had been handed letters it
# cannot say.
#
# Spelling the word in Cyrillic is not a translation and is not meant to be
# perfect English. It only has to land inside the sound inventory the voice was
# trained on, which turns an unpronounceable burst into a recognisable word.
_TRANSLIT_WORDS = {
    "arduino": "ардуино", "python": "пайтон", "javascript": "джаваскрипт",
    "java": "джава", "linux": "линукс", "windows": "виндоус",
    "android": "андроид", "google": "гугл", "youtube": "ютуб",
    "telegram": "телеграм", "github": "гитхаб", "docker": "докер",
    "raspberry": "распберри", "esp": "и-эс-пи", "bluetooth": "блютус",
    "wifi": "вайфай", "usb": "ю-эс-би", "code": "код", "open": "оупен",
    "source": "сорс", "server": "сервер", "online": "онлайн",
    "email": "имейл", "internet": "интернет", "computer": "компьютер",
    "hello": "хэлло", "ok": "окей", "okay": "окей",
}

# Letter names, for an acronym that is read out rather than pronounced.
_LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф",
    "g": "джи", "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл",
    "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл-ю", "x": "икс",
    "y": "уай", "z": "зед",
}

# Longest first: "sh" must win over "s", or "ship" comes out "с-х-ип".
_DIGRAPHS = [
    ("sch", "ш"), ("tch", "ч"), ("sh", "ш"), ("ch", "ч"), ("ph", "ф"),
    ("th", "т"), ("ck", "к"), ("qu", "кв"), ("oo", "у"), ("ee", "и"),
    ("ea", "и"), ("ou", "ау"), ("ay", "эй"), ("ai", "эй"), ("oy", "ой"),
    ("ya", "я"), ("yu", "ю"), ("ju", "джу"), ("ja", "джа"),
]
_SINGLES = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}

_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _spell_out(word: str) -> str:
    return "-".join(_LETTER_NAMES.get(ch, ch) for ch in word.lower())


def _translit_word(word: str) -> str:
    low = word.lower()
    known = _TRANSLIT_WORDS.get(low)
    if known:
        return known
    # An all-caps run of 2-5 letters is an acronym (API, HTTP, USB): those are
    # read letter by letter, not pronounced, and "апи" would be wrong.
    if 2 <= len(word) <= 5 and word.isupper():
        return _spell_out(word)
    if len(word) == 1:
        return _LETTER_NAMES.get(low, low)
    out, i = [], 0
    while i < len(low):
        for src, dst in _DIGRAPHS:
            if low.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(_SINGLES.get(low[i], ""))
            i += 1
    return "".join(out)


_DIGIT_RUN = re.compile(r"\d+")


def _digits_to_words(text: str) -> str:
    """Spell numbers out in Russian for the fallback path.

    The LLM pass handles this and much more, but when it is unavailable the
    digits would otherwise reach the voice as digits -- the same class of
    unpronounceable token as the Latin, and the reason this fallback exists.
    """
    try:
        from num2words import num2words
    except Exception:
        return text

    def one(m):
        try:
            return num2words(int(m.group(0)), lang="ru")
        except Exception:
            return m.group(0)

    # "ESP32" becomes "и-эс-пи" + "тридцать два": without a separator those
    # run together into one unreadable word.
    text = re.sub(r"(?<=[^\W\d_])(?=\d)", " ", text)
    return _DIGIT_RUN.sub(one, _ONE_TWO_RE.sub(_agree_one_two, text))


def latin_to_cyrillic(text: str) -> str:
    """Respell Latin-script words so the Russian voice can pronounce them.

    Applied BEFORE the accentor: RUAccent has nothing to say about Latin, and a
    word respelled afterwards would reach synthesis unstressed.
    """
    if not text:
        return text
    # "Wi-Fi" is one word spoken; split on the hyphen it became "ви-фи".
    text = re.sub(r"\b[Ww]i-?[Ff]i\b", "wifi", text)
    out = _digits_to_words(_LATIN_RUN.sub(
        lambda m: _translit_word(m.group(0)), text))
    # "GPT-4" -> "джи-пи-ти" + "-" + "-четыре": one hyphen is enough.
    return re.sub(r"(?<=\w)-{2,}(?=\w)", "-", out)


_UNSPEAKABLE = re.compile(r"[A-Za-z0-9]")

_SPOKEN_SYSTEM = (
    "Ты готовишь русский текст к озвучке. Перепиши его так, чтобы КАЖДОЕ слово "
    "читалось по-русски вслух: латиницу запиши русскими буквами по звучанию "
    "(Arduino -> ардуино, USB -> ю-эс-би), числа и даты — словами "
    "(2024 -> две тысячи двадцать четвёртый, 15% -> пятнадцать процентов), "
    "единицы и символы — словами. Смысл, порядок слов и пунктуацию не меняй, "
    "ничего не добавляй и не убирай. Верни ТОЛЬКО переписанный текст."
)

# The rewrite is deterministic for a given string and the same replies recur
# (greetings, tool names), so a small cache keeps the cost near zero.
_spoken_cache: dict = {}
_spoken_cache_lock = threading.Lock()
_SPOKEN_CACHE_MAX = 512


def needs_spoken_form(text: str) -> bool:
    """Does anything here have no Russian pronunciation? Latin or digits."""
    return bool(_UNSPEAKABLE.search(text or ""))


def _spoken_form_is_sane(src: str, out: str) -> bool:
    """Accept the rewrite only if it did the job and nothing else.

    Two failures are worth catching: a model that answers ABOUT the text instead
    of rewriting it (length runs away), and one that leaves the Latin in place,
    which would send the same unpronounceable tokens to the voice while making
    us believe they were handled.
    """
    if not out or _UNSPEAKABLE.search(out):
        return False
    n, m = len(src), len(out)
    return 0.4 * n <= m <= 3.0 * n + 40


def to_spoken_form(ctx, text: str) -> str:
    """Rewrite anything unpronounceable into Russian words, via the LLM.

    The model is already loaded and this is a few dozen tokens, so it costs
    little; the rule table below is the fallback for when it is not available
    (no ctx, LLM down, or a reply that fails the sanity check).
    """
    if not needs_spoken_form(text):
        return text
    key = text.strip()
    with _spoken_cache_lock:
        hit = _spoken_cache.get(key)
    if hit is not None:
        return hit

    out = None
    try:
        if ctx is not None and (getattr(ctx, "model_name", "") or "").strip():
            import llm
            reply = llm.call_llm_simple(
                ctx, _SPOKEN_SYSTEM, text, temperature=0.0,
                max_tokens=max(64, len(text) * 2))
            reply = (reply or "").strip().strip('"')
            if _spoken_form_is_sane(text, reply):
                out = reply
            elif reply:
                logger.warning("spoken-form rewrite rejected (%d -> %d chars, "
                               "latin/digits left: %s)", len(text), len(reply),
                               bool(_UNSPEAKABLE.search(reply)))
    except Exception:
        logger.exception("spoken-form rewrite failed — falling back to the table")

    if out is None:
        out = latin_to_cyrillic(text)
    with _spoken_cache_lock:
        if len(_spoken_cache) < _SPOKEN_CACHE_MAX:
            _spoken_cache[key] = out
    return out


def preprocess_text_for_synthesis(
        ctx,
        raw_text: str,
        use_censoring: bool = True,
        apply_stress: bool = True,
) -> str:
    text = clean_text(raw_text)
    if not text:
        return ""
    if use_censoring:
        text = censor_profanity(text)
    if text.isupper():
        text = text.capitalize()
    text = spell_cyrillic_acronyms(text)
    text = to_spoken_form(ctx, text)
    if apply_stress:
        text = stress_plus(ctx, text)
    # Collapse accidental double markers, then KEEP a single '+' before the stressed
    # vowel: the F5 vocab contains '+' (verified) and the Russian model was trained
    # with it, so the stress must reach synthesis. (Previously every '+' was stripped
    # here, which silently discarded all accentuation before the TTS ever saw it.)
    text = re.sub(r"\+\+", "+", text)
    # Drop a stray '+' that is NOT immediately before a vowel (e.g. from "C++"), so
    # only real stress markers survive.
    text = re.sub(r"\+(?![аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouyAEIOUY])", "", text)
    # === THE ENDING FIX -- DO NOT REMOVE (see config.TTS_END_PADDING) ==========
    # This fine-tune eats the end of the phrase; a run of trailing periods is
    # the one thing that stops it, because F5 sizes the output from gen_text's
    # byte length. Appended here and ONLY here, so every surface -- the desktop
    # app, Telegram and Skyrim -- gets it exactly once by going through this
    # function. rstrip() first so re-synthesising already-padded text cannot
    # stack two runs. Guarded by tests/test_tts_end_padding.py.
    padding = TTS_END_PADDING or ("." * 15)
    text = text.rstrip(". \t\r\n") + padding
    return text


def get_actor_ref_and_speed(actor_name: str) -> Tuple[Optional[str], str, float]:
    actors = {
        "DC": (str(DC_REF_WAV), "", 1.0),
    }
    if actor_name not in actors:
        return None, "", DEFAULT_ACTOR_SPEED
    ref_wav, ref_text, speed = actors[actor_name]
    return ref_wav, ref_text or "", float(speed)


_ref_wav_cache: dict = {}  # (src_path, mtime) -> converted temp wav path


def resolve_ref_audio(path: Optional[str]) -> Optional[str]:
    """Return a WAV path usable as a TTS reference, converting from any audio
    format (mp3/ogg/m4a/flac/...) if needed. Format-insensitive; results cached.
    """
    if not path or not os.path.exists(path):
        return None
    if path.lower().endswith(".wav"):
        return path
    key = (path, os.path.getmtime(path))
    cached = _ref_wav_cache.get(key)
    if cached and os.path.exists(cached):
        return cached
    try:
        seg = AudioSegment.from_file(path).set_channels(1)  # any format via ffmpeg
        # Deterministic name (path+mtime) so re-runs reuse the converted file
        # instead of orphaning a new one each process (hash() is salted per-run).
        digest = hashlib.md5(f"{path}|{key[1]}".encode("utf-8")).hexdigest()[:12]
        tmp = OUTPUT_DIR / f"_ref_{digest}.wav"
        seg.export(tmp, format="wav")
        _ref_wav_cache[key] = str(tmp)
        logger.info("Converted reference voice %s -> %s", os.path.basename(path), tmp.name)
        return str(tmp)
    except Exception as exc:
        logger.error("Could not load reference audio %s: %s", path, exc)
        return None


def _gain_linear(gain_db: float) -> float:
    """dB -> linear multiplier, skipping the pow() call for the common 0 dB case.
    Shared by every mic input path (MicRecorder, VadListener,
    record_audio_until_enter) so the dB convention can't drift between them."""
    return 10 ** (gain_db / 20.0) if gain_db != 0.0 else 1.0


def _chunk_rms(chunk: np.ndarray) -> float:
    """RMS loudness of one audio chunk, for the GUI mic-level visualizer. Never
    raises (returns 0.0) so a bad chunk can't kill the input callback."""
    try:
        return float(np.sqrt(np.mean(np.square(chunk))))
    except Exception:
        return 0.0


class MicRecorder:
    """Non-blocking microphone recorder: start(), then stop() -> np.ndarray.

    Unlike record_audio_until_enter, it does not block on console input, so a UI
    loop (e.g. camera mode) can start/stop it on key presses while staying live.
    """

    def __init__(self, fs: int = SAMPLE_RATE, gain_db: float = 0.0):
        self.fs = fs
        self._gain = _gain_linear(gain_db)
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        # Live input loudness (RMS, 0..~1) of the most recent chunk, for the GUI
        # mic visualizer. Plain float write/read — GIL-atomic, no lock needed.
        self._level: float = 0.0

    def level(self) -> float:
        """Current microphone loudness (RMS of the last chunk, post-gain). 0 when idle."""
        return self._level

    def start(self) -> None:
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("%s", status)
            chunk = indata.copy()
            if self._gain != 1.0:
                chunk = np.clip(chunk * self._gain, -1.0, 1.0)
            self._level = _chunk_rms(chunk)
            self._q.put(chunk)

        self._stream = sd.InputStream(samplerate=self.fs, channels=1, dtype="float32", callback=callback)
        self._stream.start()

    def stop(self) -> np.ndarray:
        self._level = 0.0
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get())
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks, axis=0).flatten().astype(np.float32)


# --- Hands-free VAD listening (GUI auto-voice mode) --------------------------

_SILERO_LOCK = threading.Lock()
_SILERO_MODEL = None


def _load_silero_vad_model():
    """Lazy singleton for the Silero VAD model (tiny, runs on CPU)."""
    global _SILERO_MODEL
    with _SILERO_LOCK:
        if _SILERO_MODEL is None:
            from silero_vad import load_silero_vad
            _SILERO_MODEL = load_silero_vad()
            logger.info("Silero VAD model loaded")
        return _SILERO_MODEL


class VadListener:
    """Hands-free voice listener: continuously reads the microphone, detects
    utterance boundaries with Silero VAD, and hands each finished utterance
    (float32 mono @ 16 kHz, with a little pre-roll) to ``on_utterance`` from a
    worker thread.

    ``set_paused(True)`` makes it deaf — incoming audio is discarded and the
    detection state resets — so the GUI mutes it while a request is running or
    the assistant's own voice is playing; otherwise the assistant would hear
    and answer itself.
    """

    FRAME = 512  # samples @ 16 kHz = 32 ms — the window Silero VAD requires

    def __init__(self, on_utterance, fs: int = SAMPLE_RATE, gain_db: float = 0.0, device=None):
        from config import (VAD_THRESHOLD, VAD_END_SILENCE_S, VAD_MIN_SPEECH_S,
                            VAD_PRE_ROLL_S, VAD_MAX_UTTERANCE_S)
        if fs != 16000:
            raise ValueError("VadListener requires 16 kHz audio (Silero VAD constraint)")
        self.on_utterance = on_utterance
        self.fs = fs
        self._gain = _gain_linear(gain_db)
        self.threshold = VAD_THRESHOLD
        self.end_silence_frames = max(1, int(VAD_END_SILENCE_S * fs / self.FRAME))
        self.min_speech_frames = max(1, int(VAD_MIN_SPEECH_S * fs / self.FRAME))
        self.pre_roll_frames = max(1, int(VAD_PRE_ROLL_S * fs / self.FRAME))
        self.max_utterance_frames = max(1, int(VAD_MAX_UTTERANCE_S * fs / self.FRAME))
        self._q: queue.Queue = queue.Queue()
        self._paused = threading.Event()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._model = None
        from collections import deque
        self._pre_roll = deque(maxlen=self.pre_roll_frames)
        self._speech: list = []
        self._silence_run = 0
        self._voiced = 0
        self._speaking = False
        self._device = device
        self._level = 0.0  # live mic loudness for the GUI visualizer (0 when paused)

    def level(self) -> float:
        """Current microphone loudness (RMS, post-gain); 0 while paused/idle."""
        return self._level

    def start(self) -> None:
        self._model = _load_silero_vad_model()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("VAD mic: %s", status)
            if not self._paused.is_set() and not self._stop_evt.is_set():
                chunk = indata.copy()
                if self._gain != 1.0:
                    chunk = np.clip(chunk * self._gain, -1.0, 1.0)
                self._level = _chunk_rms(chunk)
                self._q.put(chunk)
            else:
                self._level = 0.0

        self._stream = sd.InputStream(samplerate=self.fs, channels=1, dtype="float32",
                                      blocksize=self.FRAME, callback=callback,
                                      device=self._device)
        self._stream.start()
        self._thread = threading.Thread(target=self._run, name="vad-listener", daemon=True)
        self._thread.start()
        logger.info("VAD listener started")

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def is_healthy(self) -> bool:
        """False when the input stream died under us (mic unplugged / device
        switched): PortAudio marks the stream inactive and the callback stops —
        without this check VAD sits silently deaf with the button still ON."""
        stream = self._stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:
            return False

    def stop(self) -> None:
        self._stop_evt.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("VAD listener stopped")

    # -- detection worker ------------------------------------------------------

    def _reset_state(self) -> None:
        self._speech = []
        self._silence_run = 0
        self._voiced = 0
        self._speaking = False
        self._pre_roll.clear()
        try:
            self._model.reset_states()  # Silero keeps LSTM state between calls
        except Exception:
            pass

    def _run(self) -> None:
        import torch
        self._reset_state()
        while not self._stop_evt.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._paused.is_set():
                # Drain audio queued just before the pause and reset, so a
                # half-captured utterance can't leak into the next turn.
                self._reset_state()
                continue
            frame = chunk.flatten()
            if len(frame) != self.FRAME:  # partial buffer while closing
                continue
            try:
                with torch.no_grad():
                    prob = float(self._model(torch.from_numpy(frame).unsqueeze(0), self.fs).item())
            except Exception as exc:
                logger.error("VAD inference failed: %s", exc)
                sd.sleep(500)
                continue
            if not self._speaking:
                self._pre_roll.append(frame)
                if prob >= self.threshold:
                    self._speaking = True
                    self._speech = list(self._pre_roll)
                    self._pre_roll.clear()
                    self._silence_run = 0
                    self._voiced = 1
            else:
                self._speech.append(frame)
                if prob >= self.threshold:
                    self._voiced += 1
                    self._silence_run = 0
                else:
                    self._silence_run += 1
                if (self._silence_run >= self.end_silence_frames
                        or len(self._speech) >= self.max_utterance_frames):
                    self._finish_utterance()

    def _finish_utterance(self) -> None:
        speech, voiced = self._speech, self._voiced
        self._reset_state()
        if voiced < self.min_speech_frames:
            logger.info("VAD: discarded a short burst (%.2f s of voice)",
                        voiced * self.FRAME / self.fs)
            return
        utterance = np.concatenate(speech).astype(np.float32)
        logger.info("VAD: utterance captured (%.1f s)", len(utterance) / self.fs)
        try:
            self.on_utterance(utterance)
        except Exception:
            logger.exception("VAD on_utterance callback failed")


def safe_normalize_segment(seg: AudioSegment, target_dBFS: float = TARGET_DBFS) -> AudioSegment:
    try:
        if seg.dBFS == float("-inf") or seg.dBFS is None:
            return seg
        return seg.apply_gain(target_dBFS - seg.dBFS)
    except Exception:
        return seg


def record_audio_until_enter(fs: int = SAMPLE_RATE, gain_db: float = 0.0) -> np.ndarray:
    """Record microphone input between two ENTER presses.

    Args:
        fs: Sample rate in Hz.
        gain_db: Optional pre-amplification in dB (0 = no gain).
    """
    logger.info("Press ENTER to START recording")
    try:
        input()
    except EOFError:
        return np.array([], dtype=np.float32)

    gain_linear = _gain_linear(gain_db)
    q: queue.Queue[np.ndarray] = queue.Queue()
    recording: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            logger.warning("%s", status)
        if not stop_event.is_set():
            chunk = indata.copy()
            if gain_linear != 1.0:
                chunk = np.clip(chunk * gain_linear, -1.0, 1.0)
            q.put(chunk)

    stream = sd.InputStream(samplerate=fs, channels=1, dtype="float32", callback=callback)
    try:
        stream.start()
        logger.info("Recording started... press ENTER to stop")

        def wait_stop():
            try:
                input()
            except EOFError:
                pass
            stop_event.set()
            try:
                stream.stop()
            except Exception:
                pass

        stopper = threading.Thread(target=wait_stop, daemon=True)
        stopper.start()

        while stream.active or not q.empty():
            try:
                recording.append(q.get(timeout=0.1))
            except queue.Empty:
                if stop_event.is_set() and not stream.active:
                    break

        stopper.join(timeout=1.0)
    finally:
        try:
            stream.close()
        except Exception:
            pass

    if not recording:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(recording, axis=0).flatten().astype(np.float32)
    if gain_db != 0.0:
        energy = np.sqrt(np.mean(audio ** 2))
        logger.info("Audio energy level: %.6f", energy)
        if energy < 0.01:
            logger.warning("Very low audio energy - check microphone")
    return audio


def _log_detected_language(info) -> None:
    """Surface Whisper's auto-detected language (only meaningful when not forced)."""
    if WHISPER_LANGUAGE is not None:
        return
    lang = getattr(info, "language", None)
    if lang:
        prob = getattr(info, "language_probability", 0.0) or 0.0
        logger.info("Detected language: %s (%.0f%%)", lang, prob * 100)


_giga_model = None
_giga_lock = threading.Lock()


def _use_gigaam() -> bool:
    """Is the Russian-only engine safe to use for this installation?

    Only when the language is pinned to Russian. GigaAM does not detect or
    refuse another language -- it renders it as confident Russian nonsense --
    and this same function transcribes the F5 reference clips, which are
    English. An unset WHISPER_LANGUAGE means "detect it", and only Whisper can.
    """
    mode = str(getattr(config, "ASR_ENGINE", "auto") or "auto").lower()
    if mode == "whisper":
        return False
    if mode == "gigaam":
        return True
    return (WHISPER_LANGUAGE or "").lower().startswith("ru")


def _gigaam():
    global _giga_model
    if _giga_model is None:
        with _giga_lock:
            if _giga_model is None:
                import onnx_asr
                name = getattr(config, "GIGAAM_MODEL", "gigaam-v3-e2e-rnnt")
                logger.info("Loading %s (CPU, no VRAM)...", name)
                _giga_model = onnx_asr.load_model(name)
    return _giga_model


def _as_float32_16k(source):
    """ndarray or path -> mono float32 at 16 kHz, or None if it cannot be read.

    float32 is not a preference: GigaAM accepts int16 without complaint and
    returns nonsense for it.
    """
    try:
        if isinstance(source, np.ndarray):
            x, sr = source, 16000          # the mic path already records at 16k
        else:
            x, sr = sf.read(str(source), dtype="float32", always_2d=False)
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(axis=1)
        x = np.asarray(x, dtype=np.float32)
        if sr != 16000 and len(x):
            n = int(round(len(x) * 16000.0 / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n),
                          np.arange(len(x)), x).astype(np.float32)
        return x
    except Exception:
        logger.exception("could not read audio for GigaAM")
        return None


def _split_at_pauses(x, sr: int = 16000, lo_s: float = 12.0, hi_s: float = 22.0,
                     win_s: float = 0.05) -> list:
    """Split audio into pieces of lo_s..hi_s seconds, each cut placed at the
    lowest-energy win_s window in that span (a pause between words)."""
    pieces, start, n = [], 0, len(x)
    win = max(1, int(win_s * sr))
    while n - start > hi_s * sr:
        a, b = start + int(lo_s * sr), start + int(hi_s * sr)
        seg = x[a:b]
        k = len(seg) // win
        energy = np.square(seg[:k * win]).reshape(k, win).mean(axis=1)
        cut = a + int(np.argmin(energy)) * win + win // 2
        pieces.append(x[start:cut])
        start = cut
    pieces.append(x[start:])
    return pieces


def _gigaam_transcribe(ctx, audio_source) -> str:
    """GigaAM, chunked. Returns "" when it cannot, so the caller can fall back.

    Whisper's vad_filter returned nothing for a silent take; GigaAM has no such
    filter and will decode room tone into words, so silence is rejected here.
    """
    x = _as_float32_16k(audio_source)
    if x is None or not len(x):
        return ""
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms < 0.004:
        return ""
    model = _gigaam()
    with ctx.asr_lock:
        if len(x) <= 25 * 16000:
            return (model.recognize(x, sample_rate=16000) or "").strip()
        # 20-30 s is the architecture's limit; longer audio is split.
        # Cut at the quietest point, not every 20 s blind: a cut through a word
        # loses it on both sides (~6 pp WER on long Russian audio).
        out = []
        for chunk in _split_at_pauses(x):
            if len(chunk) < 1600:
                continue
            out.append((model.recognize(chunk, sample_rate=16000) or "").strip())
    return " ".join(p for p in out if p).strip()


# Below this Whisper's guess about the language is a coin toss: a one-word
# "Угу." was heard as French (69%) and came back as a goodbye the bot then
# answered (live, 2026-09-12, journey 22). Under it, the clip is re-read in
# the language the user is known to speak.
ASR_LANG_CONFIDENCE = _cfg_env.env_float("ASR_LANG_CONFIDENCE", 0.85)


def _whisper_transcribe(ctx, audio_source, engine: str = "auto",
                        lang_hint: str = "") -> str:
    """Run faster-whisper on ``audio_source`` (an ndarray or a file path) under
    the shared ASR lock and join the segments into one string. Shared by the
    array and file entry points below so the transcribe call/args can't drift
    between them."""
    if engine != "whisper" and _use_gigaam():
        try:
            text = _gigaam_transcribe(ctx, audio_source)
            if text:
                return text
            logger.info("GigaAM returned nothing - falling through to Whisper")
        except Exception:
            logger.exception("GigaAM failed - falling back to Whisper")
    with ctx.asr_lock:
        segments, info = ctx.models.whisper.transcribe(
            audio_source, language=WHISPER_LANGUAGE, beam_size=5, vad_filter=True,
        )
        text = " ".join(seg.text for seg in segments).strip()
    _log_detected_language(info)
    if _doubtful_language(info, lang_hint):
        logger.info("ASR: language guess %s (%.0f%%) is doubtful — re-reading as %s",
                    getattr(info, "language", "?"),
                    (getattr(info, "language_probability", 0.0) or 0.0) * 100, lang_hint)
        with ctx.asr_lock:
            segments, _ = ctx.models.whisper.transcribe(
                audio_source, language=lang_hint, beam_size=5, vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()
    return text


def detect_media_language(ctx, path: str, min_prob: float = 0.5) -> str:
    """Spoken language of a whole media file ("" when unsure or on failure).

    A video's language is its content's, not the chat's: passing the session
    language as the hint re-read an English clip as Russian whenever Whisper was
    below ASR_LANG_CONFIDENCE. Detected ONCE over several windows of the whole
    file, so short diarized turns and video portions inherit a stable answer."""
    if WHISPER_LANGUAGE:
        return WHISPER_LANGUAGE
    try:
        from faster_whisper import decode_audio
        wav = decode_audio(path, sampling_rate=16000)
        with ctx.asr_lock:
            lang, prob, _ = ctx.models.whisper.detect_language(
                wav, vad_filter=True, language_detection_segments=3)
        logger.info("ASR: media language %s (%.0f%%)", lang, prob * 100)
        return lang if prob >= min_prob else ""
    except Exception:
        logger.warning("media language detection failed", exc_info=True)
        return ""


def _doubtful_language(info, lang_hint: str) -> bool:
    """True when auto-detection was unsure AND disagrees with the language the
    user is known to speak. A forced WHISPER_LANGUAGE never gets here."""
    if WHISPER_LANGUAGE or not lang_hint:
        return False
    lang = getattr(info, "language", None)
    prob = getattr(info, "language_probability", None)
    if lang is None or prob is None:
        return False
    return lang != lang_hint and prob < ASR_LANG_CONFIDENCE


# A voice note that opens with a first-person future verb — «Нарисую синего
# слона», «Сделаю слона розовым» — is the ASR mishearing the imperative
# («Нарисуй», «Сделай»): nobody tells the bot what THEY are about to draw.
# The model took the statement at face value and replied with filler about
# the session memory, no picture (live 2026-09-13, mega run 3, step 25).
# Only the first word of the transcript is touched.
_VOICE_IMPERATIVE = {
    "нарисую": "нарисуй", "сделаю": "сделай", "напишу": "напиши", "найду": "найди",
    "покажу": "покажи", "расскажу": "расскажи", "переведу": "переведи",
    "сочиню": "сочини", "запомню": "запомни", "скажу": "скажи",
    "посчитаю": "посчитай", "объясню": "объясни", "придумаю": "придумай",
    "составлю": "составь", "отправлю": "отправь", "проверю": "проверь",
    "исправлю": "исправь", "поменяю": "поменяй", "уберу": "убери",
    "добавлю": "добавь", "озвучу": "озвучь", "спою": "спой", "повторю": "повтори",
    "прочитаю": "прочитай", "перескажу": "перескажи", "сокращу": "сократи",
    "напомню": "напомни", "включу": "включи", "выключу": "выключи",
    "продолжу": "продолжи", "открою": "открой", "закрою": "закрой",
}
_VOICE_FIRST_WORD_RE = re.compile(r"^(\s*)([А-Яа-яЁё]+)")


def fix_voice_imperative(text: str) -> str:
    """«Нарисую синего слона» -> «Нарисуй синего слона» at the start of a voice note."""
    m = _VOICE_FIRST_WORD_RE.match(text or "")
    if not m:
        return text
    word = m.group(2)
    fixed = _VOICE_IMPERATIVE.get(word.lower())
    if not fixed:
        return text
    if word[:1].isupper():
        fixed = fixed[:1].upper() + fixed[1:]
    return text[:m.start(2)] + fixed + text[m.end(2):]


def transcribe_audio_array(ctx, audio_data: np.ndarray) -> str:
    logger.info("Recognizing speech...")
    try:
        text = _whisper_transcribe(ctx, audio_data)
        if text:
            logger.info("Recognized: %s", text)
        return text
    except Exception as exc:
        logger.error("ASR array error: %s", exc)
        return ""
    finally:
        safe_empty_cuda_cache()


def transcribe_audio_file(ctx, audio_path: str, engine: str = "auto",
                          lang_hint: str = "") -> str:
    """Transcribe a file. Pass engine="whisper" when the language is NOT known
    to be Russian -- a TTS reference clip, most of all: the Mantella speaker
    library is read in English, and the Russian-only engine would render it as
    confident Russian nonsense with nothing to signal that it had."""
    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        return ""

    # The engine is part of the key: two engines transcribe the same file
    # differently, and a cache shared between them would serve one model's
    # words as the other's.
    cache_key = "%s_%s" % ("whisper" if engine == "whisper" else "asr",
                           audio_hash_from_path(audio_path))
    cached = ctx.transcription_cache.get(cache_key)
    if isinstance(cached, str):
        safe_empty_cuda_cache()
        return cached

    try:
        text = _whisper_transcribe(ctx, audio_path, engine=engine, lang_hint=lang_hint)
        ctx.transcription_cache[cache_key] = text
        ctx.save_cache()
        return text
    except Exception as exc:
        logger.error("ASR file error: %s", exc)
        return ""
    finally:
        safe_empty_cuda_cache()


# F5 clips any reference longer than this and says so in a print, then carries
# on. utils_infer.py:324 — the number is theirs, repeated here because the
# consequence is ours to avoid.
REF_CAP_MS = 12_000


def trim_ref_to_cap(ref_wav: str) -> str:
    """Cut a reference down to what F5 will actually listen to.

    F5 clips a long reference to 12 s but is handed the transcript of the WHOLE
    file. The model then still has text left to say when the reference audio
    ends, and finishes it — the tail of the reference is spoken INTO the reply.
    A 20 s reference here put "это довольно длинное сообщение" into a sentence
    about roubles and minutes, which reads as "the voice went bad" rather than
    as a mismatch.

    Trimming on our side makes the transcript describe exactly the audio F5
    hears, and its own clipping a no-op. Cut on silence, like F5 does, so the
    reference never ends mid-word. The result is cached beside the original.
    """
    try:
        from pydub import silence as _silence
        seg = AudioSegment.from_file(ref_wav)
        if len(seg) <= REF_CAP_MS:
            return ref_wav
        cached = Path(ref_wav).with_name(Path(ref_wav).stem + "_ref12.wav")
        if cached.is_file():
            try:
                if len(AudioSegment.from_file(cached)) <= REF_CAP_MS:
                    return str(cached)
            except Exception:
                pass
        chunks = _silence.split_on_silence(
            seg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000,
            seek_step=10)
        out = AudioSegment.silent(duration=0)
        for ch in (chunks or [seg]):
            if len(out) > 6000 and len(out) + len(ch) > REF_CAP_MS:
                break
            out += ch
        if len(out) < 1000:                    # nothing usable: hard cut
            out = seg[:REF_CAP_MS]
        if len(out) > REF_CAP_MS:
            out = out[:REF_CAP_MS]
        out.export(str(cached), format="wav")
        logger.info("reference trimmed for F5: %s (%.1fs) -> %s (%.1fs)",
                    Path(ref_wav).name, len(seg) / 1000.0,
                    cached.name, len(out) / 1000.0)
        return str(cached)
    except Exception:
        logger.exception("could not trim the reference — using it as it is")
        return ref_wav


def synth_single_segment(
        ctx,
        idx: int,
        actor: str,
        raw_text: str,
        out_stem: Optional[str] = None,
        use_censoring: bool = True,
        apply_stress: bool = True,
) -> Optional[str]:
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    try:
        processed_text = preprocess_text_for_synthesis(
            ctx, raw_text, use_censoring=use_censoring, apply_stress=apply_stress,
        )
        if not processed_text:
            return None

        ref_wav, ref_text, actor_speed = get_actor_ref_and_speed(actor)
        custom = getattr(ctx, "custom_ref_wav", None)
        if custom:
            # A caller that already knows the transcript says so. The
            # Mantella server prepares each of ~150 voices once and keeps
            # the text; re-deriving it per line was a second a line.
            ref_wav = custom
            ref_text = str(getattr(ctx, "custom_ref_text", "") or "")
        ref_wav = resolve_ref_audio(ref_wav)  # accept any audio format
        if ref_wav is not None:
            # BEFORE the transcript is taken, or the transcript describes audio
            # F5 will not hear.
            ref_wav = trim_ref_to_cap(ref_wav)
        if ref_wav is None:
            logger.error("Reference audio not found/usable for actor %s", actor)
            return None

        # Transcribe the reference OURSELVES when no transcript is configured.
        #
        # Left empty, f5_tts falls back to its own ASR inside
        # preprocess_ref_audio_text, which decodes audio through torchaudio ->
        # torchcodec. torchcodec is version-locked to both torch AND the
        # installed FFmpeg, so a mismatch there raises and takes the whole
        # synthesis down — every reply loses its voice while the text still
        # arrives, which is why this reads as "TTS randomly stopped working"
        # rather than as a crash. Our faster-whisper path decodes via PyAV and
        # has no such coupling, and its result is cached in transcriptions.json,
        # so this also stops re-transcribing the same reference on every turn.
        if not ref_text:
            ref_text = transcribe_audio_file(ctx, ref_wav, engine="whisper") or ""
        ref_file, ref_text_proc = preprocess_ref_audio_text(ref_wav, ref_text)

        with ctx.tts_lock:
            # 16, not 24: f5_tts.model.cfm's EPSS table (Empirically Pruned Step
            # Sampling) only has hand-tuned non-uniform timesteps for
            # {5,6,7,10,12,16} -- 24 isn't one of them, so it silently fell back
            # to a naive linspace(0,1,25), the one schedule shape the model was
            # never tuned for. 16 IS in the table and is the official F5 "fast"
            # point (~2x speedup, WER 2.42%->2.53% on the upstream benchmark).
            # A/B'd 2026-09-19 on our own RU checkpoint: 1.95s -> 0.82s per
            # segment, samples sent to Telegram for a listen before switching.
            wav, sr, _ = infer_process(
                ref_file, ref_text_proc, processed_text,
                ctx.models.tts_model, ctx.models.vocoder,
                cross_fade_duration=0.1, nfe_step=config.TTS_NFE_STEP,
                speed=actor_speed, device=config.DEVICE,
            )

        if wav is None or not isinstance(wav, np.ndarray) or wav.size == 0:
            return None

        if out_stem is None:
            out_wav = OUTPUT_DIR / f"seg_{idx}_{actor}.wav"
        else:
            out_wav = Path(out_stem + f"_{actor}_{idx}.wav") if not out_stem.endswith(".wav") else Path(out_stem)

        sf.write(out_wav, wav, sr)
        seg = AudioSegment.from_wav(out_wav)
        seg = safe_normalize_segment(seg, TARGET_DBFS)
        if TTS_TAIL_SILENCE_MS > 0:
            seg = seg + AudioSegment.silent(duration=TTS_TAIL_SILENCE_MS)  # guard last word vs playback truncation
        seg.export(out_wav, format="wav")
        return str(out_wav)
    except Exception as exc:
        logger.error("TTS error: %s", exc)
        return None
    finally:
        safe_empty_cuda_cache()


def play_audio_file(result_path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(result_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", result_path])
        else:
            subprocess.Popen(["xdg-open", result_path])
        import time
        time.sleep(0.5)
    except Exception as exc:
        logger.warning("Could not open audio file: %s", exc)


class AudioPlayer:
    """Pausable WAV playback via a sounddevice output stream.

    play(path) -> toggle_pause() -> stop(). Used by the GUI so the assistant's
    spoken reply can be paused and resumed mid-sentence.
    """

    def __init__(self):
        self._stream = None
        self._data = None
        self._pos = 0
        self._paused = False
        self._lock = threading.Lock()

    def play(self, path: str) -> None:
        self.stop()
        try:
            data, sr = sf.read(path, dtype="float32")
        except Exception as exc:
            logger.warning("Playback load failed: %s", exc)
            return
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        channels = data.shape[1]

        def callback(outdata, frames, time_info, status):
            with self._lock:
                data_ref = self._data
                if self._paused or data_ref is None:
                    outdata[:] = 0
                    return
                chunk = data_ref[self._pos:self._pos + frames]
                self._pos += len(chunk)
            n = len(chunk)
            outdata[:n] = chunk
            if n < frames:
                outdata[n:] = 0
                raise sd.CallbackStop()

        with self._lock:
            self._data = data
            self._pos = 0
            self._paused = False
        try:
            self._stream = sd.OutputStream(
                samplerate=int(sr), channels=channels, dtype="float32",
                callback=callback, finished_callback=self._on_finished,
            )
            self._stream.start()
        except Exception as exc:
            logger.warning("Playback start failed: %s", exc)
            self._stream = None

    def _on_finished(self):
        with self._lock:
            self._data = None
            self._pos = 0
            self._paused = False

    def toggle_pause(self) -> bool:
        """Flip pause state. Returns True if now paused, False otherwise."""
        with self._lock:
            if self._stream is None or self._data is None:
                return False
            self._paused = not self._paused
            return self._paused

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self._lock:
            self._data = None
            self._pos = 0
            self._paused = False

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._stream is not None and self._data is not None
