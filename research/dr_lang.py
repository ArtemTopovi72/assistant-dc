"""Output-language selection for research reports.

Which language the finished document is written in, and the fixed scaffolding
(abstract/TOC/appendix headings) that the assembler — not the model — emits.

`lang_of_text` sniffs the language of a topic or a user turn, `_norm_out_lang`
clamps it to a supported code, `_lab` looks up a scaffolding label and
`_lang_directive` is the instruction appended to every synthesis prompt.

Split out of deep_research.py. Pure: no DR_* knob, no I/O.
"""
import re
from typing import Optional


# Script-level language sniffing. Shared with deep_research's query-diversity
# check, which asks the same question of a search query.
_LANG_RE_CYR = re.compile(r"[а-яё]", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Output language.
#
# The pipeline is English-INTERNALLY on purpose: queries, briefs and analysis stay
# English because that is where the evidence and the model's competence are (see
# the english-first design). Only the DOCUMENT the user reads is written in their
# language, and only at the writing calls — never at the search/brief calls.
# --------------------------------------------------------------------------- #
DR_LANG_NAMES = {"en": "English", "ru": "Russian"}


# Fixed scaffolding the assembler emits itself, not the model.
_DOC_LABELS = {
    "en": {"abstract": "Abstract", "toc": "Table of Contents",
           "appendix": "Source Appendix", "research": "Research",
           "generated": "Generated", "depth": "depth",
           "sources_searched": "sources searched", "pages_read": "pages read",
           "briefs_used": "briefs used", "filtered": "filtered",
           "source_quality": "Source quality", "query": "Query", "why": "Why",
           "clarify": "Suggested clarification",
           "how_made": "How this report was made",
           "gap_open": "reflection flagged a gap that this depth did not pursue"},
    "ru": {"abstract": "Аннотация", "toc": "Оглавление",
           "appendix": "Список источников", "research": "Исследование",
           "generated": "Подготовлено", "depth": "глубина",
           "sources_searched": "источников найдено", "pages_read": "страниц прочитано",
           "briefs_used": "конспектов использовано", "filtered": "отфильтровано",
           "source_quality": "Качество источников", "query": "Запрос",
           "why": "Почему", "clarify": "Уточняющий вопрос",
           "how_made": "Как сделан этот отчёт",
           "gap_open": "рефлексия отметила пробел, который на этой глубине не отрабатывался"},
}


def _norm_out_lang(lang: Optional[str]) -> str:
    code = (lang or "en").split("-")[0].strip().lower()
    return code if code in DR_LANG_NAMES else "en"


def lang_of_text(text: str, default: str = "en") -> str:
    """Guess the document language from what the user actually typed.

    Script-level only, which is all we need for en/ru: a Cyrillic-heavy request
    means a Russian reader. Callers that KNOW the language (the Telegram bot has
    an explicit per-user setting) must pass it instead of guessing.
    """
    chars = [c for c in (text or "") if c.isalpha()]
    if not chars:
        return default
    cyr = sum(1 for c in chars if "Ѐ" <= c <= "ӿ")
    return "ru" if cyr / len(chars) >= 0.3 else default


def _lab(out_lang: str, key: str) -> str:
    return _DOC_LABELS.get(_norm_out_lang(out_lang), _DOC_LABELS["en"]).get(key, key)


def _lang_directive(out_lang: str) -> str:
    """The instruction appended to every writing prompt.

    Emphatic on purpose: the evidence handed to the model is entirely in English,
    and a model reading English briefs under an English system prompt will write
    English unless told otherwise in the strongest terms.
    """
    code = _norm_out_lang(out_lang)
    if code == "en":
        return ""
    name = DR_LANG_NAMES[code]
    return (
        f"\n\nOUTPUT LANGUAGE — THIS OVERRIDES EVERYTHING ELSE IN THIS PROMPT:\n"
        f"Write the entire document in {name}. Every heading, every sentence, every "
        f"table cell, every caption and the abstract must be in {name}. The evidence "
        f"below is in English; you are TRANSLATING its content as you write, not "
        f"copying its language. Do NOT emit an English draft, do NOT add an English "
        f"version alongside, and do NOT explain that you are writing in {name}.\n"
        # The report skeleton above says "write in clear English Markdown" and lists
        # its sections with English headings. Told only "write in Russian", the model
        # split the difference and emitted a Russian body under "# Executive Summary".
        # Name the conflict and resolve it explicitly.
        f"The prompt above fixes the SECTION STRUCTURE using English headings and may "
        f"say to write in English — that wording is about structure and order only, "
        f"and this instruction overrides it. Translate every one of those headings "
        f"into {name}, keeping the same sections in the same order. No heading may "
        f"remain in English.\n"
        f"Keep UNTRANSLATED, verbatim: URLs, domain names, source titles, author "
        f"names, mathematical notation and equations, code, and established technical "
        f"terms that have no accepted {name} equivalent (give the {name} term first "
        f"with the English in parentheses on first use).\n")


# Russian counts agree with their noun, and the report shows three of them in one
# line. Left alone, a run that read one page printed "1 страниц прочитано" and a
# run with two briefs printed "2 конспектов использовано" -- small, and directly
# in the reader's eye on the first line of the document.
_RU_PLURAL = {
    "источников найдено": ("источник найден", "источника найдено", "источников найдено"),
    "страниц прочитано": ("страница прочитана", "страницы прочитано", "страниц прочитано"),
    "конспектов использовано": ("конспект использован", "конспекта использовано",
                                "конспектов использовано"),
}


_EN_SINGULAR = {
    "sources searched": "source searched",
    "pages read": "page read",
    "briefs used": "brief used",
}


def count_label(n: int, label: str, out_lang: str = "en") -> str:
    """"3 pages read" / "1 страница прочитана" -- the noun agrees with the count."""
    if _norm_out_lang(out_lang) != "ru":
        if abs(int(n)) == 1 and label in _EN_SINGULAR:
            return "1 " + _EN_SINGULAR[label]
        return "%d %s" % (n, label)
    forms = _RU_PLURAL.get(label)
    if not forms:
        return "%d %s" % (n, label)
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        form = forms[0]
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        form = forms[1]
    else:
        form = forms[2]
    return "%d %s" % (n, form)
