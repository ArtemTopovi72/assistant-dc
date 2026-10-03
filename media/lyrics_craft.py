"""✨ Better lyrics: a draft is checked by rules and by the model, then revised,
until it passes or the rounds run out; the best version wins.

Owner 10-03: the model writes decent lyrics when it is pushed again and again
to keep the metre and the rhymes -- so push it, but with real checks, not its
own opinion of itself: «с валидированными рифмами, без тупых рифм типа свет и
рассвет». The approach follows the lyric-editing literature: revise a draft
against explicit constraints (REFFLY, arXiv 2409.00292), count syllables per
line, and keep rule-based validation over the model's output.

The rules (Russian is the main case; English is checked more loosely):
  * rhyme    -- the clausula: from the stressed vowel to the end of the line,
               read phonetically (е/э, ё/о, я/а, ю/у, и/ы merge; a final
               voiced consonant is devoiced; ь/ъ drop). An open masculine
               rhyme («моя / я») also needs the consonant before the vowel.
               Stress comes from the voice's accentor (voice/stress.py): RUAccent,
               with silero-stress deciding homographs, run on the whole line for
               context; without it the last two vowels stand in for it.
  * banned   -- the same word, one word inside the other (свет/рассвет,
               видеть/ненавидеть), and the worn pairs (любовь/кровь, розы/
               морозы, ночь/прочь...). Paired infinitives are noted as weak.
  * rhythm   -- in each stanza the stressed syllables of the longer words
               should fall on the same beat (binary feet: even or odd
               syllables); a line that breaks it is named.
  * length   -- lines that rhyme should be about as long (±2 syllables).
  * form     -- a later verse keeps the first verse's syllable shape (same
               melody), repeated choruses stay identical, the chorus has a short
               hook line (under 10 words, first or last), no word closes lines
               in two different stanzas.
  * sense    -- the model reads it as an editor would (the Pattison / Berklee
               craft): meaning, show-don't-tell, verse/chorus/bridge roles, the
               hook, singability, clichés, filler.
✍️ writing plans first (hook, story per section, concrete images, metre,
rhyme scheme -- decompose-then-write, DECRIM), drafts several versions and
keeps the best by the same checks (over-generate and rank, PoeLM), then
polishes it.
"""
from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger("assistant.lyrics")

ROUNDS = 3
LONG_LINE = 13              # syllables: a sung line longer than this is crammed
_VOWELS_RU = "аеёиоуыэюя"
_VOWELS = _VOWELS_RU + "aeiouy"
_TAG = re.compile(r"^\s*[\[(].*[\])]\s*$")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*")

_WORN = [
    {"любовь", "кровь", "вновь", "морковь", "бровь"},
    {"розы", "морозы", "слёзы", "слезы", "грёзы", "грезы", "угрозы"},
    {"ночь", "прочь", "дочь", "помочь", "превозмочь"},
    {"мечты", "цветы", "ты", "красоты", "пустоты", "высоты"},
    {"огонь", "ладонь", "конь"},
    {"судьба", "борьба", "мольба"},
    {"навсегда", "никогда", "года", "всегда", "звезда"},
    {"тебя", "любя", "себя", "меня", "огня"},
    {"сердце", "дверца", "сердца", "дверце"},
    {"грусть", "пусть"},
    {"fire", "desire", "higher"},
    {"heart", "apart", "start"},
    {"love", "above", "of"},
    {"night", "light", "tonight", "bright", "right"},
    {"pain", "rain", "again"},
    {"soul", "whole", "control"},
]

_DEVOICE = str.maketrans("бвгджз", "пфктшс")
_VOWEL_MERGE = str.maketrans("ёэяюы", "оеаии")


# ── stress ───────────────────────────────────────────────────────────────────
ACCENT_STUB = None          # suites: ACCENT_STUB(line) -> the line with «+» marks
_ACC = None
_ACC_LOCK = threading.Lock()
_PLUS_WORD = re.compile(r"[A-Za-zА-Яа-яЁё+]+(?:-[A-Za-zА-Яа-яЁё+]+)*")


def _accentor(ctx):
    """The voice's own accentor: RUAccent, with silero-stress deciding the
    homographs (voice/stress.py, bench 2026-10-02) -- or None."""
    global _ACC
    acc = getattr(getattr(ctx, "models", None), "accentor", None)
    if acc is not None:
        return acc
    with _ACC_LOCK:
        if _ACC is None:
            try:
                from stress import BilingualAccentor
                from config import RUACCENT_MODEL, RUACCENT_WORKDIR, RUACCENT_DEVICE
                _ACC = BilingualAccentor(ru_model=RUACCENT_MODEL, ru_workdir=RUACCENT_WORKDIR,
                                         device=RUACCENT_DEVICE, enable_english=False)
            except Exception as exc:
                logger.info("lyrics: no accentor (%s) -- approximate rhymes", exc)
                _ACC = False
        return _ACC or None


def _accent_line(ctx, line: str, words: list) -> list:
    """The line's words with «+» before the stressed vowel, accented as a whole
    line so a homograph (замо́к / за́мок) is read in its context; [] when the
    stress is unknown."""
    if ACCENT_STUB is not None:
        out = ACCENT_STUB(line)
    else:
        if os.getenv("F5_TEST_RUN") or not words:
            return []
        acc = _accentor(ctx)
        if acc is None:
            return []
        try:
            out = acc(" ".join(words))
        except Exception:
            return []
    got = _PLUS_WORD.findall(out or "")
    if [w.replace("+", "").lower() for w in got] != [w.lower() for w in words]:
        return []                       # tokenised differently: no guessing
    return got


def _syllables(word: str) -> int:
    return sum(1 for ch in word.lower() if ch in _VOWELS)


def _stress_index(plus_word: str) -> int:
    """0-based syllable the «+» marks, -1 when there is none."""
    i = plus_word.find("+")
    if i < 0:
        return -1
    return sum(1 for ch in plus_word[:i].lower() if ch in _VOWELS)


# ── one line ─────────────────────────────────────────────────────────────────
def _clause(word: str, stress: int) -> str:
    """The rhyme part of the last word, read phonetically."""
    w = word.lower().replace("-", "")
    vpos = [i for i, ch in enumerate(w) if ch in _VOWELS]
    if not vpos:
        return w
    if stress < 0 or stress >= len(vpos):
        stress = max(0, len(vpos) - 2) if len(vpos) > 1 else 0
    start = vpos[stress]
    tail = w[start:]
    if start + 1 == len(w) and start > 0:          # open masculine: the consonant before counts
        tail = w[start - 1:]
    tail = tail.replace("ь", "").replace("ъ", "").replace("й", "и")
    tail = tail.translate(_VOWEL_MERGE)
    if tail and tail[-1] in "бвгджз":
        tail = tail[:-1] + tail[-1].translate(_DEVOICE)
    return tail


def _lines(text: str) -> list:
    """Stanzas of sung lines: a blank line or a [tag] line starts a new one."""
    return [st for _, st in _sections(text)]


def _section_kind(tag: str) -> str:
    t = tag.lower()
    if re.search(r"chorus|припев|refrain|hook", t):
        return "chorus"
    if re.search(r"verse|куплет", t):
        return "verse"
    return ""


def _sections(text: str) -> list:
    """[(kind, [(line_no, line)])]: kind is "verse", "chorus" or "" from the
    nearest [tag] above (a blank line keeps it: a second stanza of a verse)."""
    out, cur, kind = [], [], ""
    for n, raw in enumerate((text or "").splitlines(), 1):
        if not raw.strip() or _TAG.match(raw):
            if cur:
                out.append((kind, cur))
                cur = []
            if raw.strip():
                kind = _section_kind(raw)
            continue
        cur.append((n, raw.strip()))
    if cur:
        out.append((kind, cur))
    return out


def _row(ctx, n: int, line: str):
    """One sung line read for sound: syllables, stressed beats, the rhyme part
    of its last word (and the consonant before it, for a rich rhyme)."""
    ws = _WORD.findall(line)
    if not ws:
        return None
    got = _accent_line(ctx, line, ws)
    stressed = bool(got)
    pw = got if stressed else ws
    syl, beats, pos = 0, [], 0
    for w, p in zip(ws, pw):
        k = _syllables(w)
        s = _stress_index(p) if stressed else -1
        if k > 1 and s >= 0:
            beats.append(pos + s)
        pos += k
        syl += k
    last, plast = ws[-1], pw[-1]
    if stressed:
        idx = [_stress_index(plast)]
    else:                                  # unknown stress: the last or the one before
        k = _syllables(last)
        idx = [i for i in (k - 1, k - 2) if i >= 0]
    clauses = list(dict.fromkeys(_clause(last, i) for i in idx))
    supports = list(dict.fromkeys(_support(last, i) for i in idx))
    return {"n": n, "text": line, "syl": syl, "beats": beats, "last": last,
            "stressed": stressed, "clauses": clauses, "supports": supports}


def _support(word: str, stress: int) -> str:
    """The sound right before the rhyme part (опорный звук): when it matches
    too, the rhyme is rich -- «луна/стена» is plain, «луна/струна» and
    «туман/обман» rich (Russian prosody; rhyme scored by phonetic distance)."""
    w = word.lower().replace("-", "").replace("ь", "").replace("ъ", "")
    vpos = [i for i, ch in enumerate(w) if ch in _VOWELS]
    if not vpos:
        return ""
    if stress < 0 or stress >= len(vpos):
        stress = max(0, len(vpos) - 2) if len(vpos) > 1 else 0
    start = vpos[stress]
    if start + 1 == len(w) and start > 0:          # open: the consonant is in the clause
        start -= 1
    return w[start - 1].translate(_DEVOICE).translate(_VOWEL_MERGE) if start > 0 else ""


def _rich(a: dict, b: dict) -> bool:
    return any(x and x == y for x in a.get("supports", []) for y in b.get("supports", []))


def _rhymes(a: dict, b: dict) -> bool:
    return any(_clauses_rhyme(x, y) for x in a["clauses"] for y in b["clauses"])


def _clauses_rhyme(ca: str, cb: str) -> bool:
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    # A near rhyme: the same vowels and at most one consonant apart.
    va = [c for c in ca if c in _VOWELS]
    vb = [c for c in cb if c in _VOWELS]
    if va != vb or abs(len(ca) - len(cb)) > 1:
        return False
    diff = sum(1 for x, y in zip(ca, cb) if x != y) + abs(len(ca) - len(cb))
    return diff <= 1 and len(ca) >= 3


def _banned(a: str, b: str):
    """Why a rhyme pair is lazy, or ''."""
    x, y = a.lower().replace("ё", "е"), b.lower().replace("ё", "е")
    if x == y:
        return "same_word"
    if len(x) >= 3 and len(y) >= 3 and (x.endswith(y) or y.endswith(x)):
        return "same_root"
    for group in _WORN:
        g = {w.replace("ё", "е") for w in group}
        if x in g and y in g:
            return "worn"
    if re.search(r"(ть|ться|ти)$", x) and re.search(r"(ть|ться|ти)$", y):
        return "verbs"
    return ""


_SCHEMES = {4: [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)), ((1, 3),)]}


def analyse(text: str, ctx=None) -> dict:
    """{"issues": [...], "score": 0..10, "stressed": bool} for the lyric."""
    sections = _sections(text)
    stanzas = [st for _, st in sections]
    info, n_lines, n_stressed = [], 0, 0
    for st in stanzas:
        rows = [r for r in (_row(ctx, n, line) for n, line in st) if r]
        n_lines += len(rows)
        n_stressed += sum(r["stressed"] for r in rows)
        if rows:
            info.append(rows)

    issues, penalty, rich = [], 0.0, 0
    for rows in info:
        if len(rows) < 2:
            continue
        # which pairs should rhyme: the scheme that rhymes most, or neighbours
        if len(rows) == 4:
            schemes = _SCHEMES[4]
        else:
            schemes = [tuple((i, i + 1) for i in range(0, len(rows) - 1, 2)),
                       tuple((i, i + 2) for i in range(0, len(rows) - 2, 4))
                       + tuple((i + 1, i + 3) for i in range(0, len(rows) - 3, 4))]
        schemes = [s for s in schemes if s]
        best = max(schemes, key=lambda s: (sum(_rhymes(rows[i], rows[j]) for i, j in s), len(s)))
        for i, j in best:
            a, b = rows[i], rows[j]
            if a["text"].lower() == b["text"].lower():
                continue                               # a repeated refrain line
            if not _rhymes(a, b):
                issues.append({"kind": "no_rhyme", "lines": (a["n"], b["n"]),
                               "words": (a["last"], b["last"])})
                penalty += 1.5
                continue
            why = _banned(a["last"], b["last"])
            if why:
                issues.append({"kind": why, "lines": (a["n"], b["n"]), "words": (a["last"], b["last"])})
                penalty += 0.7 if why == "verbs" else 2.0
            elif _rich(a, b):
                rich += 1
            if abs(a["syl"] - b["syl"]) > 2:
                issues.append({"kind": "length", "lines": (a["n"], b["n"]), "syl": (a["syl"], b["syl"])})
                penalty += 0.5
        for r in rows:
            # one line = one sung phrase: past ~13 syllables it no longer fits a
            # bar and the singer crams it (Suno / topline practice: 6-10, at most 12)
            if r["syl"] > LONG_LINE:
                issues.append({"kind": "long_line", "lines": (r["n"],), "syl": (r["syl"], r["syl"])})
                penalty += 0.4
        if all(r["stressed"] for r in rows):
            beats = [b for r in rows for b in r["beats"]]
            if len(beats) >= 4:
                even = sum(1 for b in beats if b % 2 == 0)
                parity = 0 if even * 2 >= len(beats) else 1
                for r in rows:
                    off = [b for b in r["beats"] if b % 2 != parity]
                    if r["beats"] and len(off) * 2 > len(r["beats"]):
                        issues.append({"kind": "rhythm", "lines": (r["n"],)})
                        penalty += 0.5
    f_issues, f_pen = _form_issues(sections, info)
    issues += f_issues
    penalty += f_pen
    # rich rhymes earn back a little: up to one point
    penalty -= min(1.0, 0.25 * rich)
    return {"issues": issues, "score": max(0.0, min(10.0, round(10 - penalty, 1))),
            "stressed": bool(n_lines) and n_stressed == n_lines, "rich": rich}


def _form_issues(sections: list, info: list) -> tuple:
    """Song form, beyond single rhymes (songwriting practice: the melody repeats,
    so must the words' shape):
      verse_shape   a later verse's line is not as long as the first verse's line
                    in the same place (it is sung to the same melody, ±2 syllables)
      chorus_drift  a repeated chorus is worded differently from the first one
      hook_long     the chorus has no short line to hold the hook: neither its
                    first nor its last line is 10 words or fewer
      end_repeat    the same word closes lines in two different stanzas
    """
    rows_of = {}
    flat = [r for rows in info for r in rows]
    for r in flat:
        rows_of[r["n"]] = r
    issues, pen = [], 0.0
    verses = [st for kind, st in sections if kind == "verse"]
    if len(verses) > 1:
        first = verses[0]
        for other in verses[1:]:
            for (n1, _), (n2, _) in zip(first, other):
                a, b = rows_of.get(n1), rows_of.get(n2)
                if a and b and abs(a["syl"] - b["syl"]) > 2:
                    issues.append({"kind": "verse_shape", "lines": (n1, n2), "syl": (a["syl"], b["syl"])})
                    pen += 0.4
    choruses = [st for kind, st in sections if kind == "chorus"]
    if choruses:
        ref = [l.lower() for _, l in choruses[0]]
        for ch in choruses[1:]:
            got = [l.lower() for _, l in ch]
            if got and got != ref[:len(got)]:
                issues.append({"kind": "chorus_drift", "lines": (ch[0][0],)})
                pen += 0.5
        first = choruses[0]
        if len(first) >= 2 and all(len(_WORD.findall(l)) > 10 for _, l in (first[0], first[-1])):
            issues.append({"kind": "hook_long", "lines": (first[0][0],)})
            pen += 0.5
    seen = {}
    chorus_lines = {n for st in choruses for n, _ in st}
    for si, rows in enumerate(info):
        for r in rows:
            if r["n"] in chorus_lines:
                continue
            w = r["last"].lower().replace("ё", "е")
            if w in seen and seen[w][0] != si and len(w) > 2:
                issues.append({"kind": "end_repeat", "lines": (seen[w][1], r["n"]), "words": (r["last"], r["last"])})
                pen += 0.3
            seen.setdefault(w, (si, r["n"]))
    return issues, pen


_ISSUE_EN = {
    "no_rhyme": "lines {a} and {b} do not rhyme («{x}» / «{y}»)",
    "same_word": "lines {a} and {b} rhyme a word with itself («{x}»)",
    "same_root": "lines {a} and {b}: «{x}» / «{y}» is one word inside the other, not a rhyme",
    "worn": "lines {a} and {b}: «{x}» / «{y}» is a worn-out rhyme",
    "verbs": "lines {a} and {b}: two verbs in the same form («{x}» / «{y}») is a weak rhyme",
    "length": "lines {a} and {b} differ in length ({s1} vs {s2} syllables)",
    "rhythm": "line {a} breaks the rhythm of its stanza",
    "verse_shape": "line {b} is sung to the melody of line {a} but has {s2} syllables, not about {s1}",
    "chorus_drift": "the chorus from line {a} is worded differently from the first chorus",
    "hook_long": "the chorus (line {a}) has no short hook line: make its first or last line 10 words or fewer",
    "end_repeat": "lines {a} and {b} end on the same word «{x}»",
    "long_line": "line {a} is too long to sing in one breath ({s1} syllables): one line, one phrase",
}


_ISSUE_RU = {
    "no_rhyme": "строки {a} и {b} не рифмуются («{x}» / «{y}»)",
    "same_word": "строки {a} и {b}: слово рифмуется само с собой («{x}»)",
    "same_root": "строки {a} и {b}: «{x}» / «{y}» — одно слово внутри другого, это не рифма",
    "worn": "строки {a} и {b}: «{x}» / «{y}» — затёртая рифма",
    "verbs": "строки {a} и {b}: два глагола в одной форме («{x}» / «{y}») — слабая рифма",
    "length": "строки {a} и {b} разной длины ({s1} и {s2} слогов)",
    "rhythm": "строка {a} сбивает ритм куплета",
    "verse_shape": "строка {b} поётся на мелодию строки {a}, но в ней {s2} слогов, а не около {s1}",
    "chorus_drift": "припев со строки {a} написан иначе, чем первый припев",
    "hook_long": "в припеве (строка {a}) нет короткой строки-хука: первая или последняя строка — до 10 слов",
    "end_repeat": "строки {a} и {b} кончаются одним и тем же словом «{x}»",
    "long_line": "строка {a} слишком длинная, на одном дыхании не спеть ({s1} слогов): одна строка — одна фраза",
}


def describe(issue: dict, lang: str = "en") -> str:
    ln = issue["lines"]
    w = issue.get("words") or ("", "")
    s = issue.get("syl") or (0, 0)
    table = _ISSUE_RU if lang == "ru" else _ISSUE_EN
    return table[issue["kind"]].format(a=ln[0], b=ln[-1], x=w[0], y=w[1], s1=s[0], s2=s[1])


# ── the model ────────────────────────────────────────────────────────────────
LLM_STUB = None             # suites: LLM_STUB(role, system, user) -> str
MIN_P = float(os.getenv("LYRICS_MIN_P", "0.05") or 0)
_CREATIVE = ("draft", "revise", "lines")


def _call(ctx, role: str, system: str, user: str, *, temperature: float, max_tokens: int,
          schema=None) -> str:
    if LLM_STUB is not None:
        return LLM_STUB(role, system, user)
    if os.getenv("F5_TEST_RUN"):
        return ""                      # suites that do not stub it: no model call
    from llm import call_llm_simple
    kw = {"json_schema": schema} if schema else {}
    if role in _CREATIVE and MIN_P:
        # min-p keeps high temperature coherent and off the cliché tokens
        # (Nguyen et al. 2024, arxiv 2407.01082)
        kw["sampling"] = {"min_p": MIN_P}
    return call_llm_simple(ctx, system, user, temperature=temperature, max_tokens=max_tokens,
                           prefill="<think></think>", **kw) or ""


def _lang_name(text: str) -> str:
    return "Russian" if re.search(r"[А-Яа-яЁё]", text or "") else "English"


_RULES = (
    "Rules of the craft: every rhyme is a real sound rhyme on the stressed vowel; no "
    "word rhymed with itself or with a word containing it (свет/рассвет, видеть/"
    "ненавидеть); no worn pairs (любовь/кровь, розы/морозы, ночь/прочь, мечты/цветы, "
    "fire/desire, heart/apart); avoid rhyming two verbs in the same form; one steady "
    "metre per section (the stresses fall on the same beats in every line), rhyming "
    "lines of about the same length; one line = one sung phrase of 6-12 "
    "syllables (a breath; no line runs on into the next); concrete images, a clear thought, no filler words "
    "put in for the rhyme; natural word order; it must be easy to sing.")

_CRITIC = (
    "You are a demanding song editor (the Pattison / Berklee school). Read the lyric "
    "(lines are numbered) and list only real problems of MEANING and SOUND: unclear or "
    "illogical lines; filler or padding words put in for the rhyme or the count; "
    "clichés and stock images; forced word order; clumsy or hard-to-sing phrases "
    "(consonant pile-ups, a stressed syllable on a weak word); emotions NAMED instead "
    "of shown through concrete, sensory detail; verses that do not move the story or "
    "add new detail; a chorus that is not the emotional summary or has no memorable "
    "hook line; a bridge that brings no new angle; images that contradict each other. "
    "Name the line number in each problem. Return JSON {\"score\": 0-10, "
    "\"problems\": [\"...\"]}; an empty list when it is good.")

_CRITIC_SCHEMA = {"type": "object", "properties": {
    "score": {"type": "integer"}, "problems": {"type": "array", "items": {"type": "string"}}},
    "required": ["score", "problems"], "additionalProperties": False}


def _numbered(text: str) -> str:
    return "\n".join(f"{n}: {l}" for n, l in enumerate(text.splitlines(), 1))


def critique(ctx, text: str) -> tuple:
    """(score 0..10, [problems]) from the model, written in the lyric's own
    language (they are shown to the user); (7, []) when it cannot say."""
    from utils import safe_json_from_llm
    try:
        system = _CRITIC + f" Write the problems in {_lang_name(text)}."
        raw = _call(ctx, "critic", system, _numbered(text), temperature=0.0, max_tokens=600,
                    schema=_CRITIC_SCHEMA)
        data = safe_json_from_llm(raw, ["score", "problems"]) or {}
        probs = [str(p).strip() for p in (data.get("problems") or []) if str(p).strip()][:8]
        return max(0, min(10, int(data.get("score") or 0))), probs
    except Exception:
        logger.warning("lyrics critic failed", exc_info=True)
        return 7, []


def _clean(raw: str) -> str:
    t = (raw or "").strip()
    t = re.sub(r"^```\w*\n|\n```$", "", t).strip()
    # a model that copies the numbering back: «3: строка» -> «строка»
    lines = [l.rstrip() for l in t.splitlines()]
    if lines and sum(bool(re.match(r"^\d+:\s", l)) for l in lines if l) * 2 > len([l for l in lines if l]):
        lines = [re.sub(r"^\d+:\s?", "", l) for l in lines]
    return "\n".join(lines).strip()


def revise(ctx, text: str, problems: list, lang: str) -> str:
    system = (
        f"You are a skilled {lang} lyricist and editor. Rewrite the song lyric to fix "
        "EVERY listed problem. Keep its theme, story, language and section tags ([verse], "
        "[chorus]...); a line no problem names stays word for word. When a rhyme pair "
        "is wrong, rewrite the weaker line of the pair, not both. Keep every repeated "
        "chorus identical. " + _RULES + " Output only the full lyric, no comments, no "
        "line numbers.")
    user = ("Lyric (lines numbered as the problems name them):\n" + _numbered(text)
            + "\n\nProblems to fix:\n" + "\n".join(f"- {p}" for p in problems))
    return _clean(_call(ctx, "revise", system, user, temperature=0.7,
                        max_tokens=max(800, len(text) * 2)))


DRAFTS = max(1, int(os.getenv("LYRICS_DRAFTS", "3") or 3))

_PLAN = (
    "You plan a song before it is written (decompose first, then write). For the "
    "theme below, in {lang}, give: the HOOK -- the title line, under 10 words, that "
    "sums up the song's feeling; the story: what verse 1 shows, what verse 2 adds "
    "(new detail, time moves on), what the bridge turns to (a new angle); 6-10 "
    "concrete sensory images (things one can see, hear, touch -- object writing) to "
    "SHOW the feeling instead of naming it; the metre (e.g. 4-foot trochee) and the "
    "rhyme scheme (ABAB or AABB). Plain text, short.")


def plan(ctx, topic: str, lang: str) -> str:
    try:
        return _clean(_call(ctx, "plan", _PLAN.format(lang=lang), "Theme: " + topic,
                            temperature=0.7, max_tokens=500))
    except Exception:
        return ""


def draft(ctx, topic: str, lang: str, the_plan: str = "") -> str:
    system = (
        f"You are a skilled songwriter. Write an original song lyric in {lang} on the "
        "theme below: [verse 1] 4 lines, [chorus] 4 lines, [verse 2] 4 lines, [chorus], "
        "[bridge] 2-4 lines, [chorus]. Pick one metre and keep it; rhyme ABAB or AABB in "
        "every section; verse 2 has the same syllable shape as verse 1 (it is sung to the "
        "same melody). Verses SHOW through concrete detail, the chorus states the feeling "
        "and holds the hook in its first or last line, every chorus is identical; a hook may "
        "come back with a small variation (repetition with variation makes it stick). "
        + _RULES + " Output only the lyric with its section tags.")
    user = "Theme: " + topic + (("\n\nPlan:\n" + the_plan) if the_plan else "")
    return _clean(_call(ctx, "draft", system, user, temperature=0.9, max_tokens=1200))


def write(ctx, topic: str, lang: str, on_round=None) -> dict:
    """✍️ A theme -> a lyric: plan, several drafts (over-generate and rank, as
    PoeLM), the two best by the checks meet head to head (a judge picking one of
    two is steadier than its 0-10 scores), then the polish loop."""
    the_plan = plan(ctx, topic, lang)
    scored = []
    for _ in range(DRAFTS):
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            break
        d = draft(ctx, topic, lang, the_plan)
        if not d:
            continue
        score = _total(analyse(d, ctx), critique(ctx, d)[0])
        logger.info("lyrics draft scored %.2f", score)
        scored.append((score, d))
    if not scored:
        raise RuntimeError("no draft")
    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 1.5:
        best = compare(ctx, topic, scored[0][1], scored[1][1])
    return polish(ctx, best, on_round=on_round)


_JUDGE = (
    "You judge two song lyrics on the same theme as a hit songwriter would: which "
    "one is the better SONG -- a hook that sticks, images shown not told, clean "
    "rhymes, a steady singable rhythm, a story that moves. Answer with one letter: "
    "A or B.")


def _verdict(ctx, topic: str, a: str, b: str) -> str:
    raw = _call(ctx, "judge", _JUDGE, f"Theme: {topic}\n\nA:\n{a}\n\nB:\n{b}",
                temperature=0.0, max_tokens=5)
    m = re.search(r"\b([AB])\b", (raw or "").strip().upper())
    return m.group(1) if m else ""


def compare(ctx, topic: str, first: str, second: str) -> str:
    """The better of two lyrics by the model, asked both ways round (a judge
    leans to one position); `first` -- the checks' pick -- unless both say B."""
    try:
        one = _verdict(ctx, topic, first, second)
        two = _verdict(ctx, topic, second, first)
    except Exception:
        logger.warning("lyrics judge failed", exc_info=True)
        return first
    return second if (one, two) == ("B", "A") else first


_LINES = (
    "You fix one line of a {lang} song. Give {k} different versions of the line "
    "marked >>> so that it ends on a true rhyme with «{word}» (the stressed vowel and "
    "what follows sound alike, not the same word or root, not a worn pair), keeps "
    "about {syl} syllables and the same meaning in context. One version per line, "
    "nothing else.")


def fix_rhymes(ctx, text: str, rule: dict, lang: str, k: int = 3) -> str:
    """Line by line before the whole rewrite: for a pair that does not rhyme,
    a few versions of its weaker (later) line, the first that passes the rule
    check is kept (Amuse / line-level variants). The rest of the lyric is left
    as it is."""
    lines = text.splitlines()
    rows = {r["n"]: r for r in (_row(ctx, n, l) for n, l in enumerate(lines, 1)) if r}
    done = 0
    for issue in rule.get("issues", []):
        if issue["kind"] not in ("no_rhyme", "worn", "same_root", "same_word") or done >= 4:
            continue
        a_n, b_n = issue["lines"]
        a, b = rows.get(a_n), rows.get(b_n)
        if not a or not b:
            continue
        ctx_lines = "\n".join((">>> " if n == b_n else "") + l
                              for n, l in enumerate(lines, 1) if abs(n - b_n) <= 3)
        try:
            raw = _call(ctx, "lines", _LINES.format(lang=lang, k=k, word=a["last"], syl=a["syl"]),
                        ctx_lines, temperature=0.9, max_tokens=300)
        except Exception:
            continue
        for cand in _clean(raw).splitlines()[:k + 2]:
            cand = re.sub(r"^\s*(?:>>>|[-•*]|\d+[.):])\s*", "", cand).strip()
            r = _row(ctx, b_n, cand) if cand else None
            if (r and _rhymes(a, r) and not _banned(a["last"], r["last"])
                    and abs(a["syl"] - r["syl"]) <= 2):
                lines[b_n - 1] = cand
                rows[b_n] = r
                done += 1
                break
    return "\n".join(lines)


def _total(rule: dict, sense: int) -> float:
    return round(0.6 * rule["score"] + 0.4 * sense, 2)


def polish(ctx, text: str, rounds: int = ROUNDS, on_round=None) -> dict:
    """The check -> revise loop. {"text", "score", "rounds", "left": [problems],
    "original"}; the best scoring version is kept, never a worse rewrite."""
    lang = _lang_name(text)
    cur, best = text.strip(), None
    done = 0
    for r in range(rounds + 1):
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            break
        rule = analyse(cur, ctx)
        sense, probs = critique(ctx, cur)
        problems = [describe(i) for i in rule["issues"]] + probs
        total = _total(rule, sense)
        logger.info("lyrics round %d: rules %.1f sense %d -> %.2f, %d problems",
                    r, rule["score"], sense, total, len(problems))
        if best is None or total > best["score"]:
            # the user reads what is left: rule findings in their language, by the caller
            best = {"text": cur, "score": total, "left": problems,
                    "left_rules": rule["issues"], "left_sense": probs}
        if not problems or r == rounds:
            break
        if on_round:
            on_round(r + 1)
        fixed = fix_rhymes(ctx, cur, rule, lang)
        if fixed != cur:                            # the rhymes mended line by line first
            rule = analyse(fixed, ctx)
            problems = [describe(i) for i in rule["issues"]] + probs
            cur = fixed
            done += 1
            if not problems:
                continue                            # scored on the next round
        new = revise(ctx, cur, problems, lang)
        done += 1
        if not new or len(new) < len(cur) * 0.4:
            break                                   # a broken rewrite: keep what we have
        cur = new
    best.update(rounds=done, original=text.strip())
    return best


def changes_note(ctx, before: str, after: str, lang_ui: str) -> list:
    """3-7 short bullets: what changed and why, in the user's language."""
    if before.strip() == after.strip():
        return []
    ui = "Russian" if lang_ui == "ru" else "English"
    system = (f"Compare the ORIGINAL and the NEW song lyric. In {ui}, list 3-7 concrete "
              "changes, one per line, each «строка N: было → стало — зачем» style: which "
              "rhyme, rhythm or meaning problem it fixed. Plain lines starting with «• », "
              "nothing else.")
    try:
        raw = _call(ctx, "notes", system, f"ORIGINAL:\n{before}\n\nNEW:\n{after}",
                    temperature=0.2, max_tokens=600)
    except Exception:
        return []
    return [l.strip() for l in (raw or "").splitlines() if l.strip()][:7]
