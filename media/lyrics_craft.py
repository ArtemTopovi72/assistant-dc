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
  * sense    -- the model reads it as an editor would: meaning, images, logic,
               singability, clichés.
"""
from __future__ import annotations

import logging
import os
import re
import threading

logger = logging.getLogger("assistant.lyrics")

ROUNDS = 3
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
    stanzas, cur = [], []
    for n, raw in enumerate((text or "").splitlines(), 1):
        if not raw.strip() or _TAG.match(raw):
            if cur:
                stanzas.append(cur)
                cur = []
            continue
        cur.append((n, raw.strip()))
    if cur:
        stanzas.append(cur)
    return stanzas


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
    stanzas = _lines(text)
    info, n_lines, n_stressed = [], 0, 0
    for st in stanzas:
        rows = []
        for n, line in st:
            ws = _WORD.findall(line)
            if not ws:
                continue
            got = _accent_line(ctx, line, ws)
            n_lines += 1
            n_stressed += bool(got)
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
                clauses = [_clause(last, _stress_index(plast))]
            else:                              # unknown stress: the last or the one before
                k = _syllables(last)
                clauses = list(dict.fromkeys(_clause(last, i) for i in (k - 1, k - 2) if i >= 0))
            rows.append({"n": n, "text": line, "syl": syl, "beats": beats, "last": last,
                         "stressed": stressed, "clauses": clauses})
        if rows:
            info.append(rows)

    issues, penalty = [], 0.0
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
            if abs(a["syl"] - b["syl"]) > 2:
                issues.append({"kind": "length", "lines": (a["n"], b["n"]), "syl": (a["syl"], b["syl"])})
                penalty += 0.5
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
    return {"issues": issues, "score": max(0.0, round(10 - penalty, 1)),
            "stressed": bool(n_lines) and n_stressed == n_lines}


_ISSUE_EN = {
    "no_rhyme": "lines {a} and {b} do not rhyme («{x}» / «{y}»)",
    "same_word": "lines {a} and {b} rhyme a word with itself («{x}»)",
    "same_root": "lines {a} and {b}: «{x}» / «{y}» is one word inside the other, not a rhyme",
    "worn": "lines {a} and {b}: «{x}» / «{y}» is a worn-out rhyme",
    "verbs": "lines {a} and {b}: two verbs in the same form («{x}» / «{y}») is a weak rhyme",
    "length": "lines {a} and {b} differ in length ({s1} vs {s2} syllables)",
    "rhythm": "line {a} breaks the rhythm of its stanza",
}


_ISSUE_RU = {
    "no_rhyme": "строки {a} и {b} не рифмуются («{x}» / «{y}»)",
    "same_word": "строки {a} и {b}: слово рифмуется само с собой («{x}»)",
    "same_root": "строки {a} и {b}: «{x}» / «{y}» — одно слово внутри другого, это не рифма",
    "worn": "строки {a} и {b}: «{x}» / «{y}» — затёртая рифма",
    "verbs": "строки {a} и {b}: два глагола в одной форме («{x}» / «{y}») — слабая рифма",
    "length": "строки {a} и {b} разной длины ({s1} и {s2} слогов)",
    "rhythm": "строка {a} сбивает ритм куплета",
}


def describe(issue: dict, lang: str = "en") -> str:
    ln = issue["lines"]
    w = issue.get("words") or ("", "")
    s = issue.get("syl") or (0, 0)
    table = _ISSUE_RU if lang == "ru" else _ISSUE_EN
    return table[issue["kind"]].format(a=ln[0], b=ln[-1], x=w[0], y=w[1], s1=s[0], s2=s[1])


# ── the model ────────────────────────────────────────────────────────────────
LLM_STUB = None             # suites: LLM_STUB(role, system, user) -> str


def _call(ctx, role: str, system: str, user: str, *, temperature: float, max_tokens: int,
          schema=None) -> str:
    if LLM_STUB is not None:
        return LLM_STUB(role, system, user)
    from llm import call_llm_simple
    kw = {"json_schema": schema} if schema else {}
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
    "lines of about the same length; concrete images, a clear thought, no filler words "
    "put in for the rhyme; natural word order; it must be easy to sing.")

_CRITIC = (
    "You are a demanding song editor. Read the lyric (lines are numbered) and list "
    "only real problems of MEANING and SOUND: unclear or illogical lines, filler or "
    "padding, clichés, forced word order, clumsy or hard-to-sing phrases, images that "
    "contradict each other. Name the line number in each problem. Return JSON "
    "{\"score\": 0-10, \"problems\": [\"...\"]}; an empty list when it is good.")

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
        "EVERY listed problem. Keep its theme, story, language, section tags ([verse], "
        "[chorus]...) and the lines that are fine; change only what the problems need. "
        + _RULES + " Output only the full lyric, no comments, no line numbers.")
    user = ("Lyric (lines numbered as the problems name them):\n" + _numbered(text)
            + "\n\nProblems to fix:\n" + "\n".join(f"- {p}" for p in problems))
    return _clean(_call(ctx, "revise", system, user, temperature=0.7,
                        max_tokens=max(800, len(text) * 2)))


def draft(ctx, topic: str, lang: str) -> str:
    system = (
        f"You are a skilled songwriter. Write an original song lyric in {lang} on the "
        "theme below: [verse 1] 4 lines, [chorus] 4 lines, [verse 2] 4 lines, [chorus], "
        "[bridge] 2-4 lines, [chorus]. Pick one metre and keep it; rhyme ABAB or AABB in "
        "every section. The chorus carries the hook and is the most memorable part. "
        + _RULES + " Output only the lyric with its section tags.")
    return _clean(_call(ctx, "draft", system, "Theme: " + topic, temperature=0.9, max_tokens=1200))


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
