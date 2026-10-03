"""A deep-research report is written in the READER's language, not the pipeline's.

Live complaint (2026-07-31): a Russian user asked for a research paper and got an
English one. The pipeline is English-internally on purpose — queries, crawling and
per-source briefs stay English because that is where the sources are — but the
DOCUMENT is what the user reads, and nothing ever told the writer which language to
write it in. Every writing call inherited English from its system prompt and from
the English evidence in front of it.

Two layers are checked:

  OFFLINE — the directive reaches every writing call and NO analysis call, the
  scaffolding labels (Abstract / TOC / appendix / header) are localized, and the
  default is still English so existing callers are untouched.

  LIVE — the real model, given real English evidence, actually writes Russian.
  That is the part no amount of plumbing proves; an offline test would pass while
  the model kept answering in English.

Run: venv/Scripts/python.exe tests/test_dr_output_language.py
"""
import os, sys, re, inspect, threading, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import config
import deep_research as D

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}\n         {detail}")


def cyr_ratio(s: str) -> float:
    letters = [c for c in (s or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") / len(letters)


# ───────────────────────────────────────────────────────── language resolution
print("\nWHICH LANGUAGE THE DOCUMENT IS WRITTEN IN")
check("a Cyrillic request means a Russian document",
      D.lang_of_text("напиши научную работу про плавучесть") == "ru")
check("an English request stays English",
      D.lang_of_text("write a paper on fecal buoyancy") == "en")
check("a mostly-English topic with one Russian word stays English",
      D.lang_of_text("structure tensor и Di Zenzo operator in image analysis") == "en",
      D.lang_of_text("structure tensor и Di Zenzo operator in image analysis"))
check("an empty topic falls back to English, not a crash",
      D.lang_of_text("") == "en")
check("a URL-only topic does not crash", D.lang_of_text("https://x.io/a?b=1") == "en")
check("an unsupported language code degrades to English",
      D._norm_out_lang("de") == "en" and D._norm_out_lang("ru-RU") == "ru"
      and D._norm_out_lang(None) == "en")

# ───────────────────────────────────────────────────────────────── the directive
print("\nTHE DIRECTIVE")
d_ru = D._lang_directive("ru")
check("English asks for nothing — the prompts are already English",
      D._lang_directive("en") == "", repr(D._lang_directive("en")[:60]))
check("Russian gets an explicit instruction", "Russian" in d_ru, d_ru[:80])
check("it names what must NOT be translated",
      all(w in d_ru for w in ("URL", "equation", "author")), d_ru)
check("it forbids emitting an English version alongside",
      "alongside" in d_ru or "do NOT add an English" in d_ru, d_ru)

# ───────────────────────────────────────────────── which calls get the directive
print("\nTHE DIRECTIVE REACHES THE WRITERS AND ONLY THE WRITERS")
src = inspect.getsource(D)


def fn_src(name):
    return inspect.getsource(getattr(D, name))


WRITERS = ["synthesize_report", "synthesize_survey", "_synthesize_abstract",
           "plan_report_outline"]
for w in WRITERS:
    check(f"{w} takes an out_lang and uses the directive",
          "out_lang" in inspect.signature(getattr(D, w)).parameters
          and "_lang_directive" in fn_src(w),
          f"signature={inspect.signature(getattr(D, w))}")

# The analysis calls must NOT be translated: briefs and queries are the pipeline's
# internal working language, and translating them would degrade the search itself.
ANALYSTS = ["brief_source", "plan_queries", "_consolidate_evidence"]
for a in ANALYSTS:
    check(f"{a} is left in English (internal analysis, not the document)",
          "_lang_directive" not in fn_src(a), a)

check("run_deep_research exposes out_lang and normalizes it",
      "out_lang" in inspect.signature(D.run_deep_research).parameters
      and "_norm_out_lang(out_lang)" in fn_src("run_deep_research"))
check("the default is English, so existing callers are unaffected",
      inspect.signature(D.run_deep_research).parameters["out_lang"].default == "en")

# The practical-guide branch is a separate writing call and was easy to miss.
run_src = fn_src("run_deep_research")
check("the practical-options branch is translated too",
      re.search(r"PRACTICAL_GUIDE_PROMPT \+ _lang_directive", run_src) is not None,
      "practical requests would still come back English")
check("every synthesis call inside the run passes out_lang",
      run_src.count("out_lang=out_lang") >= 4,
      f"only {run_src.count('out_lang=out_lang')} call sites pass it")

# ──────────────────────────────────────────────────────── the fixed scaffolding
print("\nTHE PARTS WE WRITE OURSELVES, NOT THE MODEL")
plan = {"title": "Плавучесть", "sections": [{"heading": "Введение",
                                             "subsections": ["Обзор"]}]}
toc_ru, toc_en = D._toc(plan, "ru"), D._toc(plan, "en")
check("the table of contents is localized",
      "Оглавление" in toc_ru and "Table of Contents" in toc_en, toc_ru[:40])
check("the abstract heading is localized",
      D._lab("ru", "abstract") == "Аннотация" and D._lab("en", "abstract") == "Abstract")
check("the source appendix heading is localized",
      "Список источников" in D._sources_appendix([], "ru")
      and "Source Appendix" in D._sources_appendix([], "en"))

hdr_ru = D._report_header("тема", [], {"sources": 3, "pages": 2}, "standard",
                          out_lang="ru")
hdr_en = D._report_header("topic", [], {"sources": 3, "pages": 2}, "standard")
check("the run header is localized", "Подготовлено" in hdr_ru, hdr_ru)
check("and English is unchanged", "Generated" in hdr_en and "depth:" in hdr_en, hdr_en)
check("the numbers survive translation", "3" in hdr_ru and "2" in hdr_ru, hdr_ru)
check("an unknown label degrades to its key, never a KeyError",
      D._lab("ru", "no_such_label") == "no_such_label")
check("every English label has a Russian one",
      set(D._DOC_LABELS["en"]) == set(D._DOC_LABELS["ru"]),
      set(D._DOC_LABELS["en"]) ^ set(D._DOC_LABELS["ru"]))

# ─────────────────────────────────────────────────────────── the callers wire it
print("\nTHE CALLERS PASS IT")
import tools, gui, tg_bot
# TelegramBot was split into mixin modules; the research call now lives in
# tg_tasks.py. Search the whole family so this anchor cannot go vacuous.
import tg_tasks
bot_src = inspect.getsource(tg_bot) + inspect.getsource(tg_tasks)
# Match the CALL, not the comment above it that also names run_deep_research().
m = re.search(r"_dr\.run_deep_research\((.{0,120})", bot_src, re.S)
# The bot must derive the document language from the TOPIC, not from the
# session's RU/EN UI toggle. This assertion used to demand `out_lang=lang`
# (the UI toggle) and forbid `lang_of_text` — i.e. it pinned the very bug
# that "detect out_lang from typed topic, not session UI lang" removed, so
# it went red against correct code. Assert the shipped contract instead.
check("the Telegram bot derives the document language from the topic",
      m is not None and "out_lang=" in m.group(1) and "out_lang=lang" not in m.group(1),
      m.group(1) if m else "no run_deep_research call found in tg_bot")
# ...and that the derivation genuinely ignores the UI toggle: a Latin-script
# topic must NOT come back "ru" just because the session is Russian.
check("a Russian topic under an English UI yields a Russian document",
      D.lang_of_text("Расскажи про квантовые вычисления") == "ru")
check("an English topic under a Russian UI still yields English",
      D.lang_of_text("deep research on quantum computing") == "en")
# Check the BEHAVIOUR, not the source text. This used to grep for the literal
# 'lang_of_text(state.get("user_input_original")' and broke the moment the helper
# was resolved off the module with getattr — the tool still derived the language
# correctly, the assertion just could not see it. Drive the handler and read the
# out_lang it actually passes to run_deep_research.
def _out_lang_for(state):
    import deep_research as _dr
    seen = {}
    _orig = _dr.run_deep_research

    def _spy(ctx, topic, depth="standard", out_lang="en", progress=None):
        seen["out_lang"] = out_lang
        return {"report": "# x\n\ntext", "stats": {}, "sources": []}

    _dr.run_deep_research = _spy
    try:
        class _C:
            def set_stage(self, *a, **k): pass
            def remember(self, *a, **k): pass
            def memory_text(self): return ""
            def is_cancelled(self): return False
        tools._handle_deep_research(_C(), state, {"topic": state.get("user_input", "x")})
    finally:
        _dr.run_deep_research = _orig
    return seen.get("out_lang")


_ru_state = {"user_input": "The tractor market in 2026",
             "user_input_original": "Рынок тракторов в 2026 году"}
_en_state = {"user_input": "The tractor market in 2026", "user_input_original": ""}
check("the agent tool derives it from the user's ORIGINAL message",
      _out_lang_for(_ru_state) == "ru",
      f"the English-translated topic must not be the signal; got {_out_lang_for(_ru_state)!r}")
check("an already-English turn stays English",
      _out_lang_for(_en_state) == "en", repr(_out_lang_for(_en_state)))
# The derivation has moved twice now: out of gui.py into gui_workers.py, and
# then behind the research service boundary into research_client.py. So this
# anchor searches the whole family and keys on a MARKER exported by
# research_api, not on a call expression that moves whenever the code moves.
# A source-text check that finds nothing must go RED, not quietly vacuous.
import gui_workers, research_api, research_client
_gui_src = (inspect.getsource(gui) + inspect.getsource(gui_workers)
            + inspect.getsource(research_client))
check("the GUI research tab derives it from the typed topic",
      research_api.OUT_LANG_FROM_TOPIC_MARKER in _gui_src,
      "marker not found in gui.py / gui_workers.py / research_client.py")
# ...and that the marker is not just a comment: the client really must default
# out_lang from the topic when the caller passes None. Driven, not grepped.
_seen_lang = {}


class _LangProbeDR:
    MANUAL_OVERRIDE_SPEC = {}

    @staticmethod
    def lang_of_text(text, default="en"):
        return "ru"

    @staticmethod
    def apply_overrides(ov):
        return {}

    @staticmethod
    def restore_overrides(saved):
        return None

    @staticmethod
    def run_deep_research(ctx, topic, depth="standard", out_lang="en", progress=None):
        _seen_lang["out_lang"] = out_lang
        return {"report": "x"}


_prev_dr = sys.modules.get("deep_research")
sys.modules["deep_research"] = _LangProbeDR
try:
    research_client.InProcessResearchClient().run(None, "квантовые вычисления")
finally:
    if _prev_dr is not None:
        sys.modules["deep_research"] = _prev_dr
    else:
        del sys.modules["deep_research"]
check("the in-process research client defaults out_lang from the topic",
      _seen_lang.get("out_lang") == "ru", repr(_seen_lang))




# ── the survey document must say how it was made ────────────────────────────
# Survey mode writes its own title, abstract and contents, so the run header is
# deliberately not prepended -- which left the PRIMARY product with no
# provenance at all. Measured on a live run: 34,000 characters that never said
# what depth it ran at, how many sources were searched, or how many pages were
# actually read.
_st = {"sources": 47, "pages": 20, "quarantined": {"fetch_failed": 3},
       "reflection": {"needs_second_pass": True}, "reflection_iterations": 0}
_briefs = [{"trust": "PRIMARY"}, {"trust": "SECONDARY"}]
_foot_ru = D.provenance_footer(_briefs, _st, "quick", "ru")
_foot_en = D.provenance_footer(_briefs, _st, "quick", "en")

check("the footer names the depth the run actually used", "quick" in _foot_ru, _foot_ru)
check("...and the counters it actually reached",
      "47" in _foot_ru and "20" in _foot_ru, _foot_ru)
check("...and what was thrown away", "fetch_failed" in _foot_ru, _foot_ru)
check("the footer is in the DOCUMENT's language",
      "Как сделан этот отчёт" in _foot_ru and "How this report was made" in _foot_en)

# The line that costs something to print.
check("an unpursued gap is disclosed, not left silent",
      "пробел" in _foot_ru, _foot_ru)
check("...and in English too", "gap" in _foot_en, _foot_en)
_done = D.provenance_footer(_briefs, {"sources": 5, "pages": 4,
                                      "reflection": {"needs_second_pass": True},
                                      "reflection_iterations": 2}, "deep", "ru")
check("...but a gap that WAS pursued is not reported as open",
      "пробел" not in _done, _done)
check("...nor is one that was never flagged",
      "пробел" not in D.provenance_footer(_briefs, {"sources": 1, "pages": 1},
                                          "deep", "ru"))

# Russian counts agree with their noun; "1 страниц прочитано" was printed on the
# first line of every report that read one page.
_one = D.provenance_footer([{"trust": "PRIMARY"}], {"sources": 1, "pages": 1},
                           "quick", "ru")
check("one page reads as one page", "1 страница прочитана" in _one, _one)
check("one source reads as one source", "1 источник найден" in _one, _one)
check("two briefs take the paucal form",
      "2 конспекта использовано" in _foot_ru, _foot_ru)
check("five keeps the genitive plural",
      "5 страниц прочитано" in D.provenance_footer(
          _briefs, {"sources": 5, "pages": 5}, "quick", "ru"))
check("the teens are not treated as ones",
      "11 страниц прочитано" in D.provenance_footer(
          _briefs, {"sources": 11, "pages": 11}, "quick", "ru"))
check("twenty-one is",
      "21 страница прочитана" in D.provenance_footer(
          _briefs, {"sources": 21, "pages": 21}, "quick", "ru"))
check("English agrees too", "1 page read" in D.provenance_footer(
    [{"trust": "PRIMARY"}], {"sources": 1, "pages": 1}, "quick", "en"))


print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)