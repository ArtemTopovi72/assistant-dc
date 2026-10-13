"""Synthesis template classification: a comparison/factual topic must not be routed
to the practical-guide template.

"product" is a CATEGORY that also covers pure comparison/spec questions (two
tractor models, two phones) that are NOT a "help me buy/book one" ask. Routed to
PRACTICAL_GUIDE_PROMPT (built for "find me a plumber" style asks) such a topic
cannot be answered no matter how it is nudged, and `_think_call` burns most of
its wall-clock guard on empty (reasoning-only) calls. Two things are pinned,
independently:

  1. CLASSIFICATION — `_is_practical_request` no longer treats "product" as a
     blanket-practical category, and an explicit comparison/factual phrasing
     ("differ", "vs", "compare", "отличается", "разница", "сравнение") always
     wins over a practical category guess, so a genuine comparison question
     is never routed to the practical-guide template regardless of category.
     "local"/"shopping"/"travel" (genuinely near-always actionable) and the
     existing survey-mode / contradiction-query gate are untouched.

  2. FAIL-FAST RETRY — `_think_call` accepts a `retries` parameter. The
     practical-guide call site passes `retries=1` (one retry, not two) so a
     template that is fundamentally wrong for the topic gives up after two
     attempts instead of three, and the caller's existing "practical synthesis
     failed — falling back" path kicks in sooner.

No real LM Studio/GPU call is made — the engine's one model contact is handed in on
the context (`text_model`) to simulate the "pure reasoning, no answer" failure.

Run: venv/Scripts/python.exe tests/test_dr_synthesis_classification.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import deep_research as D
# the model's reads are stubbed; the phrases run live in bench/intent_sweep_live.py
import intent
def _kind(q, t):
    t = t.lower()
    if any(w in t for w in ("differ", "отличается", " vs ", "difference")):
        return "compare"
    return "practical" if any(w in t for w in ("buy", "book", "find me", "deals")) else "research"
intent.CHOICE_STUB = _kind
import types

import dr_calls  # noqa: F401

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}\n         {detail}")


# ─────────────────────────────────────────────────────── (a) comparison topics
print("\nCOMPARISON/FACTUAL TOPICS DO NOT GET THE PRACTICAL TEMPLATE")

check("the live-incident RU tractor comparison is NOT practical",
      not D._is_practical_request(
          "Чем трактор К-700 отличается от К-744", {"category": "product"}))

check("an equivalent EN comparison ('how does X differ from Y') is NOT practical",
      not D._is_practical_request(
          "How does a Kubota tractor differ from a John Deere",
          {"category": "product"}))

check("a 'vs' comparison phrasing is NOT practical",
      not D._is_practical_request(
          "iPhone 15 vs Galaxy S24 camera comparison", {"category": "product"}))

check("a bare 'product' category with no comparison/action wording is NOT "
      "practical (product no longer blanket-practical)",
      not D._is_practical_request(
          "The history and design philosophy of the K-700 tractor",
          {"category": "product"}))

check("comparison wording wins even if the sentence also has a practical-ish "
      "word ('price') sprinkled in",
      not D._is_practical_request(
          "How does the price difference between the K-700 and K-744 reflect "
          "their design differences", {"category": "product"}))

# ─────────────────────────────────────────────────── (b) genuinely practical
print("\nGENUINELY PRACTICAL TOPICS STILL GET THE PRACTICAL TEMPLATE")

check("'find me a plumber near Boston' (local category) is practical",
      D._is_practical_request(
          "find me a plumber near Boston", {"category": "local"}))

check("an explicit booking ask in a 'general' category is practical",
      D._is_practical_request(
          "book me a hotel room for this weekend", {"category": "general"}))

check("a 'product' category WITH an explicit buy/price ask is still practical "
      "(the category alone no longer decides it, but the action wording does)",
      D._is_practical_request(
          "where to buy the cheapest RTX 4070 right now, with links",
          {"category": "product"}))

check("shopping category stays blanket-practical",
      D._is_practical_request("best deals this week", {"category": "shopping"}))

check("travel category stays blanket-practical",
      D._is_practical_request("book me a flight to Rome", {"category": "travel"}))

# ─────────────────────────────────────────────────── (c) scholarly still wins
print("\nSCHOLARLY CATEGORY STILL OVERRIDES (untouched invariant)")

check("a legal/medical topic mentioning cost is still NOT practical",
      not D._is_practical_request(
          "what is the cost of malpractice liability", {"category": "legal"}))

# ───────────────────────────────────────────── (d) fail-fast retry behaviour
print("\nFAIL-FAST RETRY ON A TEMPLATE THAT CANNOT PRODUCE AN ANSWER")

_calls = []


def _fake_empty(ctx, system, user, temperature=0.3, max_tokens=100, force_think=True, **_kw):
    """Simulates the live incident: every call returns empty (pure reasoning,
    no final answer), regardless of budget/nudge — because the TEMPLATE is
    wrong for the topic, not because of a budget cliff."""
    _calls.append(max_tokens)
    return ""


_ctx = types.SimpleNamespace(text_model=_fake_empty)      # the model, handed in
_calls.clear()
out = D._think_call(_ctx, "SYSTEM", "USER", 1000, retries=1)
check("retries=1 gives up after 2 attempts, not 3",
      len(_calls) == 2, f"got {len(_calls)} calls: {_calls}")
check("a fully-exhausted call returns empty (not a crash)", out == "")

_calls.clear()
out = D._think_call(_ctx, "SYSTEM", "USER", 1000)
check("default retries=2 preserves the old 3-attempt behaviour for other "
      "callers (unrelated to this fix)",
      len(_calls) == 3, f"got {len(_calls)} calls: {_calls}")

# ──────────────────────────────────────── (e) survey-mode invariant untouched
print("\nEXISTING SURVEY-MODE GATE IS UNTOUCHED")

check("DR_SURVEY_MODE flag still exists and is a bool-typed config knob",
      "DR_SURVEY_MODE" in D.MANUAL_OVERRIDE_SPEC
      and D.MANUAL_OVERRIDE_SPEC["DR_SURVEY_MODE"]["type"] == "bool")

check("_PRACTICAL_CATEGORIES no longer contains 'product' but still contains "
      "the genuinely-actionable categories",
      D._PRACTICAL_CATEGORIES == {"local", "shopping", "travel"},
      D._PRACTICAL_CATEGORIES)

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)
