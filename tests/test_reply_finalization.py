"""The final answer must not be doubled, and must be in the user's language.

Both reproduced live in Telegram on 2026-07-29 13:18 from ONE Russian question
("Кто такой Виктор Робертович Шемякин", via the 🔍 Search button):

    I could not find any information about Viktor Robertovich Shemyakin.I could
    not find any information about Viktor Robertovich Shemyakin.

  · doubled — the same sentence twice with NO separator. The activity log shows a
    single user_msg, so it is not the debounce double-send of
    [tg-image-delivery-and-double-tap]; the model emitted it twice. The streaming
    repetition guard samples every LLM_REPEAT_CHECK_EVERY chars and cannot see a
    doubled one-liner.
  · English — the english-first pipeline asks for the reply in the user's
    language with a directive. On a plain turn that holds (13:11, same session,
    answered in Russian). After a tool chain it does not: the tool output and the
    model's own reasoning are all English, so the closing turn is English too.

Run: venv/Scripts/python.exe tests/test_reply_finalization.py
"""
import os, sys, ast, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import utils as U
import graph as G
import graph_personality as GP

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


LIVE = ("I could not find any information about Viktor Robertovich Shemyakin."
        "I could not find any information about Viktor Robertovich Shemyakin.")
ONCE = "I could not find any information about Viktor Robertovich Shemyakin."


print("=" * 70)
print("THE DOUBLED ANSWER")
print("=" * 70)

check("the live doubled reply collapses to one sentence",
      U.collapse_verbatim_repetition(LIVE) == ONCE,
      repr(U.collapse_verbatim_repetition(LIVE)))
check("a single copy is untouched", U.collapse_verbatim_repetition(ONCE) == ONCE)
check("a space between the copies is handled too",
      U.collapse_verbatim_repetition(ONCE + " " + ONCE) == ONCE)
check("three copies collapse to one",
      U.collapse_verbatim_repetition(ONCE + ONCE + ONCE) == ONCE)
check("a doubled LINE collapses",
      U.collapse_verbatim_repetition("A fairly long paragraph of text here.\n"
                                     "A fairly long paragraph of text here.")
      == "A fairly long paragraph of text here.")
check("Cyrillic doubling collapses",
      U.collapse_verbatim_repetition("Ничего не нашёл про Виктора Шемякина."
                                     "Ничего не нашёл про Виктора Шемякина.")
      == "Ничего не нашёл про Виктора Шемякина.")
check("case-insensitive duplicates collapse",
      U.collapse_verbatim_repetition("The answer is not available right now. "
                                     "the answer is not available right now.")
      == "The answer is not available right now.")

print()
print("-- and repetition that is NOT a bug survives --")
for name, text in {
    "short repeats (No. No.)": "No. No.",
    "no sentence punctuation": "Ha ha ha",
    "two different sentences": "I searched the web. I found nothing at all today.",
    "a non-adjacent echo": ("Alpha beta gamma delta epsilon zeta. Something else "
                            "entirely here. Alpha beta gamma delta epsilon zeta."),
    "a markdown list": "- item one here for you\n- item two here for you",
    "a numbered list": "1. Первый пункт с описанием\n2. Второй пункт с описанием",
    "empty": "",
}.items():
    check(f"kept: {name}", U.collapse_verbatim_repetition(text) == text,
          repr(U.collapse_verbatim_repetition(text)))

# Code is the one place a repeated line is normal and load-bearing.
code = "```python\nx = 1\nx = 1\n```"
check("a code fence is left completely alone",
      U.collapse_verbatim_repetition(code) == code)


print()
print("=" * 70)
print("THE LANGUAGE DRIFT")
print("=" * 70)

class _Ctx:
    def __init__(self): self.cancelled = False
    def is_cancelled(self): return self.cancelled

RU_Q = "Кто такой Виктор Робертович Шемякин"
RU_A = "Я не нашёл информации о Викторе Робертовиче Шемякине."

# NOTE: patch the LLM at its own module (llm.call_llm_simple), not at whichever
# module happens to call it. _match_reply_language moved from graph to
# graph_language; a patch on graph.call_llm_simple silently stopped reaching it,
# which is the failure mode where the stub dies and the suite still passes.
import llm as _LLMMOD
calls = []
def _fake_llm(ctx, system, user, **kw):
    calls.append((system, user, kw))
    return RU_A

_real = _LLMMOD.call_llm_simple
_LLMMOD.call_llm_simple = _fake_llm
try:
    calls.clear()
    out = G._match_reply_language(_Ctx(), ONCE, RU_Q)
    check("an English answer to a Russian question is translated back", out == RU_A, out)
    check("the translation call was actually made", len(calls) == 1, str(len(calls)))
    check("the target language is named", calls and "into Russian" in calls[0][0])

    # Everything below must NOT spend a call.
    for name, (answer, question) in {
        "answer already in the user's language": (RU_A, RU_Q),
        "the user wrote in English": (ONCE, "Who is Viktor Shemyakin"),
        "no original (turn was never translated)": (ONCE, ""),
        "answer too short to judge": ("OK", RU_Q),
        "empty answer": ("", RU_Q),
    }.items():
        calls.clear()
        got = G._match_reply_language(_Ctx(), answer, question)
        check(f"untouched: {name}", got == answer and not calls,
              f"{got!r} calls={len(calls)}")

    # A cancelled turn must not spend an LLM call on cosmetics.
    calls.clear()
    c = _Ctx(); c.cancelled = True
    check("a cancelled turn skips the back-translation",
          G._match_reply_language(c, ONCE, RU_Q) == ONCE and not calls)

    # Fail OPEN: a translator that errors, returns nothing, or echoes the English
    # back must never cost the user their answer.
    for name, ret in {"returns None": None, "returns empty": "   ",
                      "echoes the English back": ONCE}.items():
        _LLMMOD.call_llm_simple = lambda *a, **k: ret
        check(f"fails open when the translator {name}",
              G._match_reply_language(_Ctx(), ONCE, RU_Q) == ONCE)

    def _boom(*a, **k): raise RuntimeError("backend down")
    _LLMMOD.call_llm_simple = _boom
    check("fails open when the translator raises",
          G._match_reply_language(_Ctx(), ONCE, RU_Q) == ONCE)
finally:
    _LLMMOD.call_llm_simple = _real

# A mixed answer (Russian prose, English names/links) is already correct and must
# not be re-translated — that would mangle the names.
mixed = "Нестор Махно — Nestor Makhno (1888-1934), см. https://example.com/makhno"
check("a mixed-script answer counts as the user's language",
      G._foreign_ratio(mixed) > 0.1, str(G._foreign_ratio(mixed)))
check("a pure-English answer is detected as drifted", G._foreign_ratio(ONCE) == 0.0)
check("a Russian question is detected as non-Latin", G._foreign_ratio(RU_Q) > 0.3)
check("an English question is not", G._foreign_ratio("Who is he") < 0.3)


print()
print("=" * 70)
print("BOTH ARE WIRED INTO THE ONE FINALISATION POINT")
print("=" * 70)

# Assert the calls REASSIGN final_answer — a check that merely finds the function
# name would also pass if the result were computed and thrown away.
# These checks read the SOURCE of _finalize_answer, so they must follow it. It
# has moved twice (graph.py -> graph_personality.py -> graph_finalize.py), and
# each move would silently empty the checks if the file were named here by hand
# — an AST walk over the wrong module finds no `final_answer` assignment at all
# and every check below would fail (or, worse, a future rename could make them
# vacuously pass). Resolve the owning module from the function object instead.
_FIN = inspect.getmodule(GP._finalize_answer)
tree = ast.parse(open(_FIN.__file__, encoding="utf-8").read())
assigns = [n for n in ast.walk(tree)
           if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "final_answer"
                   for t in n.targets)]
def _assigned_from(fn_name):
    return any(isinstance(n.value, ast.Call)
               and getattr(n.value.func, "id", "") == fn_name for n in assigns)

check("final_answer is reassigned from collapse_verbatim_repetition",
      _assigned_from("collapse_verbatim_repetition"))
check("final_answer is reassigned from _match_reply_language",
      _assigned_from("_match_reply_language"))

# ...and it happens AFTER every branch that can produce an answer, or the
# last-resort/forced paths would ship unfiltered.
src = open(_FIN.__file__, encoding="utf-8").read()
i_collapse = src.index("collapse_verbatim_repetition(final_answer)")
for marker in ["forcing a closing reply", "Last resort", "final_answer = strip_textual_tool_calls"]:
    check(f"the filter runs after: {marker[:38]}", src.index(marker) < i_collapse)
# The persist is not in the same module any more: the filter runs inside
# _finalize_answer (graph_finalize), and the turn's own ctx.remember() /
# state["final_answer"] persist is in its CALLER, personality_node. So the
# ordering is now enforced across a call boundary — assert it there: the
# _finalize_answer(...) call must come before both persists in the caller's
# source. ctx.remember() appears more than once (the fast path has its own),
# so the check is on the first occurrence AFTER the finalisation call.
caller_src = inspect.getsource(inspect.getmodule(GP.personality_node))
i_final = caller_src.index("_finalize_answer(ctx, state, messages")
check("the filter runs before the answer is persisted to memory",
      caller_src.find('ctx.remember(', i_final) != -1
      and 'state["final_answer"] = final_answer' in caller_src
      and caller_src.index('state["final_answer"] = final_answer') > i_final)


print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)
