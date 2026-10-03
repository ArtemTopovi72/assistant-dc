"""Lettering must be painted in the user's alphabet, not translated to English.

"нарисуй плакат с надписью «Добро пожаловать»" came back showing WELCOME.

Two independent causes, both fixed here:

  1. graph._translate_to_english rewrote the whole turn before anything saw it,
     so the layout planner was handed "Welcome". The quoted span is not a
     description of the request — it IS the artwork.
  2. the Ideogram planner prompt said "Write every field in English, whatever
     language the request is in", which covers the `text` field too. Even an
     untranslated prompt (the GUI path) came back with English lettering.

The protection is deliberately DETERMINISTIC rather than a prompt instruction:
the translator's output passes an ASCII gate (>90%), so a model that correctly
kept the Cyrillic literal would have its whole answer thrown away and the turn
would stay untranslated. An ASCII placeholder keeps both the literal and the gate.

No LLM and no GPU: the translator is driven through a scripted stub.

Run: venv/Scripts/python.exe tests/test_lettering_language.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import graph as G
import ideogram as I

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


# A drawing request is the model's read (agent/intent.py `wants`).
import intent
intent.STUB = lambda t: ({"needs_tool": True, "wants": ["generate_image"]}
                         if t.startswith(("нарисуй", "draw")) else None)

print("=" * 70)
print("1. Quoted lettering survives the trip through English")
print("=" * 70)
QUOTED = {
    "«Добро пожаловать»": 'нарисуй плакат с надписью «Добро пожаловать»',
    '"Кафе Роза"':        'нарисуй вывеску с текстом "Кафе Роза"',
    "“Открыто”":          'нарисуй табличку “Открыто” на двери',
}
for literal, msg in QUOTED.items():
    protected, keep = G._protect_literals(msg)
    check(f"{literal}: the literal is replaced by a placeholder",
          literal not in protected and "⟦KEEP0⟧" in protected, protected)
    check(f"{literal}: the placeholder is pure ASCII apart from the brackets",
          all(ord(c) < 128 for c in "KEEP0"), "")
    restored = G._restore_literals(protected, keep)
    check(f"{literal}: restoring gives the message back exactly",
          restored == msg, restored)

print()
print("=" * 70)
print("2. The ASCII gate still passes once the literal is protected")
print("=" * 70)
# This is the subtle half. The gate measures the TRANSLATION; with the Cyrillic
# hidden behind a placeholder a normal English translation scores ~100% ASCII,
# so it is accepted and the literal comes back afterwards. Without the
# placeholder the model must either translate the literal (the bug) or return
# Cyrillic and be rejected (the turn stays Russian).
msg = 'нарисуй плакат с надписью «Добро пожаловать»'
protected, keep = G._protect_literals(msg)
english = "draw a poster with the inscription ⟦KEEP0⟧"
ratio = sum(ch.isascii() for ch in english) / len(english)
check("a translation carrying the placeholder is >90% ASCII", ratio > 0.9, f"{ratio:.2%}")
final = G._restore_literals(english, keep)
check("the restored prompt carries the RUSSIAN lettering",
      "«Добро пожаловать»" in final, final)
check("...and is otherwise English",
      "draw a poster" in final and "нарисуй" not in final, final)

print()
print("=" * 70)
print("3. Unquoted lettering, but only on a drawing request")
print("=" * 70)
protected, keep = G._protect_literals("нарисуй плакат с надписью Добро пожаловать")
check("an unquoted inscription is protected on a draw request",
      "Добро пожаловать" not in protected and keep, protected)
check("...and comes back on restore",
      "Добро пожаловать" in G._restore_literals(protected, keep))

# The gate that keeps this from firing everywhere.
for msg in ("убери надпись сверху", "что написано на этой картинке?",
            "переведи надпись на фото"):
    protected, keep = G._protect_literals(msg)
    check(f"not a draw request, left alone: {msg!r}", not keep, keep)

print()
print("=" * 70)
print("4. The whole translate step, driven end to end")
print("=" * 70)
class _Ctx: pass

# NOTE: patch the LLM at its own module (llm.call_llm_simple), not at whichever
# module happens to call it. The translate/reply-match functions moved from
# graph to graph_language; a patch on graph.call_llm_simple silently stopped
# reaching them, which is the failure mode where the stub dies and the suite
# still passes.
import llm as _LLMMOD
_calls = []
def _fake_llm(ctx, system, user, **kw):
    _calls.append(user)
    # A cooperative model: translates the prose, copies the placeholder through.
    return user.replace("нарисуй плакат с надписью", "draw a poster with the inscription")

_real = _LLMMOD.call_llm_simple
_LLMMOD.call_llm_simple = _fake_llm
try:
    out = G._translate_to_english(_Ctx(), 'нарисуй плакат с надписью «Добро пожаловать»')
finally:
    _LLMMOD.call_llm_simple = _real

check("the model never saw the Cyrillic literal",
      _calls and "Добро пожаловать" not in _calls[0], _calls)
check("the translated turn keeps the lettering in Russian",
      "«Добро пожаловать»" in out, out)
check("the instruction itself is English", "draw a poster" in out, out)

# An English turn must not be touched at all.
_calls.clear()
_LLMMOD.call_llm_simple = _fake_llm
try:
    same = G._translate_to_english(_Ctx(), 'draw a poster reading "Welcome"')
finally:
    _LLMMOD.call_llm_simple = _real
check("an ASCII turn short-circuits with no LLM call", not _calls, _calls)
check("...and is returned unchanged", same == 'draw a poster reading "Welcome"', same)

print()
print("=" * 70)
print("5. A preserved literal reaches the lettering extractor")
print("=" * 70)
# ideogram.requested_strings is what forces the string into a text element. It
# only ever sees the post-translation prompt, which is why the protection above
# has to happen first.
prompt_ru = 'draw a poster with the inscription «Добро пожаловать»'
got = I.requested_strings(prompt_ru)
check("the Russian literal is extracted as lettering",
      got == ["Добро пожаловать"], got)
prompt_bad = "draw a poster with the inscription Welcome"
check("...whereas the translated version yields the English string",
      I.requested_strings(prompt_bad) == [], I.requested_strings(prompt_bad))

print()
print("=" * 70)
print("6. The planner is told to keep `text` in the user's alphabet")
print("=" * 70)
src = I._PLANNER_PROMPT if hasattr(I, "_PLANNER_PROMPT") else ""
if not src:
    import inspect
    src = inspect.getsource(I)
check("the prompt exempts `text` from the English rule",
      'EXCEPT' in src and '"text"' in src, "")
check("...and says so with a concrete example",
      "Добро пожаловать" in src, "")
check("the old unconditional rule is gone",
      "Write every field in English, whatever language the request is in.\n" not in src, "")

_p, _k = G._protect_literals("видео: внук спрашивает 'бабуль, а пирожки скоро?', don't translate")
check("single-quoted lines survive translation, apostrophes do not start one",
      list(_k.values()) == ["'бабуль, а пирожки скоро?'"], _k)

_p, _k = G._protect_literals("нарисуй логотип для кофейни Зерно")
check("a brand name after 'логотип' is kept, not transliterated", list(_k.values()) == ["Зерно"], _k)
check("a city in a plain drawing is still translated", G._protect_literals("нарисуй кота в Москве")[1] == {}, "")

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
