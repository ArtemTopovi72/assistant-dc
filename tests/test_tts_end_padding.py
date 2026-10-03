"""The end-of-phrase padding must reach the model, on every surface.

This Russian F5 fine-tune swallows the end of the phrase. F5 sizes its output
from the BYTE LENGTH of gen_text, the estimate runs short, and the last word is
never generated. A run of trailing periods inflates that estimate so the clipped
tail eats silent dots instead of speech. It is the only fix that has ever
worked, and it has been deleted or regex-swept several times -- after which all
three surfaces (desktop app, Telegram, Skyrim) start clipping at once.

So this pins the whole chain: the setting, the single place it is applied, the
three call sites that reach that place, and -- the part a source scan cannot
show -- the actual gen_text batch that f5_tts hands to the model.

No model is loaded and nothing touches the GPU.

Run: venv/Scripts/python.exe tests/test_tts_end_padding.py
"""
import inspect
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import logging; logging.basicConfig(level=logging.CRITICAL)

import audio as A
import config as C

OK = BAD = 0
MIN_DOTS = 12          # 15 is shipped; below 12 the ending audibly clips again


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:220])


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    return open(os.path.join(ROOT, name), encoding="utf-8").read()


# ── the setting itself ───────────────────────────────────────────────────────
pad = C.TTS_END_PADDING
check("the padding is configured at all", bool(pad), repr(pad))
check("it is a run of at least %d periods" % MIN_DOTS,
      set(pad) == {"."} and len(pad) >= MIN_DOTS, repr(pad))

# ── applied exactly once, in the one function every path goes through ────────
src = inspect.getsource(A.preprocess_text_for_synthesis)
check("preprocess_text_for_synthesis appends it", "TTS_END_PADDING" in src)
check("with a hard-coded fallback if the env var is cleared",
      '"." * 15' in src or "'.' * 15" in src, src[-400:])

# Nowhere else: a second append is path-dependent and audible as a long pause.
# mantella/ exists only in a local checkout, never in git.
_MANTELLA = "mantella/mantella_f5_server.py"
_HAVE_MANTELLA = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _MANTELLA))
for mod in ("agent/graph.py", "bot/tg_transport.py", "gui/gui_madhouse_tab.py") + ((_MANTELLA,) if _HAVE_MANTELLA else ()):
    code = chr(10).join(l for l in read(mod).splitlines()
                        if not l.lstrip().startswith("#"))
    check("%s does NOT append the padding itself" % mod,
          "TTS_END_PADDING" not in code)

# ── every surface reaches it ─────────────────────────────────────────────────
# preprocess_text_for_synthesis is called from exactly one place: synthesis.
# So "does this surface pad?" is the same question as "does it synthesise
# through synth_single_segment?".
ssrc = inspect.getsource(A.synth_single_segment)
check("synth_single_segment preprocesses the text",
      "preprocess_text_for_synthesis" in ssrc)
check("and hands the PREPROCESSED text to infer_process, unmodified",
      "infer_process(" in ssrc
      and "processed_text," in ssrc.split("infer_process(")[1][:200]
      and "processed_text =" not in ssrc.split("processed_text =", 1)[1],
      ssrc.split("infer_process(")[1][:200])

for surface, mod in (("the desktop app", "agent/graph.py"),
                     ("Telegram", "bot/tg_transport.py")) + ((("Skyrim (Mantella)", _MANTELLA),) if _HAVE_MANTELLA else ()):
    check("%s synthesises through synth_single_segment" % surface,
          "synth_single_segment" in read(mod))

# ── behaviour: what actually comes out ───────────────────────────────────────
ctx = types.SimpleNamespace()
out = A.preprocess_text_for_synthesis(ctx, "Привет, мир", apply_stress=False)
check("a normal phrase comes back padded", out.endswith(pad), repr(out))

# Re-synthesising already-padded text must not stack two runs, and a phrase that
# simply ends in a period must not end up one dot longer than the rest.
again = A.preprocess_text_for_synthesis(ctx, out, apply_stress=False)
check("padding does not stack on re-entry",
      again.endswith(pad) and not again.endswith(pad + "."), repr(again[-40:]))
dotted = A.preprocess_text_for_synthesis(ctx, "Конец.", apply_stress=False)
check("a trailing period is absorbed, not added to",
      dotted == "Конец" + pad, repr(dotted))

check("empty input stays empty (nothing to pad)",
      A.preprocess_text_for_synthesis(ctx, "   ") == "")

# ── the part a source scan cannot show: the model's own gen_text ─────────────
# f5_tts splits gen_text into batches and synthesises each one separately, so
# what matters is the LAST batch -- that is the one whose tail gets clipped.
# chunk_text splits on punctuation followed by WHITESPACE, which is why the run
# is unspaced: it stays welded to the text it is protecting.
from f5_tts.infer.utils_infer import chunk_text

for label, phrase in (
        ("one sentence", "Степан пошёл в банк"),
        ("several sentences", "Первое. Второе! Третье? И наконец четвёртое"),
        ("long enough to be split", "Это довольно длинная реплика, " * 12),
):
    gen = A.preprocess_text_for_synthesis(ctx, phrase, apply_stress=False)
    batches = chunk_text(gen, max_chars=135)
    check("%s: the final batch the model sees still carries the padding" % label,
          batches and batches[-1].endswith(pad),
          (len(batches), batches[-1][-40:] if batches else None))
    check("%s: the padding is not orphaned into a batch of its own" % label,
          batches and batches[-1].rstrip(".").strip() != "",
          batches[-1] if batches else None)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
