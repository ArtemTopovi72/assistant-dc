"""A garbled sign that survives its own repair needs a new seed, not a bigger box.

Live 2026-09-18 (journey 25): "замени нижнюю надпись на «только до пятницы»"
came back "только до пятниИЦЬ" on TWO consecutive attempts, byte-identical.
repair_text() only enlarges the lettering box and restates the spelling in
the caption -- both geometric -- while draw_agent.run() holds the seed fixed
across attempts on purpose (so an edit only moves its own box). For stray
decorative text that already re-rolls once the wording fix visibly fails to
help (see "stray lettering survived the wording"); a garbled READING had no
matching escalation, so a seed that garbles this string kept garbling it
forever within the round budget. Now a text failure repeating on the SAME
element after it was already box-repaired once re-rolls the seed too.

Run: venv/Scripts/python.exe tests/test_garbled_text_reseed.py
"""
import os, sys, json, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import draw_agent as D
import ideogram as G

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")


def layout(text="только до пятницы"):
    els = [{"desc": "a red poster", "text": "", "x": 0.05, "y": 0.05, "w": 0.9, "h": 0.9},
           {"desc": "the bottom line", "text": text, "x": 0.30, "y": 0.80, "w": 0.4, "h": 0.06}]
    return G.normalize_layout({"background": "a red poster", "medium": "graphic design",
                               "elements": els})


class _CTX:
    def is_cancelled(self): return False


renders_seed = []
reads = ["kzqxjw fnbpr", "kzqxjw fnbpr", "только до пятницы"]


def fake_render(ctx, lay, *, width, height, seed, on_progress=None, emit=None, **_kw):
    renders_seed.append(seed)
    return f"/tmp/render{len(renders_seed)}.png"


_real_render, _real_crit = D.render_without_collage, D.critique
_real_llm = sys.modules.get("llm")
try:
    D.render_without_collage = fake_render
    D.critique = lambda ctx, img, lay, evidence=None: {"ok": True, "score": 9, "problems": [],
                                                        "ops": [], "source": "vision"}
    fake_llm = types.SimpleNamespace(
        analyze_image_with_llm=lambda ctx, **kw: json.dumps(
            {"strings": [reads[min(len(renders_seed), len(reads)) - 1]]}))
    sys.modules["llm"] = fake_llm
    res = D.run(_CTX(), "a poster", layout=layout(), rounds=2, seed=42)
finally:
    D.render_without_collage, D.critique = _real_render, _real_crit
    if _real_llm is not None:
        sys.modules["llm"] = _real_llm

check("three attempts were made (garbled, garbled, fixed)", len(renders_seed) == 3, renders_seed)
check("the FIRST repair (geometry only) keeps the same seed",
      renders_seed[0] == renders_seed[1], renders_seed)
check("the SECOND repair, same string still garbled, rerolls the seed",
      renders_seed[1] != renders_seed[2], renders_seed)
check("the loop stops once the text reads correctly", res["stopped"] == "ok", res["stopped"])

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
