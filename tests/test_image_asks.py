"""What the request asks to SEE is checked one thing at a time, and a picture missing one is redrawn.

The whole-picture judge gave Lenin-as-Doomguy 9/10 «perfectly captures»: he stood on the lava
instead of waist-deep in it, no hand was on a demon's neck, and red sunglasses stood in for
ruby eyes. Asked one at a time the same vision model said no to all three (2026-10-26).
"""
import os, sys, tempfile, threading, types
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from PIL import Image
import image_generate as G
import models

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

tmp = tempfile.mkdtemp(prefix="asks_")
def pic(name):
    p = os.path.join(tmp, name); Image.new("RGB", (64, 64), (120, 80, 40)).save(p); return p

def Ctx(**deps):
    c = models.Context(models=None, transcription_cache={}, cache_file=None,
                       asr_lock=threading.Lock(), tts_lock=threading.Lock())
    c.deps = types.SimpleNamespace(**deps)
    c.vision_model = lambda *a, **k: '{"verdict": "success", "score": 9, "reason": "perfectly captures it"}'
    return c

# The whole-picture judge says 9/10; a thing asked for is not there -> redraw, saying it plainly.
lava = ("Is he waist-deep in lava?", "He is waist-deep in lava, the lava surface at his waist")
out = G.evaluate_image(Ctx(_missing_asks=lambda ctx, p, goal: [lava]), pic("a.png"),
                       "Lenin waist-deep in lava", "p", "", 4, 1.0, 1024, 768)
check("a missing ask overrides the judge's success", out["verdict"] == "refine" and out["score"] <= 6, out)
check("the redraw states what was missing", "waist-deep in lava" in out.get("prompt_patch", ""), out)
check("the reply can say what is not there", "waist-deep" in out.get("reason", ""), out)
out = G.evaluate_image(Ctx(_missing_asks=lambda ctx, p, goal: []), pic("b.png"),
                       "a cat", "p", "", 4, 1.0, 1024, 768)
check("nothing missing -> the judge's verdict stands", out["verdict"] == "success" and out["missing"] == 0, out)

# Of the takes, the one missing the fewest asked-for things goes out -- not simply the last one:
# a retry for the missing hand can lose the lava the take before had.
takes = iter([pic("t1.png"), pic("t2.png"), pic("t3.png")])
missing = iter([1, 2, 1])
prompts = []
def draw(**k):
    prompts.append(k["prompt"]); return next(takes)
def judge(*a, **k):
    m = next(missing)
    return {"verdict": "refine", "score": 6, "reason": f"{m} missing", "missing": m,
            "prompt_patch": "Both hands are on the demons."}
got = G.generate_image_with_refinement(Ctx(generate_image_with_comfy=draw, evaluate_image=judge), "Lenin in lava")
check("the take with the fewest missing goes out", got["path"].endswith("t1.png"), got)
check("the retry carries what was missing", "Both hands are on the demons." in prompts[1], prompts)

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
