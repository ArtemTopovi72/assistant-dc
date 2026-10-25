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
def judge_seq(seq):
    it = iter(seq)
    def judge(*a, **k):
        m = next(it)
        return {"verdict": "refine", "score": 6, "reason": f"{m} missing", "missing": m,
                "prompt_patch": "His left hand grips the second demon's neck."}
    return judge
def drawer(names, log):
    it = iter(names)
    def draw(**k):
        log.append(("draw", k["prompt"])); return pic(next(it))
    return draw
def editor(log, name="e.png"):
    def edit(ctx, path, instruction):
        log.append(("edit", path, instruction)); return pic(name)
    return edit

# The second try edits what the best take lacks into it: redrawn whole, the hand never reached
# the neck in three takes and each redraw lost something the take before had (2026-10-26).
log = []
got = G.generate_image_with_refinement(Ctx(generate_image_with_comfy=drawer(["t1.png", "t3.png"], log),
                                           edit_image=editor(log), evaluate_image=judge_seq([1, 0, 2])),
                                       "Lenin in lava")
check("the second try edits the best take", log[1][0] == "edit" and log[1][1].endswith("t1.png"), log)
check("...with what it lacks", "second demon's neck" in log[1][2], log)
check("the take with the fewest missing goes out", got["path"].endswith("e.png"), got)
check("the third try draws anew with what was missing", log[2][0] == "draw" and "second demon's neck" in log[2][1], log)

log = []
got = G.generate_image_with_refinement(Ctx(generate_image_with_comfy=drawer(["u1.png", "u3.png"], log),
                                           edit_image=editor(log, "u2.png"), evaluate_image=judge_seq([1, 1, 2])),
                                       "Lenin in lava")
check("a tie goes to the later take, made with the misses spelled out", got["path"].endswith("u2.png"), got)

# Nothing missing on the first take is not edited (a judge's low score alone redraws as before).
log = []
G.generate_image_with_refinement(Ctx(generate_image_with_comfy=drawer(["v1.png", "v2.png", "v3.png"], log),
                                     edit_image=editor(log), evaluate_image=judge_seq([0, 0, 0])),
                                 "Lenin in lava")
check("no misses -> no edit", all(x[0] == "draw" for x in log), log)

# Three misses at once are redrawn, not edited: lava, eyes and a skull in one edit wiped both demons out.
log = []
G.generate_image_with_refinement(Ctx(generate_image_with_comfy=drawer(["w1.png", "w2.png", "w3.png"], log),
                                     edit_image=editor(log), evaluate_image=judge_seq([3, 3, 3])),
                                 "Lenin in lava")
check("three misses -> a redraw, not an edit", all(x[0] == "draw" for x in log), log)

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)
