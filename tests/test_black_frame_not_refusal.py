"""A black render is not a refusal card, and a quoted chat is not lettering.

Live 2026-09-14 14:41-14:51: seven character renders in a row came back as
«Не получилось нарисовать». The log said "Ideogram refusal card detected";
none was the (grey) card -- each was a black frame (mean 1-7), rendered from
a forwarded chat whose ten «...» quotes had all become lettering strips.
"""
import os, sys, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import numpy as np
from PIL import Image
import ideogram as I

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

tmp = tempfile.mkdtemp(prefix="blackframe_")
def frame(name, base, noise, seed=1, mono=False):
    rng = np.random.default_rng(seed)
    shape = (544, 960, 1) if mono else (544, 960, 3)
    a = np.clip(base + rng.normal(0, noise, shape), 0, 255).astype("uint8")
    a = np.repeat(a, 3, axis=2) if mono else a
    p = os.path.join(tmp, name); Image.fromarray(a).save(p); return p

black = frame("black.png", 3, 3, mono=True)   # the live failures: mean~3, std~3, few colours
grey = frame("grey.png", 128, 8, mono=True)   # the real card: flat grey, a few tones
night = frame("night.png", 30, 60, 2)     # a dark but drawn scene: wide spread
day = frame("day.png", 120, 60, 3)

check("a black frame is NOT the refusal card", not I.is_refusal_card(black))
check("the grey card still is", I.is_refusal_card(grey))
check("a dark scene is neither", not I.is_refusal_card(night) and not I.is_black_frame(night))
check("a black frame is a black frame", I.is_black_frame(black))
check("the grey card is not a black frame", not I.is_black_frame(grey))
check("a normal picture is neither", not I.is_refusal_card(day) and not I.is_black_frame(day))

# a black frame gets one re-roll on a new seed; a refusal is not raised for it
import comfy_client
calls = []
def fake_submit(ctx, workflow, **kw):
    calls.append(workflow[I.NODE_NOISE]["inputs"]["noise_seed"])
    return black if len(calls) == 1 else day
orig = comfy_client._submit_and_poll
comfy_client._submit_and_poll = fake_submit
try:
    cap = {"high_level_description": "x", "style_description": "photo",
           "compositional_deconstruction": {"background": "b", "elements": []}}
    out = I.generate(None, "a cat", caption=cap, seed=7)
finally:
    comfy_client._submit_and_poll = orig
check("a black frame is re-rolled once", len(calls) == 2, calls)
check("...on a different seed", len(set(calls)) == 2, calls)
check("...and the second render is delivered", out == day, out)

# ten quoted phrases in a forwarded chat are prose, not ten lettering strips
chat = ("Ах, Степан. Ты показал мне «худшую версию Матрицы». Текст был такой: "
        "«скинь фото», «Разогрев через Систему», «Риск увольнения», «обсуждение проекта», "
        "«Теневой муж», «Зеркальный удар», «уходи от него ко мне», «Разогрев, а не штурм», "
        "«скину ссылку на тест»")
check("a quoted conversation yields no lettering", I.requested_strings(chat) == [], I.requested_strings(chat))
check("a real lettering request still does",
      I.requested_strings('a neon sign reading "CAFE ROSA"') == ["CAFE ROSA"])
check("...up to the cap", len(I.requested_strings('signs saying "A1", "B2", "C3", "D4"')) == 4)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)
